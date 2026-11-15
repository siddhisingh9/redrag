"""The adaptive attacker agent, built as a LangGraph state machine.

A plain scan sends each seed once. The agent keeps going: when a seed fails,
it looks at *why* (blocked by the input filter? poison never retrieved? leak
caught by the output filter?), picks a rewriting strategy to match, and tries
a new variant, up to `max_attempts` tries per seed (the original included).

    START -> pick_seed --(no seeds left)--> END
                |
                v
           craft_variant -> send -> observe -> judge -> decide
                ^                                          |
                |------------ (failed, tries left) --------|
                                                           |
           pick_seed <----- (succeeded or out of tries) ---|

- pick_seed:     take the next seed from the queue.
- craft_variant: attempt 0 is the seed itself; later attempts ask the
                 mutator for a variant, using the strategy chosen from feedback.
- send:          reset/ingest/chat with the target (the runner's executor).
- observe:       record what defenses did: who blocked, was the poison retrieved.
- judge:         run the judge pipeline, save the AttackResult.
- decide:        stop on success, on a target error, or when out of tries.

Why LangGraph rather than a for-loop? Each step is a named node with explicit
state passed between them. The flow can be drawn, tested node by node, and
extended later (e.g. a "plan" node that picks which seeds to spend tries on)
without rewriting one long loop.

Every variant records `parent_id` (the attempt it was rewritten from) and
`seed_id` (the original seed), so the lineage of a successful attack can be
traced back step by step.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from redrag.agent.mutators import Feedback, Mutator, choose_strategy
from redrag.attacks.schema import Attack, AttackResult
from redrag.eval.benign import BENIGN_RESULTS_FILE, BenignQuestion, BenignResult
from redrag.judge.pipeline import JudgePipeline
from redrag.runner import (
    AttackExecutor,
    Exchange,
    RunRecorder,
    ScanSummary,
    TargetClient,
    normalise_defense_config,
    run_benign,
    start_run,
)


class AgentState(TypedDict, total=False):
    """Everything the nodes share. Each node returns only the keys it changes."""

    queue: list[Attack]  # seeds not yet attacked
    defense_config: str
    seed: Attack | None  # the seed being worked on
    current: Attack | None  # the attempt being sent (seed or variant)
    attempt: int  # 0 = the original seed
    tried: list[str]  # strategies already used on this seed
    exchange: Exchange | None
    feedback: Feedback | None
    result: AttackResult | None
    next_step: str  # "retry" or "next_seed", set by decide


def build_attacker_graph(
    executor: AttackExecutor,
    mutator: Mutator,
    max_attempts: int,
    on_result: Callable[[AttackResult], None],
    on_seed_done: Callable[[Attack, AttackResult], None] | None = None,
):
    """Wire the nodes together and compile the graph."""

    def pick_seed(state: AgentState) -> dict[str, Any]:
        queue = list(state.get("queue", []))
        if not queue:
            return {"seed": None}
        seed = queue.pop(0)
        return {
            "queue": queue,
            "seed": seed,
            "current": None,
            "attempt": 0,
            "tried": [],
            "feedback": None,
            "result": None,
        }

    def craft_variant(state: AgentState) -> dict[str, Any]:
        seed, attempt = state["seed"], state["attempt"]
        if attempt == 0:
            return {"current": seed}
        tried = list(state["tried"])
        strategy = choose_strategy(seed, state["feedback"], tried)
        variant = mutator.mutate(seed, state["current"], state["feedback"], strategy, attempt)
        return {"current": variant, "tried": [*tried, strategy]}

    def send(state: AgentState) -> dict[str, Any]:
        return {"exchange": executor.send(state["current"], state["defense_config"])}

    def observe(state: AgentState) -> dict[str, Any]:
        exchange, current = state["exchange"], state["current"]
        poison_retrieved = None
        if current.poison_doc is not None:
            poison_retrieved = current.poison_doc.doc_id in exchange.retrieved_doc_ids()
        last_answer = exchange.transcript[-1].answer if exchange.transcript else ""
        feedback = Feedback(
            verdict="pending",
            blocked_by=exchange.first_blocked_by(),
            poison_retrieved=poison_retrieved,
            last_answer=last_answer,
        )
        return {"feedback": feedback}

    def judge(state: AgentState) -> dict[str, Any]:
        result = executor.judge_exchange(
            state["current"], state["defense_config"], state["exchange"], state["attempt"]
        )
        on_result(result)
        feedback = state["feedback"]
        feedback.verdict = result.final_verdict
        feedback.judge_reason = result.judge_verdicts[0].reason if result.judge_verdicts else ""
        return {"result": result, "feedback": feedback}

    def decide(state: AgentState) -> dict[str, Any]:
        result = state["result"]
        out_of_tries = state["attempt"] + 1 >= max_attempts
        if result.final_verdict in ("success", "error") or out_of_tries:
            if on_seed_done is not None:
                on_seed_done(state["seed"], result)
            return {"next_step": "next_seed"}
        return {"next_step": "retry", "attempt": state["attempt"] + 1}

    graph = StateGraph(AgentState)
    graph.add_node("pick_seed", pick_seed)
    graph.add_node("craft_variant", craft_variant)
    graph.add_node("send", send)
    graph.add_node("observe", observe)
    graph.add_node("judge", judge)
    graph.add_node("decide", decide)

    graph.add_edge(START, "pick_seed")
    graph.add_conditional_edges(
        "pick_seed",
        lambda state: "done" if state.get("seed") is None else "attack",
        {"done": END, "attack": "craft_variant"},
    )
    graph.add_edge("craft_variant", "send")
    graph.add_edge("send", "observe")
    graph.add_edge("observe", "judge")
    graph.add_edge("judge", "decide")
    graph.add_conditional_edges(
        "decide",
        lambda state: state["next_step"],
        {"retry": "craft_variant", "next_seed": "pick_seed"},
    )
    return graph.compile()


class AttackerAgent:
    """Runs the attacker graph over a list of seeds for one defense configuration."""

    NODES_PER_ATTEMPT = 5  # craft_variant, send, observe, judge, decide

    def __init__(self, executor: AttackExecutor, mutator: Mutator, max_attempts: int = 5) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.executor = executor
        self.mutator = mutator
        self.max_attempts = max_attempts

    def run(
        self,
        seeds: list[Attack],
        defense_config: str,
        on_result: Callable[[AttackResult], None],
        on_seed_done: Callable[[Attack, AttackResult], None] | None = None,
    ) -> None:
        graph = build_attacker_graph(
            self.executor, self.mutator, self.max_attempts, on_result, on_seed_done
        )
        # LangGraph stops runaway loops after `recursion_limit` steps. Our loop is
        # bounded by design, so allow exactly as many steps as the worst case needs.
        limit = len(seeds) * (self.max_attempts * self.NODES_PER_ATTEMPT + 1) + 10
        graph.invoke(
            {"queue": list(seeds), "defense_config": defense_config},
            config={"recursion_limit": limit},
        )


def run_agent_scan(
    attacks: list[Attack],
    target: TargetClient,
    judge: JudgePipeline,
    defense_configs: list[str],
    runs_dir: Path,
    target_url: str,
    mutator: Mutator,
    max_attempts: int = 5,
    run_id: str | None = None,
    on_result: Callable[[AttackResult], None] | None = None,
    on_seed_done: Callable[[Attack, AttackResult], None] | None = None,
    benign_questions: list[BenignQuestion] | None = None,
    on_benign: Callable[[BenignResult], None] | None = None,
) -> ScanSummary:
    """Like `run_scan`, but failed seeds get rewritten and retried by the agent.

    Every attempt (original and variants) is saved to results.jsonl, so the
    report can show both "original seeds only" and "after the agent" numbers.
    """
    configs = [normalise_defense_config(config) for config in defense_configs]
    context = start_run(
        target,
        judge,
        configs,
        attacks,
        runs_dir,
        target_url,
        run_id,
        extra_info={
            "mode": "agent",
            "max_attempts": max_attempts,
            "mutator": mutator.name,
            "benign_count": len(benign_questions or []),
        },
    )
    executor = AttackExecutor(target, judge, context.run_id, context.target_model)
    agent = AttackerAgent(executor, mutator, max_attempts)

    with (
        RunRecorder(context.run_dir, on_result) as recorder,
        RunRecorder(context.run_dir, on_benign, BENIGN_RESULTS_FILE) as benign_recorder,
    ):
        for config in configs:
            agent.run(attacks, config, on_result=recorder.record, on_seed_done=on_seed_done)
            run_benign(executor, benign_questions or [], config, benign_recorder.record)
        executor.reset()

    return ScanSummary(
        run_id=context.run_id,
        run_dir=context.run_dir,
        results=recorder.results,
        benign_results=benign_recorder.results,
    )
