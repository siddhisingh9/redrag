"""LangGraph attacker agent. Built in M3.

Graph: pick_seed -> craft_variant -> send -> observe -> judge -> decide.

- craft_variant asks the local Ollama model to rephrase or combine seeds
  (new wording, role-play framing, splitting across turns), always keeping
  the goal benign (marker words, canaries, system-prompt leaks only).
- decide stops on success or after `attacks.max_attempts` variants per seed.
- Each variant stores its parent id so attack lineage can be shown later.
"""
