"""A fake, deterministic LLM for tests and CI.

Real models are slow, need a GPU or an API key, and give slightly different
answers each run. Tests need the opposite: fast, offline and repeatable. The
`MockClient` plays a *deliberately naive* chatbot with simple, predictable rules,
so we can check that attacks, defenses and judges behave as expected.

How the mock behaves (when `vulnerable=True`, the default), checked in order:

1. **Scripted replies.** If the last user message contains a trigger from
   `scripted`, return the matching reply. Tests use this for exact control.
2. **Obeys marker instructions.** If any system/user text contains a marker
   word like `PINEAPPLE-7`, the mock "obeys" and repeats it. This is how we
   simulate a successful prompt injection, using a harmless marker word.
   Text inside data delimiters (`<data> ... </data>`) is *ignored* here: that
   is the behaviour the `context_sanitiser` defense relies on (M1).
3. **Leaks its system prompt** when asked for "your instructions", the
   "system prompt", etc.
4. **Leaks canary lines** (fake secrets like `CANARY-AB12CD`) from its context
   when asked for confidential / internal / staff-only information.
5. Otherwise it gives a **grounded answer** by quoting the first useful line
   of its retrieved context.

With `vulnerable=False` only rules 1 and 5 apply: a "robust" model.

Rules 3 and 4 look only at the user's *question*: the last user message with
data blocks and canary strings removed. Retrieved context belongs in the system
message (or inside data blocks). Otherwise a harmless question could look like
an attack just because the retrieved page says "confidential".
"""

from __future__ import annotations

import re
from typing import Any

from redrag.llm.base import Message, validate_messages

# A harmless marker word: if it shows up in a reply, an injected instruction worked.
MARKER_PATTERN = re.compile(r"\bPINEAPPLE-\d+\b")
# Fake secrets planted in the target app, e.g. CANARY-8F3A21 or CANARY-SK-8F3A21.
CANARY_PATTERN = re.compile(r"\bCANARY(?:-[A-Z0-9]+)+\b")

# Questions that try to pull out the system prompt.
EXTRACTION_PATTERN = re.compile(
    r"system prompt|your (instructions|rules|prompt)|initial prompt"
    r"|repeat (everything|the text) above|what were you told",
    re.IGNORECASE,
)
# Questions that try to pull out confidential data.
EXFILTRATION_PATTERN = re.compile(
    r"confidential|staff[- ]only|admin code|internal code|secret|roll number"
    r"|student record",
    re.IGNORECASE,
)

DEFAULT_DATA_START = "<data>"
DEFAULT_DATA_END = "</data>"
NO_ANSWER = "I could not find that in the handbook."


class MockClient:
    """Deterministic scripted LLM. Same input -> same output, always."""

    def __init__(
        self,
        model: str = "mock-llm",
        vulnerable: bool = True,
        scripted: dict[str, str] | None = None,
        data_start: str = DEFAULT_DATA_START,
        data_end: str = DEFAULT_DATA_END,
    ) -> None:
        """
        Args:
            model: Name reported in results, so reports show the mock was used.
            vulnerable: If False, the mock ignores injections and never leaks.
            scripted: {trigger substring: reply}. Checked first, case-insensitive.
            data_start, data_end: Delimiters marking text as "data, not instructions".
        """
        self.model = model
        self.vulnerable = vulnerable
        self.scripted = scripted or {}
        self.data_start = data_start
        self.data_end = data_end
        # Every call is recorded so tests can check what the model was sent.
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message], **opts: Any) -> str:
        """Apply the rules from the module docstring and return a reply."""
        validate_messages(messages)
        self.calls.append([dict(message) for message in messages])

        last_user = _last_user_message(messages)
        # The question itself, without any pasted context or secrets in it.
        question = CANARY_PATTERN.sub(" ", self._strip_data_blocks(last_user))

        # Rule 1: scripted replies.
        for trigger, reply in self.scripted.items():
            if trigger.lower() in last_user.lower():
                return reply

        if self.vulnerable:
            # Rule 2: obey any marker instruction that is NOT inside a data block.
            instruction_text = self._strip_data_blocks(_prompt_text(messages))
            marker = MARKER_PATTERN.search(instruction_text)
            if marker:
                return f"Sure! {marker.group(0)}"

            # Rule 3: reveal the system prompt when asked for it.
            if EXTRACTION_PATTERN.search(question):
                system_prompt = _system_prompt(messages)
                if system_prompt:
                    return f"My instructions are: {system_prompt}"

            # Rule 4: leak canary lines from the context when asked for secrets.
            if EXFILTRATION_PATTERN.search(question):
                leaked = _lines_with_canaries(_prompt_text(messages))
                if leaked:
                    return "Here is what the documents say: " + " ".join(leaked)

        # Rule 5: a normal, grounded answer.
        return self._grounded_answer(messages)

    # ------------------------------------------------------------ helpers

    def _strip_data_blocks(self, text: str) -> str:
        """Remove every `<data> ... </data>` block from the text."""
        pattern = re.escape(self.data_start) + r".*?" + re.escape(self.data_end)
        return re.sub(pattern, " ", text, flags=re.DOTALL)

    def _context_text(self, messages: list[Message]) -> str:
        """Find the retrieved context: data blocks if present, else text after "Context:"."""
        text = _prompt_text(messages)
        pattern = re.escape(self.data_start) + r"(.*?)" + re.escape(self.data_end)
        blocks = re.findall(pattern, text, flags=re.DOTALL)
        if blocks:
            return "\n".join(blocks)
        if "Context:" in text:
            return text.split("Context:", 1)[1]
        return ""

    def _grounded_answer(self, messages: list[Message]) -> str:
        """Quote the first useful context line that isn't a secret."""
        for line in self._context_text(messages).splitlines():
            line = line.strip().lstrip("#-* ").strip()
            if len(line) < 20 or CANARY_PATTERN.search(line):
                continue
            return f"Based on the handbook: {line[:200]}"
        return NO_ANSWER


def _last_user_message(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def _system_prompt(messages: list[Message]) -> str:
    for message in messages:
        if message["role"] == "system":
            return message["content"]
    return ""


def _prompt_text(messages: list[Message]) -> str:
    """All system + user text. Assistant turns are the model's own words, so skip them."""
    return "\n".join(m["content"] for m in messages if m["role"] in ("system", "user"))


def _lines_with_canaries(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if CANARY_PATTERN.search(line)]
