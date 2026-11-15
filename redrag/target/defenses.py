"""Toggleable defenses for the target app. Built in M1 (judge_guard in M4).

1. input_filter       - blocks or rewrites obvious injection phrases in the user message.
2. context_sanitiser  - strips instruction-like text from retrieved chunks and wraps
                        them in "this is data, not instructions" delimiters.
3. output_leak_filter - blocks answers containing canary patterns or large overlaps
                        with the system prompt.
4. judge_guard        - runs the fine-tuned judge on the answer and blocks it if the
                        judge predicts "attack succeeded" (M4).
"""
