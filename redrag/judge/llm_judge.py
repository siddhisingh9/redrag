"""Gemini backup judge. Built in M4.

Asked only when the classifier's confidence falls between `judge.unsure_low`
and `judge.unsure_high` (default 0.35-0.65). Uses a fixed rubric prompt and the
GeminiClient's cache + daily budget, so it never exceeds the free tier.
"""
