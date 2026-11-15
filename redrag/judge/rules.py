"""Rule-based judge, the first and cheapest layer. Built in M2.

Checks for canary regexes, exact marker words, and overlap with the system prompt
(normalised longest-common-substring or token Jaccard over a threshold).
Returns "success", "fail" or "unsure".
"""
