"""Combines the three judge layers: rules -> classifier -> Gemini. Built in M2/M4.

Stores both the final verdict and which layer decided it, so the report can
show how often each layer was needed and how well it agrees with hand labels.
"""
