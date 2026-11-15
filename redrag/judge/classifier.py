"""Fine-tuned DeBERTa-v3-small judge. Built in M4.

Loads the model trained in `notebooks/finetune_judge.ipynb` and returns the
probability that an attack succeeded, given (attack goal, response). Runs on CPU.
Used when the rule judge says "unsure".
"""
