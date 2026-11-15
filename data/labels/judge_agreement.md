# Judge agreement with hand labels

72 hand-labelled attacks (18 successful), evaluated 2026-11-15T21:33:16+00:00. Coverage = share of items the layer decided (the rest it called *unsure*); the other scores use only decided items. Kappa corrects for chance: 0 = luck, 1 = perfect.

**How the labels were made:** 4 labelled blind by a person; 68 by a person reviewing a suggested label from claude-opus-5-5 (suggestion overridden 2 times). Reviewed labels may lean toward the suggestion, which can flatter judges that think like the suggesting model (e.g. Gemini).

| Judge | Coverage | Accuracy | Precision | Recall | F1 | Cohen's kappa |
|---|---|---|---|---|---|---|
| Rule judge alone (regex, canaries, overlap) | 40% | 100% | 100% | 100% | 1.00 | 1.00 |
| Fine-tuned DeBERTa alone (P >= 0.5 = success) | 100% | 90% | 74% | 94% | 0.83 | 0.76 |
| Full chain: rules -> classifier -> Gemini | 100% | 90% | 74% | 94% | 0.83 | 0.76 |

Not evaluated (not available): gemini.
