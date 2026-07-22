# Deployment analysis — per-day rollup

- source: `src/feeding_deployment/preference_learning/methods/reports/2026_7_17_opus4.8_medium_single/logs/manual_1/prediction_model_llm_calls`
- model: `claude-opus-4-8 (speed=standard, effort=medium)`
- days: **22**
- mean cold-start accuracy Acc(m=0), full bundle: **0.826**
- mean end-of-meal accuracy, full bundle: **1.000**
- total corrections: **79**
- correlated ledger (whole deployment): +8 / -23 / =3 (pos/neg/lateral)
- self-inflicted corrections: 15 · non-bearing drifts: 0 · re-corrections: 0

> **Acc(m=0)/Final** below are the full-bundle accuracies from `day_metrics` (same definition as the report's Plot A). The `pinned m=0` column is the per-day analyzer's reconstruction, whose denominator is only the dims corrected/confirmed that meal — biased low by construction, kept for the correction-dynamics view.

See `deployment_curve.png` and per-day folders (`day_XX/`).

| Day | Meal | Setting | Time | Steps | Acc(m=0) | Final | pinned m=0 | Corr | +/−/= | self-infl | non-bear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | chicken nuggets | Personal | evening | 10 | 0.682 | 1.000 | 3/7 | 9 | +2/-5/=1 | 3 | 0 |
| 2 | general tso's chicken and broccoli | Watching TV with TV in Front | afternoon | 8 | 0.636 | 1.000 | 1/6 | 7 | +0/-1/=0 | 1 | 0 |
| 3 | chicken breast strips and hash brown | Watching TV with TV in Front | afternoon | 3 | 0.955 | 1.000 | 0/1 | 2 | +0/-0/=0 | 0 | 0 |
| 4 | cantaloupes, bananas, watermelon | Personal | morning | 5 | 0.636 | 1.000 | 0/3 | 4 | +0/-0/=0 | 0 | 0 |
| 5 | chicken nuggets, broccoli, and ketchup | Social with person on Left | morning | 9 | 0.545 | 1.000 | 0/5 | 8 | +1/-1/=0 | 0 | 0 |
| 6 | chicken nuggets, broccoli, and ketchup | Watching TV with TV in Front | night | 4 | 0.773 | 1.000 | 0/0 | 3 | +0/-0/=0 | 0 | 0 |
| 7 | bananas, brownies, and chocolate sauce | Watching TV with TV on Left | evening | 4 | 0.727 | 1.000 | 0/2 | 3 | +0/-0/=0 | 0 | 0 |
| 8 | strawberries with whipped cream | Watching TV with TV in Front | night | 4 | 0.909 | 1.000 | 0/0 | 3 | +0/-0/=0 | 0 | 0 |
| 9 | general tso's chicken and broccoli | Social with person on Left | afternoon | 5 | 0.909 | 1.000 | 2/3 | 4 | +0/-2/=0 | 2 | 0 |
| 10 | buffalo chicken bites, potato wedges, and ranch dressing | Personal | night | 8 | 0.773 | 1.000 | 3/5 | 7 | +2/-5/=0 | 3 | 0 |
| 11 | chicken breast strips and hash brown | Watching TV with TV on Right | evening | 1 | 1.000 | 1.000 | 0/0 | 0 | +0/-0/=0 | 0 | 0 |
| 12 | general tso's chicken and broccoli | Watching TV with TV on Right | morning | 8 | 0.727 | 1.000 | 2/6 | 7 | +1/-3/=0 | 2 | 0 |
| 13 | bite-sized sandwiches | Personal | night | 3 | 0.909 | 1.000 | 1/2 | 2 | +0/-1/=0 | 1 | 0 |
| 14 | bite-sized sandwiches | Social with person in Front | evening | 6 | 0.818 | 1.000 | 1/4 | 5 | +1/-2/=2 | 1 | 0 |
| 15 | chicken nuggets, broccoli, and ketchup | Watching TV with TV in Front | morning | 4 | 0.909 | 1.000 | 1/2 | 3 | +0/-1/=0 | 1 | 0 |
| 16 | bite-sized pizza and broccoli | Social with person on Left | evening | 3 | 0.864 | 1.000 | 1/2 | 2 | +0/-1/=0 | 1 | 0 |
| 17 | chicken nuggets, broccoli, and ketchup | Watching TV with TV on Right | morning | 2 | 0.818 | 1.000 | 0/1 | 1 | +0/-0/=0 | 0 | 0 |
| 18 | general tso's chicken and broccoli | Social with person on Right | evening | 3 | 0.909 | 1.000 | 0/1 | 2 | +0/-0/=0 | 0 | 0 |
| 19 | chicken breast strips and hash brown | Social with person in Front | afternoon | 3 | 0.909 | 1.000 | 0/1 | 2 | +0/-0/=0 | 0 | 0 |
| 20 | strawberries with whipped cream | Social with person in Front | afternoon | 5 | 0.818 | 1.000 | 0/2 | 4 | +1/-1/=0 | 0 | 0 |
| 21 | bite-sized pizza and broccoli | Social with person in Front | evening | 1 | 1.000 | 1.000 | 0/0 | 0 | +0/-0/=0 | 0 | 0 |
| 22 | cantaloupes, bananas, watermelon | Watching TV with TV in Front | morning | 2 | 0.955 | 1.000 | 0/1 | 1 | +0/-0/=0 | 0 | 0 |
