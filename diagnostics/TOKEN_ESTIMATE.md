# ESTIMATE: extra LLM input tokens per question of the v2.1 evidence (header 'deepest', expansion cap 300) — development data, not evidence of generalisation

These are estimates, not measurements: no prompt was sent; characters are converted with a fitted rate and the nodes each saved v2 row ran are inferred from its call count.

### Calibration: LLM input tokens = a + b x evidence-block characters (saved v2 rows, 4 calls) (development data, not evidence of generalisation)
| rows | a (tokens) | b (tokens per character) | R² |
|---|---|---|---|
| 30 | 3325.4 | 0.2211 | 0.471 |

### Graph nodes that receive the evidence (agent/agent_graph.py) (development data, not evidence of generalisation)
| Node | Evidence shown | Extra characters per call: mean | p90 |
|---|---|---|---|
| grade | every article, 500 characters | 595.3 | 1183 |
| reformulate | every article, 500 characters (retries) | 595.3 | 1183 |
| answer | every article, 3000 characters | 987.8 | 1700 |
| verify | cited articles, 3000 characters | 113.4 | 214 |
| repair | every article, 3000 characters (after a failed verify) | 987.8 | 1700 |
| condense, analyze | none | 0 | 0 |

### ESTIMATED extra input tokens per question added to v2 (development data, not evidence of generalisation)
| Group | items | answered (saved v2) | LLM calls mean (saved v2) | retries first: mean | p90 | repair first: mean | p90 | saved v2 input tokens mean | extra / saved (retries first) |
|---|---|---|---|---|---|---|---|---|---|
| Stage 1 | 30 | 30 | 5.37 | 554.6 | 1152 | 572.7 | 1035 | 6225 | 8.9 % |
| Stage 2 | 40 | 39 | 5.95 | 635.0 | 1175 | 634.9 | 1111 | 7094 | 9.0 % |
| Noisy set | 40 | 39 | 5.45 | 610.0 | 1257 | 615.9 | 1144 | 6538 | 9.3 % |
| language ar | 24 | 24 | 5.79 | 546.0 | 912 | 579.3 | 1002 | 5895 | 9.3 % |
| language en | 23 | 21 | 5.57 | 699.9 | 1343 | 765.5 | 1246 | 6016 | 11.6 % |
| language fr | 63 | 63 | 5.56 | 591.1 | 1201 | 566.7 | 1105 | 7177 | 8.2 % |
| all | 110 | 108 | 5.61 | 604.0 | 1226 | 611.0 | 1111 | 6655 | 9.1 % |

