# Evidence budget: context header and cross-reference cap — development data, not evidence of generalisation

### 'Contexte :' header tokens per question (embedding tokens) (development data, not evidence of generalisation)
| Header | Group | n | top 6: mean | median | p90 | max | top 6 + added: mean | p90 |
|---|---|---|---|---|---|---|---|---|
| full | Stage 1 | 30 | 317.6 | 325.5 | 409 | 593 | 385.5 | 554 |
| full | Stage 2 | 40 | 326.8 | 314.0 | 452 | 557 | 377.4 | 543 |
| full | Noisy set | 40 | 321.6 | 326.5 | 462 | 557 | 375.9 | 492 |
| full | language ar | 24 | 338.2 | 362.0 | 476 | 557 | 379.2 | 531 |
| full | language en | 23 | 320.9 | 347 | 420 | 593 | 409.9 | 563 |
| full | language fr | 63 | 316.9 | 313 | 423 | 554 | 367.7 | 497 |
| full | all | 110 | 322.4 | 323.5 | 452 | 593 | 379.0 | 543 |
| deepest (default) | Stage 1 | 30 | 105.2 | 101.0 | 133 | 167 | 130.3 | 167 |
| deepest (default) | Stage 2 | 40 | 105.9 | 105.0 | 134 | 152 | 122.7 | 166 |
| deepest (default) | Noisy set | 40 | 105.1 | 99.0 | 142 | 161 | 122.6 | 157 |
| deepest (default) | language ar | 24 | 110.3 | 110.5 | 134 | 161 | 127.6 | 152 |
| deepest (default) | language en | 23 | 107.5 | 102 | 133 | 167 | 135.3 | 163 |
| deepest (default) | language fr | 63 | 102.8 | 98 | 134 | 152 | 119.8 | 158 |
| deepest (default) | all | 110 | 105.4 | 101.0 | 138 | 167 | 124.7 | 161 |
| none | Stage 1 | 30 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0 |
| none | Stage 2 | 40 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0 |
| none | Noisy set | 40 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0 |
| none | language ar | 24 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0 |
| none | language en | 23 | 0.0 | 0 | 0 | 0 | 0.0 | 0 |
| none | language fr | 63 | 0.0 | 0 | 0 | 0 | 0.0 | 0 |
| none | all | 110 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0 |

### Cross-reference expansion with a token cap per question, header 'deepest' (development data, not evidence of generalisation)
| Cap (tokens) | Group | n | added tokens mean | median | p90 | max | total | added articles | articles cut | gold in evidence with cross-refs | top-6 articles with a referenced article missing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | Stage 1 | 30 | 152.5 | 150.0 | 300 | 300 | 4574 | 24 | 11 | 22 (73.3 %) | 10 (5.6 %) |
| 300 | Stage 2 | 40 | 138.2 | 69.5 | 300 | 300 | 5528 | 32 | 13 | 32 (80.0 %) | 7 (2.9 %) |
| 300 | Noisy set | 40 | 147.0 | 133.0 | 300 | 300 | 5879 | 34 | 14 | 24 (60.0 %) | 9 (3.8 %) |
| 300 | language ar | 24 | 117.3 | 0.0 | 300 | 300 | 2816 | 16 | 6 | 19 (79.2 %) | 3 (2.1 %) |
| 300 | language en | 23 | 193.6 | 300 | 300 | 300 | 4452 | 21 | 13 | 16 (69.6 %) | 12 (8.7 %) |
| 300 | language fr | 63 | 138.3 | 124 | 300 | 300 | 8713 | 53 | 19 | 43 (68.3 %) | 11 (2.9 %) |
| 300 | all | 110 | 145.3 | 139.0 | 300 | 300 | 15981 | 90 | 38 | 78 (70.9 %) | 26 (3.9 %) |
| 600 | Stage 1 | 30 | 235.6 | 150.0 | 600 | 600 | 7068 | 30 | 6 | 22 (73.3 %) | 5 (2.8 %) |
| 600 | Stage 2 | 40 | 195.8 | 69.5 | 600 | 600 | 7832 | 37 | 5 | 32 (80.0 %) | 4 (1.7 %) |
| 600 | Noisy set | 40 | 207.1 | 133.0 | 580 | 600 | 8285 | 39 | 4 | 24 (60.0 %) | 6 (2.5 %) |
| 600 | language ar | 24 | 158.9 | 0.0 | 398 | 600 | 3813 | 18 | 2 | 19 (79.2 %) | 1 (0.7 %) |
| 600 | language en | 23 | 313.4 | 356 | 600 | 600 | 7208 | 29 | 7 | 16 (69.6 %) | 7 (5.1 %) |
| 600 | language fr | 63 | 193.1 | 124 | 556 | 600 | 12164 | 59 | 6 | 43 (68.3 %) | 7 (1.9 %) |
| 600 | all | 110 | 210.8 | 139.0 | 600 | 600 | 23185 | 106 | 15 | 78 (70.9 %) | 15 (2.3 %) |
| unlimited | Stage 1 | 30 | 310.3 | 150.0 | 844 | 1387 | 9308 | 36 | 0 | 22 (73.3 %) | 1 (0.6 %) |
| unlimited | Stage 2 | 40 | 240.8 | 69.5 | 688 | 1445 | 9630 | 39 | 0 | 32 (80.0 %) | 2 (0.8 %) |
| unlimited | Noisy set | 40 | 244.6 | 133.0 | 641 | 1385 | 9784 | 43 | 0 | 24 (60.0 %) | 4 (1.7 %) |
| unlimited | language ar | 24 | 171.6 | 0.0 | 398 | 844 | 4118 | 19 | 0 | 19 (79.2 %) | 0 (0.0 %) |
| unlimited | language en | 23 | 450.9 | 356 | 996 | 1445 | 10370 | 36 | 0 | 16 (69.6 %) | 2 (1.4 %) |
| unlimited | language fr | 63 | 225.9 | 124 | 556 | 1385 | 14234 | 63 | 0 | 43 (68.3 %) | 5 (1.3 %) |
| unlimited | all | 110 | 261.1 | 139.0 | 731 | 1445 | 28722 | 118 | 0 | 78 (70.9 %) | 7 (1.1 %) |

### Decision (development data, not evidence of generalisation)
| Rule (fixed before the run) | Items lost vs unlimited | Chosen cap |
|---|---|---|
| smallest cap losing at most 1 item versus unlimited | 300: 0, 600: 0, unlimited: 0 | 300 |

