# Offline comparison: db_v1_1 (switches off) vs db_v1_2 (v2.1 wave 1) — development data, not evidence of generalisation

### Configurations (development data, not evidence of generalisation)
| Configuration | Index | Switches |
|---|---|---|
| baseline | db_v1_1 (byte copy) | all v2.1 switches off (= v2 retrieval core) |
| v1.2 | db_v1_2 | headings forward, context in indexed text, sub-chunks, strict direct lookup + union, cross-reference expansion, context header; BM25 tokenizer whitespace |

### Index build facts (development data, not evidence of generalisation)
| Fact | db_v1_1 | db_v1_2 |
|---|---|---|
| articles (chunk ids) | 6031 | 6031 |
| article ids identical to db_v1_1 | - | True |
| index entries (embedded texts) | 6031 | 6112 |
| articles split into sub-chunks | 0 | 71 |
| sub-chunk entries | 0 | 152 |
| embedded texts over 512 tokens (Solon max_seq_length) | 63 | 0 |
| articles containing a section/chapter/title line | 736 | 65 |
| headings moved to the following article | 0 | 822 |
| articles whose text lost a trailing heading | 0 | 685 |
| articles without a heading hierarchy | - | 282 |
| index size on disk (MB) | 76.9 | 84.4 |
| embedding time (s) | not recorded | 4860.4 |
| split + token count time (s) | - | 76.0 |
| build total time (s) | - | 4937.8 |
| device | - | cpu (Intel64 Family 6 Model 165 Stepping 2, GenuineIntel, Windows 10) |

### Baseline reproduces the diagnostic of RETRIEVAL_DIAGNOSTIC.md (development data, not evidence of generalisation)
| Items | same funnel class | different |
|---|---|---|
| 110 | 110 | 0 |

### Funnel classes and Hit@k, baseline (db_v1_1) (development data, not evidence of generalisation)
| Group | n | found in top 6 (reranked) | in pool, reranked 7-10 | in pool, reranked 11-20 | fused rank > 20 (cut before the reranker) | absent from the pool (not cut: never a candidate) | Hit@6 | Hit@8 | Hit@10 | gold in evidence with cross-refs |
|---|---|---|---|---|---|---|---|---|---|---|
| Stage 1 | 30 | 22 (73.3 %) | 2 (6.7 %) | 3 (10.0 %) | 1 (3.3 %) | 2 (6.7 %) | 22 (73.3 %) | 23 (76.7 %) | 24 (80.0 %) | 22 (73.3 %) |
| Stage 2 | 40 | 29 (72.5 %) | 2 (5.0 %) | 2 (5.0 %) | 3 (7.5 %) | 4 (10.0 %) | 29 (72.5 %) | 29 (72.5 %) | 31 (77.5 %) | 29 (72.5 %) |
| Noisy set | 40 | 19 (47.5 %) | 4 (10.0 %) | 0 (0.0 %) | 6 (15.0 %) | 11 (27.5 %) | 19 (47.5 %) | 22 (55.0 %) | 23 (57.5 %) | 19 (47.5 %) |
| language ar | 24 | 20 (83.3 %) | 1 (4.2 %) | 0 (0.0 %) | 1 (4.2 %) | 2 (8.3 %) | 20 (83.3 %) | 20 (83.3 %) | 21 (87.5 %) | 20 (83.3 %) |
| language en | 23 | 15 (65.2 %) | 1 (4.3 %) | 2 (8.7 %) | 4 (17.4 %) | 1 (4.3 %) | 15 (65.2 %) | 16 (69.6 %) | 16 (69.6 %) | 15 (65.2 %) |
| language fr | 63 | 35 (55.6 %) | 6 (9.5 %) | 3 (4.8 %) | 5 (7.9 %) | 14 (22.2 %) | 35 (55.6 %) | 38 (60.3 %) | 41 (65.1 %) | 35 (55.6 %) |
| noise describe | 10 | 2 (20.0 %) | 1 (10.0 %) | 0 (0.0 %) | 2 (20.0 %) | 5 (50.0 %) | 2 (20.0 %) | 2 (20.0 %) | 3 (30.0 %) | 2 (20.0 %) |
| noise informal | 10 | 6 (60.0 %) | 1 (10.0 %) | 0 (0.0 %) | 1 (10.0 %) | 2 (20.0 %) | 6 (60.0 %) | 7 (70.0 %) | 7 (70.0 %) | 6 (60.0 %) |
| noise typos | 10 | 7 (70.0 %) | 0 (0.0 %) | 0 (0.0 %) | 2 (20.0 %) | 1 (10.0 %) | 7 (70.0 %) | 7 (70.0 %) | 7 (70.0 %) | 7 (70.0 %) |
| noise wrong_term | 10 | 4 (40.0 %) | 2 (20.0 %) | 0 (0.0 %) | 1 (10.0 %) | 3 (30.0 %) | 4 (40.0 %) | 6 (60.0 %) | 6 (60.0 %) | 4 (40.0 %) |
| all | 110 | 70 (63.6 %) | 8 (7.3 %) | 5 (4.5 %) | 10 (9.1 %) | 17 (15.5 %) | 70 (63.6 %) | 74 (67.3 %) | 78 (70.9 %) | 70 (63.6 %) |

### Funnel classes and Hit@k, v1.2 (db_v1_2) (development data, not evidence of generalisation)
| Group | n | found in top 6 (reranked) | in pool, reranked 7-10 | in pool, reranked 11-20 | fused rank > 20 (cut before the reranker) | absent from the pool (not cut: never a candidate) | Hit@6 | Hit@8 | Hit@10 | gold in evidence with cross-refs |
|---|---|---|---|---|---|---|---|---|---|---|
| Stage 1 | 30 | 21 (70.0 %) | 3 (10.0 %) | 3 (10.0 %) | 0 (0.0 %) | 3 (10.0 %) | 21 (70.0 %) | 23 (76.7 %) | 24 (80.0 %) | 22 (73.3 %) |
| Stage 2 | 40 | 31 (77.5 %) | 1 (2.5 %) | 1 (2.5 %) | 2 (5.0 %) | 5 (12.5 %) | 31 (77.5 %) | 31 (77.5 %) | 32 (80.0 %) | 32 (80.0 %) |
| Noisy set | 40 | 23 (57.5 %) | 3 (7.5 %) | 2 (5.0 %) | 2 (5.0 %) | 10 (25.0 %) | 23 (57.5 %) | 26 (65.0 %) | 26 (65.0 %) | 24 (60.0 %) |
| language ar | 24 | 19 (79.2 %) | 1 (4.2 %) | 1 (4.2 %) | 0 (0.0 %) | 3 (12.5 %) | 19 (79.2 %) | 20 (83.3 %) | 20 (83.3 %) | 19 (79.2 %) |
| language en | 23 | 16 (69.6 %) | 0 (0.0 %) | 1 (4.3 %) | 0 (0.0 %) | 6 (26.1 %) | 16 (69.6 %) | 16 (69.6 %) | 16 (69.6 %) | 16 (69.6 %) |
| language fr | 63 | 40 (63.5 %) | 6 (9.5 %) | 4 (6.3 %) | 4 (6.3 %) | 9 (14.3 %) | 40 (63.5 %) | 44 (69.8 %) | 46 (73.0 %) | 43 (68.3 %) |
| noise describe | 10 | 4 (40.0 %) | 0 (0.0 %) | 1 (10.0 %) | 0 (0.0 %) | 5 (50.0 %) | 4 (40.0 %) | 4 (40.0 %) | 4 (40.0 %) | 4 (40.0 %) |
| noise informal | 10 | 6 (60.0 %) | 0 (0.0 %) | 0 (0.0 %) | 1 (10.0 %) | 3 (30.0 %) | 6 (60.0 %) | 6 (60.0 %) | 6 (60.0 %) | 6 (60.0 %) |
| noise typos | 10 | 7 (70.0 %) | 1 (10.0 %) | 1 (10.0 %) | 0 (0.0 %) | 1 (10.0 %) | 7 (70.0 %) | 8 (80.0 %) | 8 (80.0 %) | 7 (70.0 %) |
| noise wrong_term | 10 | 6 (60.0 %) | 2 (20.0 %) | 0 (0.0 %) | 1 (10.0 %) | 1 (10.0 %) | 6 (60.0 %) | 8 (80.0 %) | 8 (80.0 %) | 7 (70.0 %) |
| all | 110 | 75 (68.2 %) | 7 (6.4 %) | 6 (5.5 %) | 4 (3.6 %) | 18 (16.4 %) | 75 (68.2 %) | 80 (72.7 %) | 82 (74.5 %) | 78 (70.9 %) |

### Hit@k, baseline -> v1.2 (items) (development data, not evidence of generalisation)
| Group | n | Hit@6 | Hit@8 | Hit@10 | gold in evidence with cross-refs |
|---|---|---|---|---|---|
| Stage 1 | 30 | 22 -> 21 | 23 -> 23 | 24 -> 24 | 22 -> 22 |
| Stage 2 | 40 | 29 -> 31 | 29 -> 31 | 31 -> 32 | 29 -> 32 |
| Noisy set | 40 | 19 -> 23 | 22 -> 26 | 23 -> 26 | 19 -> 24 |
| language ar | 24 | 20 -> 19 | 20 -> 20 | 21 -> 20 | 20 -> 19 |
| language en | 23 | 15 -> 16 | 16 -> 16 | 16 -> 16 | 15 -> 16 |
| language fr | 63 | 35 -> 40 | 38 -> 44 | 41 -> 46 | 35 -> 43 |
| noise describe | 10 | 2 -> 4 | 2 -> 4 | 3 -> 4 | 2 -> 4 |
| noise informal | 10 | 6 -> 6 | 7 -> 6 | 7 -> 6 | 6 -> 6 |
| noise typos | 10 | 7 -> 7 | 7 -> 8 | 7 -> 8 | 7 -> 7 |
| noise wrong_term | 10 | 4 -> 6 | 6 -> 8 | 6 -> 8 | 4 -> 7 |
| all | 110 | 70 -> 75 | 74 -> 80 | 78 -> 82 | 70 -> 78 |

### Hit@6 per item, baseline -> v1.2 (development data, not evidence of generalisation)
| Change | Items |
|---|---|
| gained | 12 |
| lost | 7 |
| same hit | 63 |
| same miss | 28 |

### Items that lost Hit@6 (development data, not evidence of generalisation)
| Set | Id | Language | baseline class | v1.2 class |
|---|---|---|---|---|
| Stage 1 | ac41 | fr | found in top 6 (reranked) | in pool, reranked 7-10 |
| Stage 2 | ac07 | ar | found in top 6 (reranked) | absent from the pool (not cut: never a candidate) |
| Stage 2 | ac08 | fr | found in top 6 (reranked) | in pool, reranked 7-10 |
| Stage 2 | ac32 | fr | found in top 6 (reranked) | absent from the pool (not cut: never a candidate) |
| Stage 2 | ac48 | fr | found in top 6 (reranked) | fused rank > 20 (cut before the reranker) |
| Noisy set | n_ac16 | ar | found in top 6 (reranked) | in pool, reranked 11-20 |
| Noisy set | n_ac18 | ar | found in top 6 (reranked) | in pool, reranked 7-10 |

### Retrieval path (development data, not evidence of generalisation)
| Configuration | Set | n | hybrid | direct (replaces the pool) | direct + hybrid (union) | direct paths with Hit@6 |
|---|---|---|---|---|---|---|
| baseline | Stage 1 | 30 | 30 | 0 | 0 | 0 |
| baseline | Stage 2 | 40 | 38 | 2 | 0 | 0 |
| baseline | Noisy set | 40 | 38 | 2 | 0 | 1 |
| v1.2 | Stage 1 | 30 | 30 | 0 | 0 | 0 |
| v1.2 | Stage 2 | 40 | 40 | 0 | 0 | 0 |
| v1.2 | Noisy set | 40 | 40 | 0 | 0 | 0 |

### Cited-style articles whose referenced article is missing from the evidence (development data, not evidence of generalisation)
| Configuration | Articles | Group | articles | with a resolvable reference | with a referenced article not in the evidence |
|---|---|---|---|---|---|
| baseline | gold articles in the top 6 | Stage 1 | 23 | 3 (13.0 %) | 3 (13.0 %) |
| baseline | gold articles in the top 6 | Stage 2 | 32 | 11 (34.4 %) | 10 (31.2 %) |
| baseline | gold articles in the top 6 | Noisy set | 22 | 5 (22.7 %) | 5 (22.7 %) |
| baseline | gold articles in the top 6 | all | 77 | 19 (24.7 %) | 18 (23.4 %) |
| baseline | all top-6 articles | Stage 1 | 180 | 30 (16.7 %) | 30 (16.7 %) |
| baseline | all top-6 articles | Stage 2 | 240 | 42 (17.5 %) | 35 (14.6 %) |
| baseline | all top-6 articles | Noisy set | 240 | 38 (15.8 %) | 30 (12.5 %) |
| baseline | all top-6 articles | all | 660 | 110 (16.7 %) | 95 (14.4 %) |
| v1.2 | gold articles in the top 6 | Stage 1 | 22 | 3 (13.6 %) | 0 (0.0 %) |
| v1.2 | gold articles in the top 6 | Stage 2 | 35 | 11 (31.4 %) | 0 (0.0 %) |
| v1.2 | gold articles in the top 6 | Noisy set | 25 | 8 (32.0 %) | 0 (0.0 %) |
| v1.2 | gold articles in the top 6 | all | 82 | 22 (26.8 %) | 0 (0.0 %) |
| v1.2 | all top-6 articles | Stage 1 | 180 | 32 (17.8 %) | 1 (0.6 %) |
| v1.2 | all top-6 articles | Stage 2 | 240 | 39 (16.2 %) | 2 (0.8 %) |
| v1.2 | all top-6 articles | Noisy set | 240 | 45 (18.8 %) | 4 (1.7 %) |
| v1.2 | all top-6 articles | all | 660 | 116 (17.6 %) | 7 (1.1 %) |

### Cross-reference expansion cost on the dev items (v1.2; tokens = embedding-model tokenizer, not the LLM's) (development data, not evidence of generalisation)
| Group | items | items with added articles | added articles mean | median | added tokens mean | median | p90 | max | total | gold only via an added article |
|---|---|---|---|---|---|---|---|---|---|---|
| Stage 1 | 30 | 18 | 1.2 | 1.0 | 353.1 | 169.0 | 886 | 1663 | 10593 | 1 |
| Stage 2 | 40 | 20 | 1.0 | 0.5 | 274.5 | 90.5 | 784 | 1553 | 10979 | 1 |
| Noisy set | 40 | 23 | 1.1 | 1.0 | 281.3 | 180.0 | 751 | 1435 | 11252 | 1 |
| language ar | 24 | 11 | 0.8 | 0.0 | 195.2 | 0.0 | 414 | 886 | 4686 | 0 |
| language en | 23 | 15 | 1.6 | 2 | 512.0 | 443 | 1096 | 1663 | 11777 | 0 |
| language fr | 63 | 35 | 1.0 | 1 | 259.7 | 145 | 642 | 1435 | 16361 | 3 |
| noise describe | 10 | 3 | 0.7 | 0.0 | 148.1 | 0.0 | 404 | 892 | 1481 | 0 |
| noise informal | 10 | 8 | 1.3 | 1.0 | 375.0 | 293.0 | 763 | 1096 | 3750 | 0 |
| noise typos | 10 | 6 | 1.1 | 1.0 | 325.3 | 256.0 | 448 | 1435 | 3253 | 0 |
| noise wrong_term | 10 | 6 | 1.2 | 1.5 | 276.8 | 280.5 | 692 | 751 | 2768 | 1 |
| all | 110 | 61 | 1.1 | 1.0 | 298.4 | 177.5 | 819 | 1663 | 32824 | 3 |

### Evidence size per question, v1.2 top 6 (embedding-model tokens) (development data, not evidence of generalisation)
| Group | items | top-6 evidence tokens mean | median | p90 | max | context-header tokens mean | median |
|---|---|---|---|---|---|---|---|
| Stage 1 | 30 | 1268.9 | 1175.0 | 1589 | 1899 | 317.6 | 325.5 |
| Stage 2 | 40 | 1214.8 | 1181.5 | 1597 | 1843 | 326.8 | 314.0 |
| Noisy set | 40 | 1157.4 | 1139.0 | 1487 | 1871 | 321.6 | 326.5 |
| all | 110 | 1208.7 | 1174.0 | 1586 | 1899 | 322.4 | 323.5 |

### Evidence window: top 6 vs top 8 vs top 10 (default unchanged: 6) (development data, not evidence of generalisation)
| Configuration | Group | n | Hit@6 | Hit@8 | Hit@10 | added tokens for top 8 (mean) | added tokens for top 10 (mean) |
|---|---|---|---|---|---|---|---|
| baseline | Stage 1 | 30 | 22 (73.3 %) | 23 (76.7 %) | 24 (80.0 %) | 295.8 | 557.9 |
| baseline | Stage 2 | 40 | 29 (72.5 %) | 29 (72.5 %) | 31 (77.5 %) | 300.1 | 566.5 |
| baseline | Noisy set | 40 | 19 (47.5 %) | 22 (55.0 %) | 23 (57.5 %) | 272.1 | 528.6 |
| baseline | all | 110 | 70 (63.6 %) | 74 (67.3 %) | 78 (70.9 %) | 288.7 | 550.4 |
| v1.2 | Stage 1 | 30 | 21 (70.0 %) | 23 (76.7 %) | 24 (80.0 %) | 451.6 | 892.2 |
| v1.2 | Stage 2 | 40 | 31 (77.5 %) | 31 (77.5 %) | 32 (80.0 %) | 445.1 | 868.7 |
| v1.2 | Noisy set | 40 | 23 (57.5 %) | 26 (65.0 %) | 26 (65.0 %) | 450.2 | 900.5 |
| v1.2 | all | 110 | 75 (68.2 %) | 80 (72.7 %) | 82 (74.5 %) | 448.7 | 886.7 |

### BM25 contribution: gold in the dense (k=20) and BM25 (k=15) lists (development data, not evidence of generalisation)
| Configuration | Group | n | in both | BM25 only | dense only | neither | BM25 only and Hit@6 |
|---|---|---|---|---|---|---|---|
| baseline | Stage 1 | 30 | 14 | 0 | 13 | 3 | 0 |
| baseline | Stage 2 | 40 | 21 | 2 | 15 | 2 | 1 |
| baseline | Noisy set | 40 | 10 | 1 | 19 | 10 | 0 |
| baseline | all | 110 | 45 | 3 | 47 | 15 | 1 |
| v1.2 | Stage 1 | 30 | 13 | 0 | 14 | 3 | 0 |
| v1.2 | Stage 2 | 40 | 21 | 1 | 12 | 6 | 1 |
| v1.2 | Noisy set | 40 | 11 | 1 | 18 | 10 | 1 |
| v1.2 | all | 110 | 45 | 2 | 44 | 19 | 2 |

### BM25 tokenizer variants on db_v1_2 (other v2.1 switches on) (development data, not evidence of generalisation)
| Tokenizer | Group | n | gold in the BM25 list (k=15) | median BM25 rank of the gold | Hit@6 | Hit@10 |
|---|---|---|---|---|---|---|
| whitespace | Stage 1 | 30 | 13 | 4 | 21 (70.0 %) | 24 (80.0 %) |
| whitespace | Stage 2 | 40 | 22 | 2.5 | 31 (77.5 %) | 32 (80.0 %) |
| whitespace | Noisy set | 40 | 12 | 2.0 | 23 (57.5 %) | 26 (65.0 %) |
| whitespace | all | 110 | 47 | 2 | 75 (68.2 %) | 82 (74.5 %) |
| lower | Stage 1 | 30 | 16 | 1.5 | 22 (73.3 %) | 25 (83.3 %) |
| lower | Stage 2 | 40 | 25 | 2 | 31 (77.5 %) | 34 (85.0 %) |
| lower | Noisy set | 40 | 13 | 2 | 22 (55.0 %) | 27 (67.5 %) |
| lower | all | 110 | 54 | 2.0 | 75 (68.2 %) | 86 (78.2 %) |
| fold | Stage 1 | 30 | 16 | 1.5 | 21 (70.0 %) | 24 (80.0 %) |
| fold | Stage 2 | 40 | 25 | 2 | 31 (77.5 %) | 32 (80.0 %) |
| fold | Noisy set | 40 | 14 | 3.0 | 22 (55.0 %) | 25 (62.5 %) |
| fold | all | 110 | 55 | 2 | 74 (67.3 %) | 81 (73.6 %) |
| elision | Stage 1 | 30 | 16 | 2.0 | 22 (73.3 %) | 25 (83.3 %) |
| elision | Stage 2 | 40 | 24 | 2.0 | 32 (80.0 %) | 32 (80.0 %) |
| elision | Noisy set | 40 | 14 | 3.5 | 23 (57.5 %) | 27 (67.5 %) |
| elision | all | 110 | 54 | 2.0 | 77 (70.0 %) | 84 (76.4 %) |
| stem | Stage 1 | 30 | 21 | 3 | 23 (76.7 %) | 25 (83.3 %) |
| stem | Stage 2 | 40 | 28 | 1.0 | 30 (75.0 %) | 33 (82.5 %) |
| stem | Noisy set | 40 | 16 | 1.5 | 23 (57.5 %) | 27 (67.5 %) |
| stem | all | 110 | 65 | 1 | 76 (69.1 %) | 85 (77.3 %) |

### BM25 tokenizer adoption rule (development data, not evidence of generalisation)
| Tokenizer | BM25-list gold, all | no set/language with fewer BM25-list gold | no set/language with lower Hit@6 | adopted |
|---|---|---|---|---|
| whitespace | 47 | True | True | True |
| lower | 54 | False | False | False |
| fold | 55 | False | False | False |
| elision | 54 | False | False | False |
| stem | 65 | True | False | False |

### Run (development data, not evidence of generalisation)
| Item | Value |
|---|---|
| items | 110 answerable items (Stage 1, Stage 2, noisy set; all languages) |
| French queries of Arabic/English items | diagnostics/translations.json |
| LLM calls | 0 |
| seconds | 1697.9 |
| fingerprint of the configuration files at run time (db_v1_2) | d9ab1df83cb3c23dd0a53fa11241c5381b9dd0d703b3e1809dd4e783f54aa944 |
| token counts | embedding-model tokenizer (Solon / XLM-R), special tokens included; the LLM tokenizer is not available offline |

