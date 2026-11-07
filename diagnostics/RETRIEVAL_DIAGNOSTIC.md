# Retrieval diagnostic (v1.1 path, db_v1_1, Stage 1, Stage 2, noisy set: answerable items, all languages)

### Definitions
| Term | Definition |
|---|---|
| found in top 6 (reranked) | gold rank 1-6 in the cross-encoder order of the pool used |
| in pool, reranked 7-10 | gold in the pool, cross-encoder rank 7-10 |
| in pool, reranked 11-20 | gold in the pool, cross-encoder rank 11-20 |
| fused rank > 20 (cut before the reranker) | hybrid path: gold in the dense or BM25 list, fused rank > RERANK_POOL (20) |
| absent from the pool (not cut: never a candidate) | gold in no candidate list (hybrid), or not in the direct-lookup pool (direct) |
| direct path | a number in the question (pipeline._ARTICLE_RE = \b\d+\b) and retriever._direct_lookup returned documents: they are the pool |
| hybrid path | dense (k=20) + BM25 (k=15) lists on the French query (+ dense k=10 on the original question for Arabic/English), reciprocal rank fusion, deduplication, first 20 |
| French query | French items: the question as written; Arabic/English items: translate.translate_to_french, run once and stored in diagnostics/translations.json |
| Hit@k | gold within the first k of the cross-encoder order of the pool used (the system keeps 6) |
| gold match | source stem + article label (run_answer_eval.key / gold_set) |

### Items analysed and skipped
| Set | Answerable items | Analysed | Skipped | Skip reasons |
|---|---|---|---|---|
| Stage 1 | 30 | 30 | 0 | - |
| Stage 2 | 40 | 40 | 0 | - |
| Noisy set | 40 | 40 | 0 | - |

### Skipped items by language
| Set | Language | Skipped |
|---|---|---|
| - | - | 0 |

### Funnel classes per set
| Group | n | found in top 6 (reranked) | in pool, reranked 7-10 | in pool, reranked 11-20 | fused rank > 20 (cut before the reranker) | absent from the pool (not cut: never a candidate) | Hit@6 | Hit@8 | Hit@10 |
|---|---|---|---|---|---|---|---|---|---|
| Stage 1 | 30 | 22 (73.3 %) | 2 (6.7 %) | 3 (10.0 %) | 1 (3.3 %) | 2 (6.7 %) | 22 (73.3 %) | 23 (76.7 %) | 24 (80.0 %) |
| Stage 2 | 40 | 29 (72.5 %) | 2 (5.0 %) | 2 (5.0 %) | 3 (7.5 %) | 4 (10.0 %) | 29 (72.5 %) | 29 (72.5 %) | 31 (77.5 %) |
| Noisy set | 40 | 19 (47.5 %) | 4 (10.0 %) | 0 (0.0 %) | 6 (15.0 %) | 11 (27.5 %) | 19 (47.5 %) | 22 (55.0 %) | 23 (57.5 %) |

### Funnel classes per set and language
| Group | n | found in top 6 (reranked) | in pool, reranked 7-10 | in pool, reranked 11-20 | fused rank > 20 (cut before the reranker) | absent from the pool (not cut: never a candidate) | Hit@6 | Hit@8 | Hit@10 |
|---|---|---|---|---|---|---|---|---|---|
| Noisy set, ar | 9 | 7 (77.8 %) | 0 (0.0 %) | 0 (0.0 %) | 1 (11.1 %) | 1 (11.1 %) | 7 (77.8 %) | 7 (77.8 %) | 7 (77.8 %) |
| Noisy set, en | 8 | 4 (50.0 %) | 1 (12.5 %) | 0 (0.0 %) | 2 (25.0 %) | 1 (12.5 %) | 4 (50.0 %) | 5 (62.5 %) | 5 (62.5 %) |
| Noisy set, fr | 23 | 8 (34.8 %) | 3 (13.0 %) | 0 (0.0 %) | 3 (13.0 %) | 9 (39.1 %) | 8 (34.8 %) | 10 (43.5 %) | 11 (47.8 %) |
| Stage 1, ar | 6 | 5 (83.3 %) | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) | 1 (16.7 %) | 5 (83.3 %) | 5 (83.3 %) | 5 (83.3 %) |
| Stage 1, en | 7 | 5 (71.4 %) | 0 (0.0 %) | 1 (14.3 %) | 1 (14.3 %) | 0 (0.0 %) | 5 (71.4 %) | 5 (71.4 %) | 5 (71.4 %) |
| Stage 1, fr | 17 | 12 (70.6 %) | 2 (11.8 %) | 2 (11.8 %) | 0 (0.0 %) | 1 (5.9 %) | 12 (70.6 %) | 13 (76.5 %) | 14 (82.4 %) |
| Stage 2, ar | 9 | 8 (88.9 %) | 1 (11.1 %) | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) | 8 (88.9 %) | 8 (88.9 %) | 9 (100.0 %) |
| Stage 2, en | 8 | 6 (75.0 %) | 0 (0.0 %) | 1 (12.5 %) | 1 (12.5 %) | 0 (0.0 %) | 6 (75.0 %) | 6 (75.0 %) | 6 (75.0 %) |
| Stage 2, fr | 23 | 15 (65.2 %) | 1 (4.3 %) | 1 (4.3 %) | 2 (8.7 %) | 4 (17.4 %) | 15 (65.2 %) | 15 (65.2 %) | 16 (69.6 %) |

### Funnel classes per language (all three sets)
| Group | n | found in top 6 (reranked) | in pool, reranked 7-10 | in pool, reranked 11-20 | fused rank > 20 (cut before the reranker) | absent from the pool (not cut: never a candidate) | Hit@6 | Hit@8 | Hit@10 |
|---|---|---|---|---|---|---|---|---|---|
| ar | 24 | 20 (83.3 %) | 1 (4.2 %) | 0 (0.0 %) | 1 (4.2 %) | 2 (8.3 %) | 20 (83.3 %) | 20 (83.3 %) | 21 (87.5 %) |
| en | 23 | 15 (65.2 %) | 1 (4.3 %) | 2 (8.7 %) | 4 (17.4 %) | 1 (4.3 %) | 15 (65.2 %) | 16 (69.6 %) | 16 (69.6 %) |
| fr | 63 | 35 (55.6 %) | 6 (9.5 %) | 3 (4.8 %) | 5 (7.9 %) | 14 (22.2 %) | 35 (55.6 %) | 38 (60.3 %) | 41 (65.1 %) |

### Original-language dense list (Arabic/English items, k=10)
| Set | items | gold in the original-language list | gold only there (not in dense or BM25 on the French query) |
|---|---|---|---|
| Stage 1 | 13 | 11 | 1 |
| Stage 2 | 17 | 15 | 0 |
| Noisy set | 17 | 11 | 0 |

### Funnel classes per noise type (noisy set)
| Group | n | found in top 6 (reranked) | in pool, reranked 7-10 | in pool, reranked 11-20 | fused rank > 20 (cut before the reranker) | absent from the pool (not cut: never a candidate) | Hit@6 | Hit@8 | Hit@10 |
|---|---|---|---|---|---|---|---|---|---|
| describe | 10 | 2 (20.0 %) | 1 (10.0 %) | 0 (0.0 %) | 2 (20.0 %) | 5 (50.0 %) | 2 (20.0 %) | 2 (20.0 %) | 3 (30.0 %) |
| informal | 10 | 6 (60.0 %) | 1 (10.0 %) | 0 (0.0 %) | 1 (10.0 %) | 2 (20.0 %) | 6 (60.0 %) | 7 (70.0 %) | 7 (70.0 %) |
| typos | 10 | 7 (70.0 %) | 0 (0.0 %) | 0 (0.0 %) | 2 (20.0 %) | 1 (10.0 %) | 7 (70.0 %) | 7 (70.0 %) | 7 (70.0 %) |
| wrong_term | 10 | 4 (40.0 %) | 2 (20.0 %) | 0 (0.0 %) | 1 (10.0 %) | 3 (30.0 %) | 4 (40.0 %) | 6 (60.0 %) | 6 (60.0 %) |

### Gold in the candidate lists (hybrid lists computed for every analysed item)
| Set | n | in dense and BM25 | absent from dense only | absent from BM25 only | absent from both |
|---|---|---|---|---|---|
| Stage 1 | 30 | 14 (46.7 %) | 0 (0.0 %) | 13 (43.3 %) | 3 (10.0 %) |
| Stage 2 | 40 | 21 (52.5 %) | 2 (5.0 %) | 15 (37.5 %) | 2 (5.0 %) |
| Noisy set | 40 | 10 (25.0 %) | 1 (2.5 %) | 19 (47.5 %) | 10 (25.0 %) |

### Retrieval path per set
| Set | n | hybrid | direct (number in the question) | direct: gold in the pool | direct: gold would be in the hybrid pool |
|---|---|---|---|---|---|
| Stage 1 | 30 | 30 (100.0 %) | 0 (0.0 %) | 0 | 0 |
| Stage 2 | 40 | 38 (95.0 %) | 2 (5.0 %) | 0 | 2 |
| Noisy set | 40 | 38 (95.0 %) | 2 (5.0 %) | 1 | 2 |

### Reproduced Hit@6 against the saved v1.1 rows
| Set | n | same | different | saved row missing |
|---|---|---|---|---|
| Stage 1 | 30 | 30 | 0 | 0 |
| Stage 2 | 40 | 39 | 1 | 0 |
| Noisy set | 40 | 39 | 1 | 0 |

### Items whose reproduced Hit@6 differs from the saved v1.1 row
| Set | Id | Language | French query | Saved v1.1 Hit@6 | Reproduced Hit@6 | Reproduced class |
|---|---|---|---|---|---|---|
| Stage 2 | ac16 | ar | translation | True | False | in pool, reranked 7-10 |
| Noisy set | n_ac42 | ar | translation | False | True | found in top 6 (reranked) |

### v2 evidence where the reproduced v1.1 funnel missed (analysed items, not in top 6)
| Set | v1.1 missed | v2 evidence has the gold | v2 status clarify/refuse | v2 row missing |
|---|---|---|---|---|
| Stage 1 | 8 | 1 | 0 | 0 |
| Stage 2 | 11 | 6 | 0 | 0 |
| Noisy set | 21 | 8 | 0 | 0 |

### v2 evidence where the saved v1.1 row missed (all answerable items, every language)
| Set | saved v1.1 missed | v2 evidence has the gold | v2 row missing |
|---|---|---|---|
| Stage 1 | 8 | 1 | 0 |
| Stage 2 | 10 | 5 | 0 |
| Noisy set | 22 | 8 | 0 |

### Cross-references in the index
| Reference kind | Chunks | Share of the index |
|---|---|---|
| numbered | 1115 | 18.5 % |
| relative | 160 | 2.7 % |
| vague | 288 | 4.8 % |
| any | 1409 | 23.4 % |
| chunks in the index | 6031 | 100 % |

### Cross-references in the articles cited by saved v2 answers ((answer, cited id) pairs)
| Measure | Value |
|---|---|
| pairs | 233 |
| cited id not in the index | 0 |
| distinct cited ids | 178 |
| pairs whose cited article contains a reference | 70 (30.0 %) |
| ... numbered (article N, articles N à M) | 60 (25.8 %) |
| ... relative (article précédent / suivant) | 11 (4.7 %) |
| ... vague (ci-dessus, susvisé, précité, ci-après) | 11 (4.7 %) |
| pairs with a resolvable reference | 69 (29.6 %) |
| pairs with a referenced article not in that answer's evidence | 53 (22.7 %) |

### Cross-references in cited articles per set
| Set | pairs | with a reference | referenced article not in the evidence |
|---|---|---|---|
| Stage 1 | 63 | 17 (27.0 %) | 16 (25.4 %) |
| Stage 2 | 89 | 24 (27.0 %) | 16 (18.0 %) |
| Noisy set | 81 | 29 (35.8 %) | 21 (25.9 %) |

### Chunk start
| Start | Chunks | Share |
|---|---|---|
| article header | 5962 | 98.9 % |
| section/chapter title | 0 | 0.0 % |
| other | 69 | 1.1 % |
| a section/chapter/title line anywhere in the chunk | 736 | 12.2 % |

### Chunk length
| Unit | min | median | mean | p90 | max |
|---|---|---|---|---|---|
| characters | 48 | 361 | 495.0 | 949 | 12101 |
| tokens (embedding model tokenizer, special tokens included) | 15 | 94 | 125.6 | 239 | 3354 |

### Chunks longer than the embedding model's max_seq_length
| max_seq_length | chunks over it | share |
|---|---|---|
| 512 | 63 | 1.0 % |

### Ten random chunks (seed 20261022), first 300 characters
| Chunk id | First 300 characters |
|---|---|
| travailFr/s01/226 | Article 226.- Le président ouvre et dirige les débats.  ⏎ Jusqu'à la clôture des débats, les parties ou leurs avocats peuvent  ⏎ présenter toutes conclusions écrites précisant leurs moyens d'appel,  ⏎ après en avoir communiqué un exemplaire aux adversaires ou à leurs  ⏎ avocats.  ⏎ Dans ce cas, il peut être d |
| PenalFr/s02/94 | Article 94.- Dans tous les cas de corruption, les choses données ou  ⏎ reçues sont confisquées au profit de l'Etat.  ⏎ Section III - La concussion |
| dtreelFr/s05/3-bis | Article 3 bis (Ajouté par la loi n° 2000-10 du 24 janvier 2000).-  ⏎ L’octroi du certificat de possession est subordonné à l’engagement de  ⏎ celui qui le demande à réaliser un projet de mise en valeur ou une  ⏎ action de développement sur la terre objet de la demande du certificat  ⏎ de possession.  ⏎ Toute  |
| EnfantFr/s03/20 | Article 20.-  ⏎ 1. - Tout enfant qui est temporairement ou définitivement  ⏎ privé de son milieu familial, ou qui dans son propre intérêt ne ⏎  ⏎   ⏎ peut être laissé dans ce milieu, a droit à une protection et une  ⏎ aide spéciales de l'Etat.  ⏎ 2. - Les Etats parties prévoient pour cet enfant une protection  ⏎ de |
| COC/s02/827 | Article 827.- Si le fermier à complété de ses derniers l'outillage  ⏎ destiné à l'exploitation par d'autres objets non compris dans  ⏎ l'inventaire, le propriétaire aura le choix, à la fin du bail, de lui en  ⏎ rembourser la valeur à dire d'experts, ou de les restituer au fermier en  ⏎ l'état où ils se trou |
| COC/s02/1258 | Article 1258.- L’apport doit être spécifié et déterminé ; lorsqu’il  ⏎ consiste dans tous les biens présents de l’un des associés, ces biens  ⏎ doivent être inventoriés. Si l’apport consiste en choses autres que du  ⏎ numéraire, elles doivent être estimées à la valeur du jour où elles ont  ⏎ été mises dans  |
| CommerceFr/s01/630 | Article 630  ⏎ Le destinataire, s'il est distinct de l'expéditeur, n'est tenu des  ⏎ obligations nées du contrat de transport que par son acceptation,  ⏎ expresse ou tacite, donnée au transporteur. |
| CommerceFr/s01/539 | Article 539  ⏎ Le syndic devra poursuivre la vente des meubles de toute  ⏎ nature, y compris le fonds de commerce dans les formes  ⏎ prescrites par l'article 494, mais sans qu'il soit besoin d'appeler  ⏎ le failli. |
| CommerceFr/s01/573 | Article 573  ⏎ En cas de résiliation des baux prévus à l'article précédent, le  ⏎ propriétaire a privilège pour les deux dernières années de  ⏎ location, échues avant le jugement déclaratif de faillite, et pour  ⏎ l'année courante, pour tout ce qui concerne l'exécution du bail et  ⏎ pour tous les dommages-int |
| ArbitrageFr/s01/27 | Article 27  ⏎ Si, devant le tribunal arbitral, est soulevée une question  ⏎ préjudicielle ne rentrant pas dans les limites de sa compétence, ⏎  ⏎   ⏎ mais liée à l'arbitrage, le tribunal arbitral sursoit à statuer  ⏎ jusqu'à ce que la juridiction saisie rende sa décision. Dans ce  ⏎ cas le délai imparti pour rend |

### Empty chunks and '(no PDF text)' chunks
| Kind | Chunks | Ids |
|---|---|---|
| empty text | 0 | - |
| '(no PDF text)' | 0 | - |

### Run
| Item | Value |
|---|---|
| index | db_v1_1 (6031 chunks) |
| index opened | byte copy of db_v1_1 |
| db_v1_1/chroma.sqlite3 bytes identical before and after the run | True |
| French queries of Arabic/English items | diagnostics/translations.json (47 items) |
| system fingerprint | dc8fd664c66230b19fa88aad2195be1ab8b67c5be5540a010ba091c4537ec5ca |
| embedding model | OrdalieTech/Solon-embeddings-large-0.1 |
| reranker | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 |
| MMR_K / BM25_K / RERANK_POOL / RERANK_TOP_K | 20 / 15 / 20 / 6 |
| LLM calls in this run | 0 (translations made once by diagnostics/translate_skipped.py) |

### Index file after the run (diagnostics/chroma_open_check.py, one Chroma open of a temporary copy of db_v1_1)
| Measure | Value |
|---|---|
| tables in the index | 21 |
| tables unchanged by one open | 20 |
| table changed by one open: acquire_write | rows 124 -> 125 |
| acquire_write rows in the real index now (read-only count) | 124 |
| embeddings rows in the real index (read-only count) | 6031 |

