# Wave 2: query normalization variants (retrieval only)

Development data (110 items), not evidence of generalisation. Pre-registration: diagnostics/WAVE2_PREREG.md.

### Normalization calls
| Items | ok | failures (fallback to the input query) | 2 attempts | input tokens | output tokens | total tokens | mean total / item | prompt sha256 |
|---|---|---|---|---|---|---|---|---|
| 110 | 109 | 1 | 2 | 55912 | 11760 | 67672 | 615 | 1c27f8922409 |

### Failed normalizations
| Id | Status | Error |
|---|---|---|
| ac69 | invalid | empty legal_query |

### Flagged outputs (numbers not in the input query; more than 40 words)
| Id | New numbers | Words | Input query | legal_query |
|---|---|---|---|---|
| ac05 | 3000 | 20 | J'ai confié à un ami, sans rien signer, des bijoux qui valent environ 3 000 dinars. La loi exigeait-elle un écrit pour cette remise en garde, et y a-t-il des situations où l'on peut s'en passer ? | exigence d'un écrit pour le contrat de dépôt de bijoux d’une valeur d’environ 3000 dinars et exceptions à l’obligation d’écrit |

### 12 random rows (random.Random(20261022).sample)
| Id | Set | Language | Input query | legal_query | Flags |
|---|---|---|---|---|---|
| ac02 | S1 | ar | Dans un arbitrage international, j’ai remarqué qu’une clause du compromis d’arbitrage n’a pas été respectée, mais j’ai poursuivi la procédure sans m’y opposer. Puis-je invoquer cette violation ultérieurement ? | possibilité de soulever ultérieurement la violation d’une clause du compromis d’arbitrage non contestée | - |
| ac15 | S1 | fr | J'ai payé un effet de commerce à la place d'un des signataires pour lui rendre service. Quels recours ai-je ensuite, et puis-je le remettre en circulation en le transmettant à quelqu'un ? | effet de commerce payé par un tiers non signataire, recours du payeur contre le débiteur et possibilité de remise en circulation par endossement | - |
| ac55 | S1 | fr | Que doit obligatoirement contenir le compte rendu écrit d'une assemblée générale d'actionnaires ? | contenu obligatoire du compte rendu écrit d'une assemblée générale d'actionnaires selon le Code des sociétés commerciales | - |
| ac68 | S1 | ar | Quelles sont les structures qui prennent en charge les intérêts locaux et régionaux en Tunisie ? | structures de la collectivité territoriale tunisienne chargées de la prise en charge des intérêts locaux et régionaux | - |
| ac01 | S2 | fr | J'ai signé avec une société étrangère un contrat qui prévoit un arbitrage international, mais elle m'a assigné devant un tribunal tunisien. Jusqu'à quand puis-je demander que l'affaire soit renvoyée aux arbitres ? | délai de demande de renvoi d'une affaire au tribunal tunisien vers l'arbitrage international prévu par le contrat | - |
| ac09 | S2 | en | Deux personnes détiennent chacune un acte écrit pour le même bien et aucune d’elles ne l’occupe. Laquelle des réclamations prévaut‑elle ? | principe de priorité entre deux titres de propriété sur le même bien non occupé, ordre des inscriptions au registre foncier | - |
| ac28 | S2 | fr | Un journal a publié la photo et le nom d'un adolescent poursuivi devant le juge. Que risque l'auteur de la publication ? | responsabilité pénale pour atteinte à la vie privée et diffamation d’un mineur par publication de sa photo et de son nom lors d’une procédure judiciaire | - |
| n_ac07 | N | ar | J'ai convenu avec une personne de garder mon bien pendant un an contre rémunération, mais le contrat a pris fin après seulement quatre mois. La personne a-t-elle droit à la rémunération convenue en totalité ? | rémunération due dans le contrat de dépôt rémunéré interrompu avant l'échéance de quatre mois | - |
| n_ac31 | N | fr | Un site internet balance des photos et des infos sur un enfant impliqué dans un procès. Que peut faire la justice pour arrêter ça et, si c'est très urgent, qui peut porter plainte devant le juge de permanence ? | violation du droit à l'image et de la vie privée d’un mineur par diffusion de photos et d’informations sur internet, mesures d’urgence et possibilité de dépôt de plainte par le représentant légal du mineur devant le juge de permanence | - |
| n_ac38 | N | fr | Quelqu'un part en fugue avec un enfant de 12 ans, sans le forcer ni le tromper, loin de la famille à qui l'enfant était confié. Quelle sanction risque-t-il ? | sanction de l'enlèvement d'enfant mineur de 12 ans sans contrainte ni tromperie | - |
| n_ac61 | N | en | Mon enfant suit des cours de formation professionnelle mais n'a jamais signé de contrat d'apprentissage, les règles d'apprentissage s'appliquent-elles toujours à lui ou nah? | application des dispositions du Code du travail relatives à l'apprentissage en l'absence de contrat d'apprentissage | - |
| n_ac70 | N | en | Quand les députés tunisiens s'asseyent-ils réellement chaque année, et si ils sont en vacances qui peut les faire revenir ? | convocation et tenue des séances annuelles de l'Assemblée des représentants du peuple ainsi que pouvoir de rappel des députés pendant les périodes de vacances | - |

### Hit@6 per group, 95% bootstrap CI (5000 resamples, seed 20261022)
| Group | n | V0 | V1 | V2 | V3 |
|---|---|---|---|---|---|
| all | 110 | 75 (68.2 %) [59.1, 77.3] | 82 (74.5 %) [66.4, 82.7] | 82 (74.5 %) [66.4, 82.7] | 85 (77.3 %) [69.1, 84.5] |
| language ar | 24 | 19 (79.2 %) [62.5, 95.8] | 17 (70.8 %) [54.2, 87.5] | 20 (83.3 %) [66.7, 95.8] | 17 (70.8 %) [54.2, 87.5] |
| language en | 23 | 16 (69.6 %) [52.2, 87.0] | 17 (73.9 %) [56.5, 91.3] | 17 (73.9 %) [56.5, 91.3] | 18 (78.3 %) [60.9, 95.7] |
| language fr | 63 | 40 (63.5 %) [52.4, 74.6] | 48 (76.2 %) [65.1, 85.7] | 45 (71.4 %) [60.3, 82.5] | 50 (79.4 %) [68.3, 88.9] |
| noise describe | 10 | 4 (40.0 %) [10.0, 70.0] | 5 (50.0 %) [20.0, 80.0] | 3 (30.0 %) [10.0, 60.0] | 5 (50.0 %) [20.0, 80.0] |
| noise informal | 10 | 6 (60.0 %) [30.0, 90.0] | 7 (70.0 %) [40.0, 100.0] | 7 (70.0 %) [40.0, 100.0] | 7 (70.0 %) [40.0, 100.0] |
| noise typos | 10 | 7 (70.0 %) [40.0, 100.0] | 7 (70.0 %) [40.0, 100.0] | 8 (80.0 %) [50.0, 100.0] | 7 (70.0 %) [40.0, 100.0] |
| noise wrong_term | 10 | 6 (60.0 %) [30.0, 90.0] | 7 (70.0 %) [40.0, 100.0] | 8 (80.0 %) [50.0, 100.0] | 7 (70.0 %) [40.0, 100.0] |
| set N | 40 | 23 (57.5 %) [42.5, 72.5] | 26 (65.0 %) [50.0, 80.0] | 26 (65.0 %) [50.0, 80.0] | 26 (65.0 %) [50.0, 80.0] |
| set S1 | 30 | 21 (70.0 %) [53.3, 86.7] | 27 (90.0 %) [80.0, 100.0] | 24 (80.0 %) [66.7, 93.3] | 28 (93.3 %) [83.3, 100.0] |
| set S2 | 40 | 31 (77.5 %) [65.0, 90.0] | 29 (72.5 %) [57.5, 85.0] | 32 (80.0 %) [67.5, 92.5] | 31 (77.5 %) [65.0, 90.0] |

### Hit@8 per group, 95% bootstrap CI (5000 resamples, seed 20261022)
| Group | n | V0 | V1 | V2 | V3 |
|---|---|---|---|---|---|
| all | 110 | 80 (72.7 %) [63.6, 80.9] | 86 (78.2 %) [70.0, 85.5] | 85 (77.3 %) [69.1, 84.5] | 91 (82.7 %) [75.5, 89.1] |
| language ar | 24 | 20 (83.3 %) [66.7, 95.8] | 18 (75.0 %) [58.3, 91.7] | 20 (83.3 %) [66.7, 95.8] | 19 (79.2 %) [62.5, 95.8] |
| language en | 23 | 16 (69.6 %) [52.2, 87.0] | 18 (78.3 %) [60.9, 95.7] | 17 (73.9 %) [56.5, 91.3] | 18 (78.3 %) [60.9, 95.7] |
| language fr | 63 | 44 (69.8 %) [58.7, 81.0] | 50 (79.4 %) [69.8, 88.9] | 48 (76.2 %) [65.1, 85.7] | 54 (85.7 %) [76.2, 93.7] |
| noise describe | 10 | 4 (40.0 %) [10.0, 70.0] | 5 (50.0 %) [20.0, 80.0] | 4 (40.0 %) [10.0, 70.0] | 6 (60.0 %) [30.0, 90.0] |
| noise informal | 10 | 6 (60.0 %) [30.0, 90.0] | 7 (70.0 %) [40.0, 100.0] | 7 (70.0 %) [40.0, 100.0] | 7 (70.0 %) [40.0, 100.0] |
| noise typos | 10 | 8 (80.0 %) [50.0, 100.0] | 7 (70.0 %) [40.0, 100.0] | 8 (80.0 %) [50.0, 100.0] | 8 (80.0 %) [50.0, 100.0] |
| noise wrong_term | 10 | 8 (80.0 %) [50.0, 100.0] | 8 (80.0 %) [50.0, 100.0] | 8 (80.0 %) [50.0, 100.0] | 7 (70.0 %) [40.0, 100.0] |
| set N | 40 | 26 (65.0 %) [50.0, 80.0] | 27 (67.5 %) [52.5, 82.5] | 27 (67.5 %) [52.5, 82.5] | 28 (70.0 %) [55.0, 85.0] |
| set S1 | 30 | 23 (76.7 %) [60.0, 90.0] | 28 (93.3 %) [83.3, 100.0] | 25 (83.3 %) [70.0, 96.7] | 29 (96.7 %) [90.0, 100.0] |
| set S2 | 40 | 31 (77.5 %) [65.0, 90.0] | 31 (77.5 %) [65.0, 90.0] | 33 (82.5 %) [70.0, 92.5] | 34 (85.0 %) [72.5, 95.0] |

### Hit@10 per group, 95% bootstrap CI (5000 resamples, seed 20261022)
| Group | n | V0 | V1 | V2 | V3 |
|---|---|---|---|---|---|
| all | 110 | 82 (74.5 %) [66.4, 82.7] | 89 (80.9 %) [73.6, 88.2] | 89 (80.9 %) [72.7, 88.2] | 94 (85.5 %) [78.2, 91.8] |
| language ar | 24 | 20 (83.3 %) [66.7, 95.8] | 19 (79.2 %) [62.5, 91.7] | 21 (87.5 %) [75.0, 100.0] | 20 (83.3 %) [66.7, 95.8] |
| language en | 23 | 16 (69.6 %) [52.2, 87.0] | 18 (78.3 %) [60.9, 95.7] | 17 (73.9 %) [56.5, 91.3] | 18 (78.3 %) [60.9, 95.7] |
| language fr | 63 | 46 (73.0 %) [61.9, 84.1] | 52 (82.5 %) [73.0, 92.1] | 51 (81.0 %) [71.4, 90.5] | 56 (88.9 %) [81.0, 95.2] |
| noise describe | 10 | 4 (40.0 %) [10.0, 70.0] | 5 (50.0 %) [20.0, 80.0] | 4 (40.0 %) [10.0, 70.0] | 6 (60.0 %) [30.0, 90.0] |
| noise informal | 10 | 6 (60.0 %) [30.0, 90.0] | 8 (80.0 %) [50.0, 100.0] | 7 (70.0 %) [40.0, 100.0] | 7 (70.0 %) [40.0, 100.0] |
| noise typos | 10 | 8 (80.0 %) [50.0, 100.0] | 8 (80.0 %) [50.0, 100.0] | 9 (90.0 %) [70.0, 100.0] | 9 (90.0 %) [70.0, 100.0] |
| noise wrong_term | 10 | 8 (80.0 %) [50.0, 100.0] | 8 (80.0 %) [50.0, 100.0] | 9 (90.0 %) [70.0, 100.0] | 8 (80.0 %) [50.0, 100.0] |
| set N | 40 | 26 (65.0 %) [50.0, 80.0] | 29 (72.5 %) [57.5, 85.0] | 29 (72.5 %) [57.5, 85.0] | 30 (75.0 %) [60.0, 87.5] |
| set S1 | 30 | 24 (80.0 %) [66.7, 93.3] | 28 (93.3 %) [83.3, 100.0] | 27 (90.0 %) [80.0, 100.0] | 29 (96.7 %) [90.0, 100.0] |
| set S2 | 40 | 32 (80.0 %) [67.5, 92.5] | 32 (80.0 %) [67.5, 92.5] | 33 (82.5 %) [70.0, 92.5] | 35 (87.5 %) [77.5, 97.5] |

### Funnel classes (all items)
| Variant | n | top6 | rerank_7_10 | rerank_11_20 | fused_rank_gt20 | absent_from_pool | gold in evidence | pool size mean |
|---|---|---|---|---|---|---|---|---|
| V0 | 110 | 75 | 7 | 6 | 4 | 18 | 78 | 20.0 |
| V1 | 110 | 82 | 7 | 4 | 2 | 15 | 83 | 20.0 |
| V2 | 110 | 82 | 7 | 6 | 4 | 11 | 83 | 20.0 |
| V3 | 110 | 85 | 9 | 1 | 4 | 11 | 86 | 20.0 |

### V0: funnel classes per group
| Group | n | top6 | rerank_7_10 | rerank_11_20 | fused_rank_gt20 | absent_from_pool |
|---|---|---|---|---|---|---|
| all | 110 | 75 | 7 | 6 | 4 | 18 |
| language ar | 24 | 19 | 1 | 1 | 0 | 3 |
| language en | 23 | 16 | 0 | 1 | 0 | 6 |
| language fr | 63 | 40 | 6 | 4 | 4 | 9 |
| noise describe | 10 | 4 | 0 | 1 | 0 | 5 |
| noise informal | 10 | 6 | 0 | 0 | 1 | 3 |
| noise typos | 10 | 7 | 1 | 1 | 0 | 1 |
| noise wrong_term | 10 | 6 | 2 | 0 | 1 | 1 |
| set N | 40 | 23 | 3 | 2 | 2 | 10 |
| set S1 | 30 | 21 | 3 | 3 | 0 | 3 |
| set S2 | 40 | 31 | 1 | 1 | 2 | 5 |

### V1: funnel classes per group
| Group | n | top6 | rerank_7_10 | rerank_11_20 | fused_rank_gt20 | absent_from_pool |
|---|---|---|---|---|---|---|
| all | 110 | 82 | 7 | 4 | 2 | 15 |
| language ar | 24 | 17 | 2 | 2 | 0 | 3 |
| language en | 23 | 17 | 1 | 0 | 0 | 5 |
| language fr | 63 | 48 | 4 | 2 | 2 | 7 |
| noise describe | 10 | 5 | 0 | 1 | 1 | 3 |
| noise informal | 10 | 7 | 1 | 0 | 0 | 2 |
| noise typos | 10 | 7 | 1 | 1 | 0 | 1 |
| noise wrong_term | 10 | 7 | 1 | 0 | 0 | 2 |
| set N | 40 | 26 | 3 | 2 | 1 | 8 |
| set S1 | 30 | 27 | 1 | 0 | 0 | 2 |
| set S2 | 40 | 29 | 3 | 2 | 1 | 5 |

### V2: funnel classes per group
| Group | n | top6 | rerank_7_10 | rerank_11_20 | fused_rank_gt20 | absent_from_pool |
|---|---|---|---|---|---|---|
| all | 110 | 82 | 7 | 6 | 4 | 11 |
| language ar | 24 | 20 | 1 | 0 | 0 | 3 |
| language en | 23 | 17 | 0 | 1 | 1 | 4 |
| language fr | 63 | 45 | 6 | 5 | 3 | 4 |
| noise describe | 10 | 3 | 1 | 2 | 1 | 3 |
| noise informal | 10 | 7 | 0 | 0 | 2 | 1 |
| noise typos | 10 | 8 | 1 | 0 | 0 | 1 |
| noise wrong_term | 10 | 8 | 1 | 0 | 0 | 1 |
| set N | 40 | 26 | 3 | 2 | 3 | 6 |
| set S1 | 30 | 24 | 3 | 2 | 0 | 1 |
| set S2 | 40 | 32 | 1 | 2 | 1 | 4 |

### V3: funnel classes per group
| Group | n | top6 | rerank_7_10 | rerank_11_20 | fused_rank_gt20 | absent_from_pool |
|---|---|---|---|---|---|---|
| all | 110 | 85 | 9 | 1 | 4 | 11 |
| language ar | 24 | 17 | 3 | 1 | 0 | 3 |
| language en | 23 | 18 | 0 | 0 | 1 | 4 |
| language fr | 63 | 50 | 6 | 0 | 3 | 4 |
| noise describe | 10 | 5 | 1 | 0 | 1 | 3 |
| noise informal | 10 | 7 | 0 | 0 | 2 | 1 |
| noise typos | 10 | 7 | 2 | 0 | 0 | 1 |
| noise wrong_term | 10 | 7 | 1 | 1 | 0 | 1 |
| set N | 40 | 26 | 4 | 1 | 3 | 6 |
| set S1 | 30 | 28 | 1 | 0 | 0 | 1 |
| set S2 | 40 | 31 | 4 | 0 | 1 | 4 |

### Paired differences versus V0 (95% bootstrap CI of the mean difference; exact two-sided sign test)
| Variant | k | V0 hits | variant hits | difference | 95% CI | gained | lost | sign test p |
|---|---|---|---|---|---|---|---|---|
| V1 | Hit@6 | 75 | 82 | +6.4 pp | [-2.7, +15.5] | 16 | 9 | 0.230 |
| V1 | Hit@8 | 80 | 86 | +5.5 pp | [-2.7, +13.6] | 14 | 8 | 0.286 |
| V1 | Hit@10 | 82 | 89 | +6.4 pp | [-1.8, +14.5] | 14 | 7 | 0.189 |
| V2 | Hit@6 | 75 | 82 | +6.4 pp | [+0.0, +12.7] | 10 | 3 | 0.092 |
| V2 | Hit@8 | 80 | 85 | +4.5 pp | [+0.0, +9.1] | 6 | 1 | 0.125 |
| V2 | Hit@10 | 82 | 89 | +6.4 pp | [+0.9, +12.7] | 9 | 2 | 0.065 |
| V3 | Hit@6 | 75 | 85 | +9.1 pp | [+0.9, +17.3] | 16 | 6 | 0.052 |
| V3 | Hit@8 | 80 | 91 | +10.0 pp | [+2.7, +17.3] | 14 | 3 | 0.013 |
| V3 | Hit@10 | 82 | 94 | +10.9 pp | [+4.5, +17.3] | 13 | 1 | 0.002 |

### Gained and lost items versus V0 (Hit@6)
| Variant | gained | lost |
|---|---|---|
| V1 | ac10, ac14, ac15, ac33, ac41, ac56, ac08, ac19, ac20, n_ac08, n_ac16, n_ac20, n_ac25, n_ac40, n_ac46, n_ac65 | ac04, ac30, ac31, ac39, ac40, n_ac01, n_ac05, n_ac26, n_ac30 |
| V2 | ac10, ac15, ac41, ac20, ac32, n_ac08, n_ac18, n_ac20, n_ac25, n_ac65 | ac60, n_ac38, n_ac53 |
| V3 | ac06, ac10, ac14, ac15, ac33, ac43, ac56, ac08, ac19, ac20, n_ac08, n_ac16, n_ac20, n_ac25, n_ac46, n_ac65 | ac04, ac30, ac39, n_ac01, n_ac26, n_ac30 |

### Decision rule (pre-registered)
| Variant | total Hit@6 (V0) | extra tokens | (i) noisy net vs V0 >= +3 | (ii) lost S1+S2 <= 1 | (iii) lost per language <= 1 | passes |
|---|---|---|---|---|---|---|
| V1 | 82 (75) | 67672 | +3 (yes) | 5 (ac04, ac30, ac31, ac39, ac40) (no) | ar: 3 (ac04, ac30, n_ac30), en: 1 (ac40), fr: 5 (ac31, ac39, n_ac01, n_ac05, n_ac26) (no) | no |
| V2 | 82 (75) | 67672 | +3 (yes) | 1 (ac60) (yes) | ar: 0, en: 0, fr: 3 (ac60, n_ac38, n_ac53) (no) | no |
| V3 | 85 (75) | 67672 | +3 (yes) | 3 (ac04, ac30, ac39) (no) | ar: 3 (ac04, ac30, n_ac30), en: 0, fr: 3 (ac39, n_ac01, n_ac26) (no) | no |

| Candidate (highest total Hit@6) | Selected |
|---|---|
| V3 | none |

