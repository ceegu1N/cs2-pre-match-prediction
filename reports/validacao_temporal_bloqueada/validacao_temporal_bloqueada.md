# Validação temporal bloqueada

Esta validação reaplica a configuração final congelada em janelas temporais expansivas da própria base local. Ela mede estabilidade retrospectiva, não substitui o holdout principal e não é avaliação prospectiva.

## Configuração

- Modelo: `univariate_top_11__logreg_l2_c0.3_cwnone`
- Classificador: regressão logística L2
- C: 0,3
- class_weight: None
- Normalização: z-score via StandardScaler no pipeline
- Limiar: 0,5
- Variáveis: 11

## Resultados por janela

| Treino | Teste | n treino | n teste | baseline | ROC-AUC | accuracy | precision | recall | F1 | log-loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024_s1 | 2024_s2 | 574 | 670 | 0,5478 | 0,6456 | 0,6015 | 0,5559 | 0,5908 | 0,5728 | 0,6593 | 0,2336 |
| 2024_s1+2024_s2 | 2025_s1 | 1244 | 688 | 0,5000 | 0,6476 | 0,6003 | 0,6075 | 0,5669 | 0,5865 | 0,6546 | 0,2320 |
| 2024_s1+2024_s2+2025_s1 | 2025_s2 | 1932 | 614 | 0,5228 | 0,6695 | 0,6026 | 0,5854 | 0,5734 | 0,5793 | 0,6418 | 0,2260 |
| 2024_s1+2024_s2+2025_s1+2025_s2 | 2026_s1 | 2546 | 294 | 0,5238 | 0,6828 | 0,6293 | 0,6165 | 0,5857 | 0,6007 | 0,6388 | 0,2243 |

## Matrizes de confusão

| Janela de teste | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| 2024_s2 | 224 | 143 | 124 | 179 |
| 2025_s1 | 218 | 126 | 149 | 195 |
| 2025_s2 | 202 | 119 | 125 | 168 |
| 2026_s1 | 103 | 51 | 58 | 82 |

## Leitura

A média de ROC-AUC nas quatro janelas foi 0,6614, com variação de 0,6456 a 0,6828. A média de accuracy foi 0,6084, variando de 0,6003 a 0,6293. Esses resultados devem ser lidos como evidência de estabilidade temporal retrospectiva do modelo congelado, e não como novo processo de seleção.

## Recomendação

Os resultados são suficientemente coerentes para entrar no TCC como uma subseção curta de robustez temporal bloqueada. A redação deve deixar claro que o holdout 2026_s1 continua sendo a avaliação principal.
