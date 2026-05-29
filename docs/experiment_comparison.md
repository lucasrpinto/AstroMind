# Comparação de Experimentos — AstroMind

Este arquivo compara os experimentos registrados nos logs acumulativos.

## Ranking geral

| Posição | Modelo | Train Run ID | Test Acc | External Acc | Best Val Acc | Nebulosa Externa | Galáxia Externa | Promovível |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | AstroMindCNNV2.4 | `train_20260527_163858_a7c772fa` | 80.77% | 90.48% | 85.71% | 0.666667(2/3) | 0.833333(5/6) | False |
| 2 | AstroMindCNNV2.4 | `train_20260528_110538_c209d69b` | 80.77% | 90.48% | 85.71% | 0.666667(2/3) | 0.833333(5/6) | False |
| 3 | AstroMindCNNV2 | `train_20260526_102743_194ad6ed` | 84.62% | 85.71% | 75.00% | 0.000000(0/3) | 1.000000(6/6) | False |
| 4 | AstroMindCNNV2.1 | `train_20260526_150026_e7870f64` | 100.00% | 80.95% | 75.00% | 0.000000(0/3) | 0.833333(5/6) | False |
| 5 | AstroMindCNNV1 | `train_20260526_092834_a7911d86` | 92.31% | 80.95% | 91.67% | 0.000000(0/3) | 0.833333(5/6) | False |
| 6 | AstroMindCNNV1 | `train_20260526_100239_364d4d4b` | 92.31% | 80.95% | 91.67% | 0.000000(0/3) | 0.833333(5/6) | False |
| 7 | AstroMindCNNV2.5 | `train_20260528_114121_f3cdc353` | 66.67% | 80.95% | 71.88% | 0.666667(2/3) | 0.500000(3/6) | True |
| 8 | AstroMindCNNV2.4 | `train_20260528_141140_ef69acef` | 66.67% | 80.95% | 71.88% | 0.666667(2/3) | 0.500000(3/6) | False |
| 9 | AstroMindCNNV2.4 | `train_20260528_153202_6c79d450` | 66.67% | 80.95% | 71.88% | 0.666667(2/3) | 0.500000(3/6) | True |
| 10 | AstroMindCNNV2.2 | `train_20260527_085121_448e3459` | 76.92% | 76.19% | 78.57% | 0.333333(1/3) | 0.500000(3/6) | False |
| 11 | AstroMindCNNV2.3 | `train_20260527_155905_0612b48a` | 65.38% | 76.19% | 64.29% | 0.000000(0/3) | 0.666667(4/6) | False |

## Melhor experimento até o momento

- Modelo: **AstroMindCNNV2.4**
- Train Run ID: `train_20260527_163858_a7c772fa`
- Test Accuracy: 80.77%
- External Accuracy: 90.48%
- Best Validation Accuracy: 85.71%
- Nebulosa externa: 0.666667(2/3)
- Galáxia externa: 0.833333(5/6)
- Promovível: False

## Melhor experimento promovível

- Modelo: **AstroMindCNNV2.5**
- Train Run ID: `train_20260528_114121_f3cdc353`
- Test Accuracy: 66.67%
- External Accuracy: 80.95%
- Best Validation Accuracy: 71.88%
- Nebulosa externa: 0.666667(2/3)
- Galáxia externa: 0.500000(3/6)
- Checkpoint: `C:\Workspace\Python\AstroMind\AstroMind\models\astronomy_classifier_best.pth`

## Observações

- O ranking prioriza a acurácia externa, pois ela mede melhor a generalização.
- Em caso de empate, considera-se acurácia de teste interno e depois validação.
- Resultados com poucos exemplos externos devem ser interpretados com cautela.