# Histórico de Experimentos — AstroMind

Este documento é gerado automaticamente a partir dos logs acumulativos do projeto.

Arquivos de origem:

- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\train_runs.csv`
- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\evaluate_runs.csv`
- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\predict_runs.csv`
- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\external_evaluate_runs.csv`

## 1. Histórico de treinos

Total de treinos registrados: **3**

| Run ID | Data | Modelo | Treino | Validação | Épocas | Melhor época | Best Val Loss | Best Val Acc | Final Train Acc | Final Val Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Run ID | Data | Treino | Validação | Épocas | Melhor época | Best Val Loss | Best Val Acc | Final Train Acc | Final Val Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_20260526_092834_a7911d86 | 2026-05-26 09:29:44 | AstroMindCNNV1 | 57 | 12 | 30 | 28 | 0.438775 | 91.67% | 71.67% | 75.00% |
| train_20260526_100239_364d4d4b | 2026-05-26 10:04:00 | AstroMindCNNV1 | 57 | 12 | 30 | 28 | 0.438775 | 91.67% | 71.67% | 75.00% |
| train_20260526_102743_194ad6ed | 2026-05-26 10:33:14 | AstroMindCNNV2 | 57 | 12 | 30 | 20 | 0.654141 | 75.00% | 58.33% | 83.33% |

**Melhor treino registrado:**

- Run ID: `train_20260526_092834_a7911d86`
- Data: 2026-05-26 09:29:44
- Modelo: AstroMindCNNV1
- Melhor época: 28
- Melhor validation accuracy: 91.67%
- Melhor validation loss: 0.438775

## 2. Histórico de avaliações no conjunto de teste

Total de avaliações registradas: **3**

| Evaluate Run ID | Train Run ID | Data | Dataset | Acurácia | Acertos | Erros |
|---|---|---:|---:|---:|---:|---:|
| evaluate_20260526_093313_ffed3e97 | train_20260526_092834_a7911d86 | 2026-05-26 09:33:13 | 13 | 92.31% | 12 | 1 |
| evaluate_20260526_100417_eb9470af | train_20260526_100239_364d4d4b | 2026-05-26 10:04:17 | 13 | 92.31% | 12 | 1 |
| evaluate_20260526_115218_049a5553 | train_20260526_102743_194ad6ed | 2026-05-26 11:52:19 | 13 | 84.62% | 11 | 2 |

**Melhor avaliação registrada:**

- Evaluate Run ID: `evaluate_20260526_093313_ffed3e97`
- Train Run ID: `train_20260526_092834_a7911d86`
- Acurácia: 92.31%
- Acertos: 12
- Erros: 1

## 3. Histórico de avaliações externas

Total de avaliações externas registradas: **3**

| External Run ID | Train Run ID | Data | Imagens | Imagens rotuladas | Acurácia externa | Acertos | Erros |
|---|---|---:|---:|---:|---:|---:|---:|
| external_evaluate_20260526_093829_9ae41a04 | train_20260526_092834_a7911d86 | 2026-05-26 09:38:30 | 21 | 21 | 80.95% | 17 | 4 |
| external_evaluate_20260526_100618_2c58c09f | train_20260526_100239_364d4d4b | 2026-05-26 10:06:18 | 21 | 21 | 80.95% | 17 | 4 |
| external_evaluate_20260526_115646_755f6152 | train_20260526_102743_194ad6ed | 2026-05-26 11:56:48 | 21 | 21 | 85.71% | 18 | 3 |

## 4. Histórico de predições individuais

Total de predições registradas: **3**

Últimas 10 predições:

| Predict Run ID | Data | Classe real | Previsto | Confiança | Acertou |
|---|---:|---:|---:|---:|---:|
| predict_20260526_093440_0237fa47 | 2026-05-26 09:34:41 | aglomerado | aglomerado | 92.87% | True |
| predict_20260526_100510_956a2cc3 | 2026-05-26 10:05:10 | aglomerado | aglomerado | 92.87% | True |
| predict_20260526_115607_874d9dc6 | 2026-05-26 11:56:08 | aglomerado | galaxia | 56.62% | False |

## 5. Observações

- As métricas de validação acompanham o desempenho durante o treino.
- As métricas de teste medem o desempenho em imagens separadas do treino.
- As avaliações externas medem a capacidade de generalização em imagens fora do dataset principal.
- Resultados com poucos exemplos devem ser interpretados com cautela.