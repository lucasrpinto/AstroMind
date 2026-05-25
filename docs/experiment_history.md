# Histórico de Experimentos — AstroMind

Este documento é gerado automaticamente a partir dos logs acumulativos do projeto.

Arquivos de origem:

- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\train_runs.csv`
- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\evaluate_runs.csv`
- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\predict_runs.csv`
- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\external_evaluate_runs.csv`

## 1. Histórico de treinos

Total de treinos registrados: **1**

| Run ID | Data | Treino | Validação | Épocas | Melhor época | Best Val Loss | Best Val Acc | Final Train Acc | Final Val Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_20260525_112437_a054efed | 2026-05-25 11:25:47 | 57 | 12 | 30 | 28 | 0.438775 | 91.67% | 71.67% | 75.00% |

**Melhor treino registrado:**

- Run ID: `train_20260525_112437_a054efed`
- Data: 2026-05-25 11:25:47
- Melhor época: 28
- Melhor validation accuracy: 91.67%
- Melhor validation loss: 0.438775

## 2. Histórico de avaliações no conjunto de teste

Total de avaliações registradas: **1**

| Evaluate Run ID | Train Run ID | Data | Dataset | Acurácia | Acertos | Erros |
|---|---|---:|---:|---:|---:|---:|
| evaluate_20260525_112556_9abd951d | train_20260525_112437_a054efed | 2026-05-25 11:25:56 | 13 | 92.31% | 12 | 1 |

**Melhor avaliação registrada:**

- Evaluate Run ID: `evaluate_20260525_112556_9abd951d`
- Train Run ID: `train_20260525_112437_a054efed`
- Acurácia: 92.31%
- Acertos: 12
- Erros: 1

## 3. Histórico de avaliações externas

Total de avaliações externas registradas: **3**

| External Run ID | Train Run ID | Data | Imagens | Imagens rotuladas | Acurácia externa | Acertos | Erros |
|---|---|---:|---:|---:|---:|---:|---:|
| external_evaluate_20260525_114904_bcc9e54c | train_20260525_112437_a054efed | 2026-05-25 11:49:04 | 1 | 1 | 100.00% | 1 | 0 |
| external_evaluate_20260525_135947_6202e17f | train_20260525_112437_a054efed | 2026-05-25 13:59:47 | 4 | 4 | 25.00% | 1 | 3 |
| external_evaluate_20260525_141845_c80e308f | train_20260525_112437_a054efed | 2026-05-25 14:18:47 | 21 | 21 | 80.95% | 17 | 4 |

## 4. Histórico de predições individuais

Total de predições registradas: **1**

Últimas 10 predições:

| Predict Run ID | Data | Classe real | Previsto | Confiança | Acertou |
|---|---:|---:|---:|---:|---:|
| predict_20260525_112604_e798f795 | 2026-05-25 11:26:04 | aglomerado | aglomerado | 92.87% | True |

## 5. Observações

- As métricas de validação acompanham o desempenho durante o treino.
- As métricas de teste medem o desempenho em imagens separadas do treino.
- As avaliações externas medem a capacidade de generalização em imagens fora do dataset principal.
- Resultados com poucos exemplos devem ser interpretados com cautela.