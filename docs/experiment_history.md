# Histórico de Experimentos — AstroMind

Este documento é gerado automaticamente a partir dos logs acumulativos do projeto.

Arquivos de origem:

- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\train_runs.csv`
- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\evaluate_runs.csv`
- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\predict_runs.csv`
- `C:\Workspace\Python\AstroMind\AstroMind\experiment_logs\external_evaluate_runs.csv`

## 1. Histórico de treinos

Total de treinos registrados: **9**

| Run ID | Data | Modelo | Treino | Validação | Épocas | Melhor época | Best Val Loss | Best Val Acc | Final Train Acc | Final Val Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Run ID | Data | Treino | Validação | Épocas | Melhor época | Best Val Loss | Best Val Acc | Final Train Acc | Final Val Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_20260526_092834_a7911d86 | 2026-05-26 09:29:44 | AstroMindCNNV1 | 57 | 12 | 30 | 28 | 0.438775 | 91.67% | 71.67% | 75.00% |
| train_20260526_100239_364d4d4b | 2026-05-26 10:04:00 | AstroMindCNNV1 | 57 | 12 | 30 | 28 | 0.438775 | 91.67% | 71.67% | 75.00% |
| train_20260526_102743_194ad6ed | 2026-05-26 10:33:14 | AstroMindCNNV2 | 57 | 12 | 30 | 20 | 0.654141 | 75.00% | 58.33% | 83.33% |
| train_20260526_150026_e7870f64 | 2026-05-26 15:12:32 | AstroMindCNNV2.1 | 57 | 12 | 50 | 43 | 0.492136 | 75.00% | 81.67% | 75.00% |
| train_20260527_085121_448e3459 | 2026-05-27 09:12:57 | AstroMindCNNV2.2 | 118 | 25 | 50 | 41 | 0.526084 | 78.57% | 65.00% | 78.57% |
| train_20260527_155905_0612b48a | 2026-05-27 16:08:40 | AstroMindCNNV2.3 | 118 | 25 | 50 | 11 | 0.729064 | 64.29% | 71.67% | 67.86% |
| train_20260527_163858_a7c772fa | 2026-05-27 17:01:20 | AstroMindCNNV2.4 | 118 | 25 | 50 | 43 | 0.521015 | 85.71% | 72.50% | 82.14% |
| train_20260528_110538_c209d69b | 2026-05-28 11:26:46 | AstroMindCNNV2.4 | 118 | 25 | 50 | 43 | 0.521015 | 85.71% | 72.50% | 82.14% |
| train_20260528_114121_f3cdc353 | 2026-05-28 12:02:39 | AstroMindCNNV2.5 | 139 | 30 | 50 | 34 | 0.442773 | 71.88% | 68.33% | 75.00% |

**Melhor treino registrado:**

- Run ID: `train_20260526_092834_a7911d86`
- Data: 2026-05-26 09:29:44
- Modelo: AstroMindCNNV1
- Melhor época: 28
- Melhor validation accuracy: 91.67%
- Melhor validation loss: 0.438775

## 2. Histórico de avaliações no conjunto de teste

Total de avaliações registradas: **9**

| Evaluate Run ID | Train Run ID | Data | Dataset | Acurácia | Acertos | Erros |
|---|---|---:|---:|---:|---:|---:|
| evaluate_20260526_093313_ffed3e97 | train_20260526_092834_a7911d86 | 2026-05-26 09:33:13 | 13 | 92.31% | 12 | 1 |
| evaluate_20260526_100417_eb9470af | train_20260526_100239_364d4d4b | 2026-05-26 10:04:17 | 13 | 92.31% | 12 | 1 |
| evaluate_20260526_115218_049a5553 | train_20260526_102743_194ad6ed | 2026-05-26 11:52:19 | 13 | 84.62% | 11 | 2 |
| evaluate_20260526_153219_467bae83 | train_20260526_150026_e7870f64 | 2026-05-26 15:32:20 | 13 | 100.00% | 13 | 0 |
| evaluate_20260527_113723_2d14081f | train_20260527_085121_448e3459 | 2026-05-27 11:37:25 | 26 | 76.92% | 20 | 6 |
| evaluate_20260527_161729_af0382e7 | train_20260527_155905_0612b48a | 2026-05-27 16:17:31 | 26 | 65.38% | 17 | 9 |
| evaluate_20260527_170141_e51050bb | train_20260527_163858_a7c772fa | 2026-05-27 17:01:44 | 26 | 80.77% | 21 | 5 |
| evaluate_20260528_112656_109d9700 | train_20260528_110538_c209d69b | 2026-05-28 11:26:58 | 26 | 80.77% | 21 | 5 |
| evaluate_20260528_120247_b5f58b9c | train_20260528_114121_f3cdc353 | 2026-05-28 12:02:49 | 30 | 66.67% | 20 | 10 |

**Melhor avaliação registrada:**

- Evaluate Run ID: `evaluate_20260526_153219_467bae83`
- Train Run ID: `train_20260526_150026_e7870f64`
- Acurácia: 100.00%
- Acertos: 13
- Erros: 0

## 3. Histórico de avaliações externas

Total de avaliações externas registradas: **9**

| External Run ID | Train Run ID | Data | Imagens | Imagens rotuladas | Acurácia externa | Acertos | Erros |
|---|---|---:|---:|---:|---:|---:|---:|
| external_evaluate_20260526_093829_9ae41a04 | train_20260526_092834_a7911d86 | 2026-05-26 09:38:30 | 21 | 21 | 80.95% | 17 | 4 |
| external_evaluate_20260526_100618_2c58c09f | train_20260526_100239_364d4d4b | 2026-05-26 10:06:18 | 21 | 21 | 80.95% | 17 | 4 |
| external_evaluate_20260526_115646_755f6152 | train_20260526_102743_194ad6ed | 2026-05-26 11:56:48 | 21 | 21 | 85.71% | 18 | 3 |
| external_evaluate_20260526_153748_205429ec | train_20260526_150026_e7870f64 | 2026-05-26 15:37:51 | 21 | 21 | 80.95% | 17 | 4 |
| external_evaluate_20260527_113926_b34f29e9 | train_20260527_085121_448e3459 | 2026-05-27 11:39:28 | 21 | 21 | 76.19% | 16 | 5 |
| external_evaluate_20260527_161803_002871a4 | train_20260527_155905_0612b48a | 2026-05-27 16:18:04 | 21 | 21 | 76.19% | 16 | 5 |
| external_evaluate_20260527_170153_55f1ce39 | train_20260527_163858_a7c772fa | 2026-05-27 17:01:55 | 21 | 21 | 90.48% | 19 | 2 |
| external_evaluate_20260528_112708_76b64f44 | train_20260528_110538_c209d69b | 2026-05-28 11:27:10 | 21 | 21 | 90.48% | 19 | 2 |
| external_evaluate_20260528_120257_d259f673 | train_20260528_114121_f3cdc353 | 2026-05-28 12:02:58 | 21 | 21 | 80.95% | 17 | 4 |

## 4. Histórico de predições individuais

Total de predições registradas: **9**

Últimas 10 predições:

| Predict Run ID | Data | Classe real | Previsto | Confiança | Acertou |
|---|---:|---:|---:|---:|---:|
| predict_20260526_093440_0237fa47 | 2026-05-26 09:34:41 | aglomerado | aglomerado | 92.87% | True |
| predict_20260526_100510_956a2cc3 | 2026-05-26 10:05:10 | aglomerado | aglomerado | 92.87% | True |
| predict_20260526_115607_874d9dc6 | 2026-05-26 11:56:08 | aglomerado | galaxia | 56.62% | False |
| predict_20260526_153425_ca65e711 | 2026-05-26 15:34:25 | aglomerado | aglomerado | 94.04% | True |
| predict_20260527_115310_63a04a7e | 2026-05-27 11:53:11 | aglomerado | aglomerado | 92.72% | True |
| predict_20260527_161819_a6a1e463 | 2026-05-27 16:18:19 | aglomerado | aglomerado | 61.62% | True |
| predict_20260527_170205_28629f86 | 2026-05-27 17:02:05 | aglomerado | aglomerado | 85.88% | True |
| predict_20260528_112840_c80cfeac | 2026-05-28 11:28:40 | aglomerado | aglomerado | 85.88% | True |
| predict_20260528_120307_8a6f74e7 | 2026-05-28 12:03:07 | aglomerado | aglomerado | 79.94% | True |

## 5. Observações

- As métricas de validação acompanham o desempenho durante o treino.
- As métricas de teste medem o desempenho em imagens separadas do treino.
- As avaliações externas medem a capacidade de generalização em imagens fora do dataset principal.
- Resultados com poucos exemplos devem ser interpretados com cautela.