# Relatório de melhorias — gráficos e análises

Resumo do que foi alterado no projeto nesta rodada, por quê, e como reproduzir.

## 1. Notebook de EDA (`notebooks/eda.ipynb`) — refeito por completo

O notebook original tinha só as Partes 1 e 2.1/2.2 (o resto era `TODO`) e usava gráficos padrão. Agora:

| Seção | O que tem |
|-------|-----------|
| 1. Visão geral | Estrutura, relatório de nulos/duplicatas com gráfico, barra empilhada da taxa geral de churn |
| 2. Numéricas | Histogramas **já separados por churn** (com assimetria), boxplots com medianas rotuladas, heatmap de correlação triangular |
| 3. Categóricas | 6 gráficos de taxa de churn com `n` por grupo e linha da média; ranking de "poder de separação" |
| 4. Cruzamentos | Scatter + heatmap tempo de casa × mensalidade, heatmap contrato × suporte, churn por ligações ao suporte e por faixa de tempo de casa |
| 5. Receita em risco (novo) | Mensalidade em clientes que cancelam, por tipo de contrato |
| 6. Resumo | 5 insights de negócio, ações sugeridas e aviso de que os dados são sintéticos |

**Correções de análise:**
- A versão antiga fazia `dropna(subset=["total_charges"])`, descartando 90 linhas que só tinham esse campo faltando. Agora usa `clean_data()` (a mesma limpeza do pipeline de treino: remove duplicatas e imputa a mediana), então a EDA analisa os mesmos dados que o modelo vê.
- Gráficos de barra passaram a mostrar o tamanho de cada grupo (`n`), para não confundir ruído com sinal.

## 2. Avaliação do modelo (novo: `src/evaluate.py`)

`python -m src.evaluate` gera em `docs/figures/` (e as métricas em `docs/metrics.json`):

`model_comparison.png` · `roc_pr_curves.png` · `confusion_matrix.png` · `threshold_analysis.png` ·
`feature_importance.png` · `calibration_lift.png` · `score_distribution.png`

**Achados principais (conjunto de teste, 600 clientes):**
- Melhor modelo: Regressão Logística, ROC-AUC 0,804 (CV 5 folds: 0,822 ± 0,013).
- Importância por permutação: tipo de contrato (0,187) e tempo de casa (0,065) dominam.
- **Limiar otimizado: 0,32** (escolhido por validação cruzada só no treino, sem usar o teste). Recall sobe de 56% para 74%; precisão cai de 64% para 54%; F1 de 0,60 para 0,62.
- Os 3 decis de maior risco capturam **59%** dos churns reais; taxa de churn real vai de 3% (decil 1) a 73% (decil 10).

## 3. Modelagem (`src/train.py`)

- **Novo candidato**: Gradient Boosting (`HistGradientBoostingClassifier`). Não superou a Regressão Logística (ROC-AUC 0,780 vs 0,804), o que reforça que o sinal do dataset sintético é majoritariamente linear.
- **Validação cruzada** (5 folds) para ROC-AUC de cada modelo, reportada junto das métricas do teste.
- **Bug corrigido**: os pipelines compartilhavam a *mesma instância* de `ColumnTransformer`, de modo que o ajuste de um sobrescrevia o do outro. Agora cada candidato recebe um clone.
- `split_data()` extraída para ser compartilhada entre `train` e `evaluate` (mesma divisão, mesmos números).

## 4. Estilo visual (novo: `src/viz.py`)

Paleta única e consistente em notebook e figuras: **azul = retido, laranja = churn**, rampa sequencial azul para magnitude e divergente azul↔vermelho para correlação. Tema com grade discreta, sem molduras, títulos alinhados à esquerda e rótulos diretos nas barras.

## 5. Testes

- Novo `tests/test_train.py` (4 testes): 3 candidatos, pré-processadores independentes, divisão estratificada e todos os modelos produzindo probabilidades válidas.
- Suíte completa: **11 testes passando**.

## 6. Documentação

- `README.md`: tabela de resultados com os 3 modelos e CV, seções com as figuras novas, "Principais insights da EDA", comandos e estrutura atualizados, status do projeto.
- `RELATORIO_MELHORIAS.md`: este arquivo.

## Arquivos

**Novos:** `src/evaluate.py`, `src/viz.py`, `tests/test_train.py`, `docs/metrics.json`, `docs/figures/*.png` (7), `RELATORIO_MELHORIAS.md`
**Alterados:** `notebooks/eda.ipynb`, `src/train.py`, `src/data_prep.py` (constante `DATA_PATH_DEFAULT`), `README.md`

## Limitações e próximos passos

- Os dados são **sintéticos**: os padrões refletem as regras de `data/generate_data.py`. Trocar pelo Telco Customer Churn real (Kaggle) é o passo mais valioso.
- O limiar otimizado (0,32) está só documentado/avaliado; a API e `churn_scored.csv` ainda usam o padrão 0,50.
- O dashboard do Power BI (`TASK_POWERBI.md`) continua pendente e não foi alterado.
- Tuning de hiperparâmetros e Docker seguem como próximos passos.

## Como reproduzir

```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
jupyter notebook notebooks/eda.ipynb
pytest -v
```
