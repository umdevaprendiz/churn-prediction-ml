# Previsão de Churn de Clientes

Projeto end-to-end de Machine Learning para prever se um cliente vai
cancelar o serviço (churn), num cenário de telecom: limpeza de dados,
engenharia de features, comparação de modelos e avaliação com métricas
apropriadas para classificação.

## O problema

Prever churn é um dos casos de uso mais comuns em análise de dados e IA
aplicada a negócio: identificar clientes com risco de cancelamento permite
que a empresa aja antes (desconto, contato, upgrade de plano) em vez de só
reagir depois que o cliente já saiu.

## Sobre os dados

O dataset (`data/churn.csv`) é **sintético**, gerado por
[`data/generate_data.py`](data/generate_data.py) com semente fixa
(reprodutível), mas construído para imitar as correlações de um cenário
real de churn de telecom (baseado no conhecido dataset público *Telco
Customer Churn*):

- Contratos mês a mês têm risco de churn muito maior que contratos de 2 anos.
- Falta de suporte técnico e muitas ligações de suporte aumentam o risco.
- Tempo de casa (`tenure_months`) alto reduz o risco.
- O dataset tem valores faltantes e duplicatas propositais, para exercitar
  a etapa de limpeza.

Isso mantém o projeto 100% reprodutível offline, sem depender de download
de um dataset externo. Trocar por um dataset real (ex: Kaggle Telco
Customer Churn) é possível sem mudar o pipeline — só ajustar as colunas em
[`src/data_prep.py`](src/data_prep.py).

## Estrutura do projeto

```
data/generate_data.py   # gera o dataset sintético (data/churn.csv)
src/data_prep.py         # limpeza e preparação (funções puras, testáveis)
src/train.py              # treino, comparação de modelos e avaliação
src/predict.py             # inferência em novos dados com o modelo salvo
src/export_for_bi.py        # gera data/churn_scored.csv (dados + previsão do modelo)
tests/test_data_prep.py    # testes unitários da limpeza/preparação
models/                     # modelo treinado (.joblib), gerado localmente
notebooks/eda.ipynb          # análise exploratória guiada (ver TASK_EDA.md)
docs/powerbi_dashboard.png    # print do dashboard de negócio (ver TASK_POWERBI.md)
```

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

python data/generate_data.py   # gera data/churn.csv
pytest -v                       # roda os testes
python -m src.train              # treina, compara modelos e salva o melhor
python -m src.predict data/churn.csv   # exemplo de predição em lote
```

## Abordagem

- **Pré-processamento**: `StandardScaler` nas features numéricas e
  `OneHotEncoder` nas categóricas, dentro de um `ColumnTransformer` (sem
  vazamento de dados entre treino/teste).
- **Modelos comparados**: Regressão Logística e Random Forest, ambos dentro
  de um `Pipeline` do scikit-learn.
- **Métricas**: accuracy, precision, recall, F1 e ROC-AUC — não só
  accuracy, já que o problema é moderadamente desbalanceado (~32% churn).
- **Seleção do modelo**: o modelo com maior ROC-AUC no conjunto de teste é
  salvo em `models/churn_model.joblib`.

## Resultados (dataset sintético atual)

| Modelo              | Accuracy | Precision | Recall | F1   | ROC-AUC |
|----------------------|----------|-----------|--------|------|---------|
| Regressão Logística  | 0.758    | 0.639     | 0.563  | 0.598| 0.804   |
| Random Forest         | 0.737    | 0.618     | 0.464  | 0.530| 0.781   |

A Regressão Logística venceu neste caso — sinal de que a relação entre as
features e o churn é majoritariamente linear no dataset gerado, o que faz
sentido dado como as probabilidades foram construídas em
`generate_data.py`. Com um dataset real, vale testar também Gradient
Boosting (XGBoost/LightGBM) para comparação.

## Análise Exploratória (EDA)

Um notebook guiado ([`notebooks/eda.ipynb`](notebooks/eda.ipynb), roteiro
completo em [`TASK_EDA.md`](TASK_EDA.md)) explora as distribuições,
correlações e a taxa de churn por segmento — em andamento.

## Dashboard no Power BI

Além da EDA em Python, os dados (com a previsão do modelo incluída, via
`src/export_for_bi.py` → `data/churn_scored.csv`) alimentam um dashboard
de negócio no Power BI, com KPIs, gráficos por segmento e uma tabela de
clientes de maior risco. Roteiro completo em
[`TASK_POWERBI.md`](TASK_POWERBI.md).

**Status: em andamento.** KPIs de Churn Real, Churn Previsto e Total de
Clientes já implementados; Receita em Risco e a tabela de clientes de
risco ainda pendentes.

![Dashboard Power BI - em andamento](docs/powerbi_dashboard.png)

## Próximos passos possíveis

- Terminar o KPI de Receita em Risco e a tabela de clientes de maior risco
  no dashboard do Power BI.
- Trocar o dataset sintético pelo Telco Customer Churn real do Kaggle.
- Adicionar tuning de hiperparâmetros (`GridSearchCV`/`RandomizedSearchCV`).
- Expor o modelo via uma API simples (FastAPI) para predição em tempo real.
