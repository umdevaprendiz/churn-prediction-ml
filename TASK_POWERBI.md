# Tarefa - Dashboard de Churn no Power BI

Objetivo: construir um dashboard de negócio em cima dos dados de churn,
agora já incluindo a previsão do modelo (quem tem mais risco de cancelar).
Isso complementa a EDA em Python (que é mais exploratória/técnica) com uma
visão executiva, interativa, do tipo que um gestor não-técnico consegue
usar.

## Arquivo de dados

Use **`data/churn_scored.csv`** (não o `churn.csv` original). Ele já tem
todas as colunas de cliente, mais duas geradas pelo modelo:

- `churn_prediction`: 0 ou 1 — o que o modelo previu.
- `churn_probability`: um número de 0 a 1 — a "confiança" do modelo de que
  aquele cliente vai cancelar (quanto mais perto de 1, maior o risco).

Se você mudar `data/churn.csv` ou retreinar o modelo, rode de novo:

```bash
python -m src.export_for_bi
```

para atualizar o `churn_scored.csv` antes de atualizar o Power BI.

---

## Parte 1 - Conectar os dados

1. Abra o Power BI Desktop → **Obter Dados** → **Texto/CSV**.
2. Selecione `data/churn_scored.csv`.
3. No preview, confira se os tipos de coluna fazem sentido (`tenure_months`
   como número inteiro, `churn_probability` como número decimal, etc.) antes
   de clicar em **Carregar**.

---

## Parte 2 - KPIs (cartões no topo do dashboard)

Crie 4 **medidas** (Nova Medida, linguagem DAX) e coloque cada uma num
cartão (visual "Cartão"):

- **Total de Clientes**: `Total Clientes = COUNTROWS(churn_scored)`
- **Taxa de Churn Real**: `% Churn Real = AVERAGE(churn_scored[churn])`
  (como churn é 0/1, a média já é a proporção — formate como porcentagem).
- **Taxa de Churn Prevista**: `% Churn Previsto = AVERAGE(churn_scored[churn_prediction])`
  Compare visualmente com a taxa real: se estiverem muito diferentes, é
  sinal de que o modelo está enviesado pra um lado.
- **Receita Mensal em Risco**:
  `Receita em Risco = CALCULATE(SUM(churn_scored[monthly_charges]), churn_scored[churn_prediction] = 1)`
  Isso traduz o problema pra linguagem de negócio: quanto de receita mensal
  vem de clientes que o modelo marcou como risco de cancelar.

---

## Parte 3 - Gráficos por segmento

Um gráfico de barras para cada uma (some ou faça a média de
`churn_probability` agrupado por categoria):

- `contract_type` no eixo, `% Churn Real` (ou a média de `churn_probability`)
  como valor.
- `tech_support` no eixo, mesma métrica.
- `payment_method` no eixo, mesma métrica.

Pergunta pra responder olhando os gráficos: o padrão que você viu na EDA em
Python (contrato mensal = mais risco) se confirma aqui também?

---

## Parte 4 - Tabela de clientes de risco

Um visual de **Tabela**, filtrado (use um filtro de visual ou slicer) para
mostrar só `churn_prediction = 1`, ordenado por `churn_probability`
decrescente. Colunas sugeridas: `tenure_months`, `monthly_charges`,
`contract_type`, `num_support_calls`, `churn_probability`.

Essa tabela responde a pergunta mais prática do dashboard: **"quais
clientes específicos a equipe de retenção deveria contatar primeiro?"**

---

## Parte 5 - Filtros (slicers)

Adicione slicers (filtros interativos) para pelo menos:
- `contract_type`
- `tech_support`

Isso permite que quem for usar o dashboard explore por conta própria (ex:
"e se eu olhar só os clientes sem suporte técnico?").

---

## Quando terminar

Tire um screenshot (ou exporte um PDF) do dashboard completo e salve em
`docs/powerbi_dashboard.png` (crie a pasta `docs/` se não existir) — isso
vai pro README do projeto, já que o `.pbix` (arquivo do Power BI) não
costuma ser versionado no GitHub por ser binário e pesado; a imagem é o que
representa o trabalho no portfólio.
