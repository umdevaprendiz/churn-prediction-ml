# Tarefa - Análise Exploratória (EDA) do Churn

Objetivo: praticar EDA de verdade em cima do `data/churn.csv` - a mesma
etapa que qualquer analista de dados/IA faz antes de treinar qualquer
modelo. O notebook `notebooks/eda.ipynb` já tem a estrutura (imports,
carregamento dos dados, e uma célula de código por pergunta) - você
implementa o código de cada célula.

Não existe teste automatizado aqui (EDA é exploratório por natureza), mas
cada seção abaixo diz o que o gráfico/número deve mostrar, pra você
conferir se o resultado faz sentido.

---

## Parte 1 - Visão geral

### 1.1 Estrutura dos dados
Mostre `shape`, `dtypes` e `describe()` do DataFrame. Confira também
`isna().sum()` por coluna (o dataset tem nulos propositais em
`total_charges`).

### 1.2 Taxa geral de churn
Calcule a proporção de clientes com `churn == 1` vs `churn == 0`
(`value_counts(normalize=True)`). Isso te diz o quão desbalanceado é o
problema - importante pra escolher métricas depois.

---

## Parte 2 - Variáveis numéricas

Colunas: `tenure_months`, `monthly_charges`, `total_charges`,
`num_support_calls`.

### 2.1 Distribuições
Um histograma (`sns.histplot` ou `df[col].hist()`) para cada variável
numérica. Veja se alguma é enviesada (skewed) ou tem outliers óbvios.

### 2.2 Numéricas vs churn
Para cada variável numérica, um boxplot separado por `churn`
(`sns.boxplot(x="churn", y=coluna, data=df)`). A pergunta que isso responde:
"clientes que cancelam têm valores tipicamente diferentes nessa variável?"
Esperado: `tenure_months` deve ser visivelmente menor em quem cancelou.

### 2.3 Matriz de correlação
Um heatmap (`sns.heatmap`) da correlação entre as variáveis numéricas
(inclua `churn` na matriz, mesmo sendo 0/1). Veja quais variáveis numéricas
têm maior correlação (positiva ou negativa) com o churn.

---

## Parte 3 - Variáveis categóricas

Colunas: `contract_type`, `internet_service`, `tech_support`,
`payment_method`, `has_dependents`, `senior_citizen`.

### 3.1 Taxa de churn por categoria
Para cada variável categórica, calcule a taxa de churn por grupo:

```python
df.groupby("contract_type")["churn"].mean()
```

e visualize com um gráfico de barras (`sns.barplot`). Repita para as outras
colunas categóricas.

### 3.2 Qual variável categórica mais separa os grupos?
Depois de ver os gráficos da 3.1, escreva (em uma célula markdown) qual
variável categórica parece ter o maior impacto na taxa de churn, e por quê
isso faz sentido no contexto de negócio (ex: "clientes em contrato mensal
podem sair a qualquer momento, sem multa").

---

## Parte 4 - Cruzamentos e conclusão

### 4.1 Scatter plot
Um `sns.scatterplot` de `tenure_months` (eixo x) por `monthly_charges`
(eixo y), colorindo os pontos por `churn` (`hue="churn"`). Veja se existe
uma região do gráfico (ex: contrato caro + pouco tempo de casa) mais
concentrada em churn.

### 4.2 Resumo de negócio
Escreva, em uma célula markdown no final do notebook, de 3 a 5 conclusões
em linguagem de negócio (não técnica) baseadas no que você encontrou.
Exemplo de formato:

> Clientes com contrato mês a mês e sem suporte técnico têm risco de churn
> muito acima da média. Ações de retenção deveriam priorizar esse grupo.

Essas conclusões podem (e devem) ser copiadas depois para o `README.md`
principal do projeto, numa seção "Principais insights da EDA".
