# Tarefa - API de Previsão de Churn

Objetivo: expor o modelo já treinado através de uma API (FastAPI), pra
poder gerar previsões e relatórios sem precisar abrir o terminal e rodar
scripts na mão toda vez.

Você vai implementar os dois endpoints em `api/main.py`. Eles estão com
`raise NotImplementedError` no lugar do código. Os testes em
`tests/test_api.py` já estão prontos e **não devem ser alterados** - use-os
para saber se sua implementação está correta.

## Como saber se está indo bem

```bash
pip install -r requirements.txt   # agora inclui fastapi, uvicorn, httpx
pytest tests/test_api.py -v
```

No começo, os 4 testes vão falhar (erro 500, porque os endpoints levantam
`NotImplementedError`). Implemente um endpoint por vez.

---

## `POST /predict`

Recebe os dados de **um cliente** (já validados pelo `CustomerInput` em
`api/schemas.py` - isso já está pronto) e deve retornar um
`PredictionOutput` com `churn_prediction` (0 ou 1) e `churn_probability`
(0.0 a 1.0).

Passos:
1. Carregue o modelo salvo com `joblib.load(MODEL_PATH)`.
2. Transforme o `customer` (que chega como um objeto `CustomerInput`) em um
   DataFrame de **uma linha só**. Dica: `customer.model_dump()` retorna um
   dicionário com os campos; `pd.DataFrame([...])` transforma uma lista de
   dicionários em DataFrame.
3. Chame `model.predict(X)` e `model.predict_proba(X)` nesse DataFrame,
   igual você já viu em `src/predict.py` e `src/export_for_bi.py`.
4. Retorne `PredictionOutput(churn_prediction=..., churn_probability=...)`.

Cuidado com os tipos: `model.predict(X)[0]` retorna um `numpy.int64`, não um
`int` do Python puro - pode ser necessário envolver com `int(...)` e
`float(...)` antes de colocar no `PredictionOutput`, senão o FastAPI pode
reclamar na validação da resposta.

---

## `POST /generate-report`

Não recebe nada no corpo da requisição. Deve rodar o scoring de **todos**
os clientes (reaproveitando o que já existe) e salvar o CSV, retornando um
`ReportSummary` com `rows` (quantas linhas foram geradas) e `output_path`
(o caminho do arquivo salvo).

Passos:
1. Chame a função `score_customers()` (importada de
   `src.export_for_bi` - já está no topo do arquivo).
2. Salve o resultado em `OUTPUT_PATH` (também já importado) com
   `.to_csv(OUTPUT_PATH, index=False)`.
3. Retorne `ReportSummary(rows=len(df), output_path=OUTPUT_PATH)`.

Repare que isso é basicamente o mesmo que `src/export_for_bi.py` já faz no
terminal - a diferença é que agora dá pra disparar isso por uma chamada de
API, sem precisar de acesso ao terminal do servidor.

---

## Testando manualmente (depois dos testes passarem)

Suba o servidor:

```bash
uvicorn api.main:app --reload
```

Abra `http://127.0.0.1:8000/docs` no navegador - o FastAPI gera
automaticamente uma interface pra testar os endpoints direto do navegador,
sem precisar escrever nenhum código de cliente.
