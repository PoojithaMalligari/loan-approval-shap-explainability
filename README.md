# Loan Approval – SHAP Explainability Add-on

Adds **per-prediction explanations** ("top 3 reasons") to a loan approval model, so a rejected applicant can see *why*.

## Files
- `explain.py` – `LoanExplainer` class (SHAP TreeExplainer, top-N reasons)
- `demo_api.py` – runnable FastAPI demo on synthetic data
- `ReasonsList.jsx` – React component to display reasons

## Run the demo
```
pip install -r requirements.txt
uvicorn demo_api:app --reload
```
Open http://127.0.0.1:8000/docs → POST /predict.

## Add to my real project
1. Copy `explain.py` into the backend.
2. Load model once: `explainer = LoanExplainer(tree_model)`.
3. In the predict endpoint return `"top_reasons": explainer.top_reasons(input_df)`.
4. Show with `ReasonsList.jsx`.

## Results (fill in with MY real numbers)
| Metric | Value |
|---|---|
| Accuracy | _ |
| ROC-AUC | _ |

## Screenshot
_(add screenshot of the reasons in the UI)_
