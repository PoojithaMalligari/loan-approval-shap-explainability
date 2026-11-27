"""Runnable DEMO of the SHAP feature on synthetic data.
Run:  uvicorn demo_api:app --reload     then open http://127.0.0.1:8000/docs
Copy the pieces marked  # >>> ADD THIS  into YOUR real loan project.
"""
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from explain import LoanExplainer   # >>> ADD THIS (import)

# ---- synthetic training data (replace with your real model: joblib.load("models/model.pkl")) ----
rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "income": rng.normal(50000, 15000, n),
    "loan_amount": rng.normal(200000, 80000, n),
    "credit_score": rng.integers(300, 850, n),
    "years_employed": rng.integers(0, 20, n),
})
score = (df.income / 20000) - (df.loan_amount / 150000) + (df.credit_score - 550) / 150 + df.years_employed / 15
y = (score + rng.normal(0, 0.5, n) > 1.2).astype(int)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(df, y)

explainer = LoanExplainer(model)       # >>> ADD THIS (create once at startup)
app = FastAPI(title="Loan Approval + SHAP demo")


class Applicant(BaseModel):
    income: float
    loan_amount: float
    credit_score: int
    years_employed: int


@app.post("/predict")
def predict(a: Applicant):
    X = pd.DataFrame([a.model_dump()])
    prob = float(model.predict_proba(X)[0][1])
    return {
        "approved": prob >= 0.5,
        "probability": round(prob, 3),
        "top_reasons": explainer.top_reasons(X),   # >>> ADD THIS (return reasons)
    }
