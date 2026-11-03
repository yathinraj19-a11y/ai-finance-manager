"""AI engine for the Personal Finance Manager.

Four AI components:
  1. TransactionCategorizer  - NLP (TF-IDF + Logistic Regression) auto-categorizes descriptions
  2. detect_anomalies        - Isolation Forest flags unusual spends
  3. forecast_next_month     - Linear regression trend forecast of spending
  4. generate_recommendations- Rule-based advisor driven by the model outputs (50/30/20 benchmark)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

from sample_data import training_data


# ---------- 1. NLP categorization ----------
class TransactionCategorizer:
    def __init__(self):
        df = training_data()
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True),
            LogisticRegression(max_iter=1000, C=10),
        )
        self.model.fit(df["description"], df["category"])

    def predict(self, descriptions) -> pd.DataFrame:
        descriptions = list(descriptions)
        proba = self.model.predict_proba(descriptions)
        idx = proba.argmax(axis=1)
        return pd.DataFrame({
            "category": self.model.classes_[idx],
            "confidence": proba.max(axis=1),
        })


# ---------- 2. Anomaly detection ----------
def detect_anomalies(expenses: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Flag expenses whose amount is unusual for their category (Isolation Forest)."""
    df = expenses.copy()
    df["anomaly"] = False
    if len(df) < 20:
        return df
    grp = df.groupby("category")["amount"]
    df["z"] = (df["amount"] - grp.transform("mean")) / grp.transform("std").replace(0, 1)
    X = np.column_stack([np.log1p(df["amount"]), df["z"].fillna(0)])
    iso = IsolationForest(contamination=contamination, random_state=0)
    df["anomaly"] = iso.fit_predict(X) == -1
    # only keep genuine overspends, not unusually small values
    df["anomaly"] &= df["z"] > 1.5
    return df.drop(columns="z")


# ---------- 3. Forecasting ----------
def forecast_next_month(expenses: pd.DataFrame) -> dict:
    m = (expenses.set_index("date").resample("MS")["amount"].sum().reset_index())
    # drop the (likely incomplete) current month for training if we have enough history
    full = m.iloc[:-1] if len(m) > 3 else m
    X = np.arange(len(full)).reshape(-1, 1)
    model = LinearRegression().fit(X, full["amount"])
    nxt = float(model.predict([[len(full)]])[0])
    return {"history": m, "next_month": max(nxt, 0.0), "slope": float(model.coef_[0])}


# ---------- 4. Recommendations ----------
def generate_recommendations(df: pd.DataFrame, forecast: dict, anomalies: pd.DataFrame, goal_pct: float = 20) -> list[dict]:
    recs = []
    income = df.loc[df["type"] == "Income", "amount"].sum()
    exp = df[df["type"] == "Expense"]
    spend = exp["amount"].sum()
    months = max(df["date"].dt.to_period("M").nunique(), 1)
    m_income, m_spend = income / months, spend / months
    savings_rate = (m_income - m_spend) / m_income * 100 if m_income else 0

    if savings_rate < goal_pct:
        gap = (goal_pct / 100 * m_income) - (m_income - m_spend)
        recs.append({"level": "warning", "title": f"Savings rate is {savings_rate:.0f}% (goal {goal_pct:.0f}%)",
                     "text": f"Cut about ₹{gap:,.0f} per month to reach your goal."})
    else:
        recs.append({"level": "success", "title": f"Great savings rate: {savings_rate:.0f}%",
                     "text": "Consider investing the surplus in an SIP or emergency fund."})

    by_cat = exp.groupby("category")["amount"].sum().sort_values(ascending=False) / months
    flexible = by_cat.drop(labels=["Rent", "Utilities & Bills"], errors="ignore")
    if len(flexible):
        top = flexible.index[0]
        recs.append({"level": "info", "title": f"Biggest flexible spend: {top}",
                     "text": f"You spend ~₹{flexible.iloc[0]:,.0f}/month here. A 15% trim saves ~₹{flexible.iloc[0]*0.15:,.0f}/month."})

    if "Rent" in by_cat and m_income and by_cat["Rent"] / m_income > 0.30:
        recs.append({"level": "warning", "title": "Rent above 30% of income",
                     "text": "Housing is taking a large share; consider this when planning other goals."})

    food = by_cat.get("Food & Dining", 0)
    if food and m_income and food / m_income > 0.12:
        recs.append({"level": "info", "title": "High eating-out spend",
                     "text": f"Food & Dining is {food/m_income*100:.0f}% of income. Cooking a few more meals at home could save ₹{food*0.2:,.0f}/month."})

    if forecast["slope"] > 500:
        recs.append({"level": "warning", "title": "Spending trend is rising",
                     "text": f"Spending is growing ~₹{forecast['slope']:,.0f} each month. Forecast next month: ₹{forecast['next_month']:,.0f}."})

    n_an = int(anomalies["anomaly"].sum())
    if n_an:
        recs.append({"level": "warning", "title": f"{n_an} unusual transaction(s) detected",
                     "text": "Review them in the Anomalies tab to confirm they are expected."})

    recs.append({"level": "info", "title": "Emergency fund target",
                 "text": f"Aim for 6 months of expenses: about ₹{m_spend*6:,.0f}."})
    return recs
