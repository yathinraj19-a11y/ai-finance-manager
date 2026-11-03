# AI-Powered Personal Finance Manager

Streamlit app that tracks income/expenses, auto-categorizes transactions with NLP, analyzes spending patterns,
forecasts next month's spend, flags unusual transactions and gives personalized savings advice.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Use the bundled sample data (default) or upload a CSV with columns `date, description, amount` (`category`, `type` optional).

## Files
- `app.py` – Streamlit UI (5 tabs)
- `ai_engine.py` – AI modules: TF-IDF + Logistic Regression categorizer, Isolation Forest anomaly detection,
  linear-regression forecast, rule-based recommendation engine
- `sample_data.py` – sample INR transactions generator + labelled training descriptions
