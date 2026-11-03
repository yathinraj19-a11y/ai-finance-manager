"""AI-Powered Personal Finance Manager  -  Streamlit app.
Run:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from ai_engine import (TransactionCategorizer, detect_anomalies,
                       forecast_next_month, generate_recommendations)
from sample_data import generate

st.set_page_config(page_title="AI Personal Finance Manager", page_icon="💰", layout="wide")
INCOME_CATS = {"Salary", "Freelance"}


@st.cache_resource
def get_categorizer():
    return TransactionCategorizer()


@st.cache_data
def sample_df():
    return generate()


def load_uploaded(file) -> pd.DataFrame:
    raw = pd.read_csv(file)
    raw.columns = [c.strip().lower() for c in raw.columns]
    missing = {"date", "description", "amount"} - set(raw.columns)
    if missing:
        st.error(f"CSV is missing columns: {', '.join(sorted(missing))}")
        st.stop()
    raw["date"] = pd.to_datetime(raw["date"])
    return raw


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Use AI to categorize any rows without a category, then derive income/expense type."""
    df = raw.copy()
    cat = get_categorizer()
    if "category" not in df.columns:
        df["category"] = None
    need = df["category"].isna() | (df["category"].astype(str).str.strip() == "")
    df["confidence"] = 1.0
    if need.any():
        pred = cat.predict(df.loc[need, "description"])
        df.loc[need, "category"] = pred["category"].values
        df.loc[need, "confidence"] = pred["confidence"].values
    if "type" not in df.columns:
        df["type"] = df["category"].apply(lambda c: "Income" if c in INCOME_CATS else "Expense")
    df["amount"] = df["amount"].abs()
    return df.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("💰 Finance AI")
source = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"])
if source == "Upload CSV":
    st.sidebar.caption("Columns: date, description, amount (category & type optional)")
    f = st.sidebar.file_uploader("Transactions CSV", type="csv")
    base = load_uploaded(f) if f else None
    if base is None:
        st.info("Upload a CSV, or switch to **Sample data** in the sidebar.")
        st.stop()
else:
    base = sample_df()

if "added" not in st.session_state:
    st.session_state.added = []

with st.sidebar.expander("➕ Add a transaction (AI categorizes it)"):
    with st.form("add", clear_on_submit=True):
        d = st.date_input("Date")
        desc = st.text_input("Description", placeholder="e.g. Swiggy order")
        amt = st.number_input("Amount (₹)", min_value=0.0, step=50.0)
        if st.form_submit_button("Add") and desc and amt > 0:
            st.session_state.added.append({"date": pd.Timestamp(d), "description": desc, "amount": amt})
            st.toast("Transaction added")

goal = st.sidebar.slider("Savings goal (% of income)", 5, 50, 20)

raw = base.copy()
if st.session_state.added:
    raw = pd.concat([raw, pd.DataFrame(st.session_state.added)], ignore_index=True)
df = prepare(raw)
exp = df[df["type"] == "Expense"].copy()
inc = df[df["type"] == "Income"]

months = sorted(df["date"].dt.to_period("M").astype(str).unique())
sel = st.sidebar.multiselect("Months", months, default=months)
view = df[df["date"].dt.to_period("M").astype(str).isin(sel)]
vexp = view[view["type"] == "Expense"]
vinc = view[view["type"] == "Income"]

# ------------------------------------------------------------------ header + KPIs
st.title("AI-Powered Personal Finance Manager")
st.caption("Track income & expenses, auto-categorize with NLP, spot unusual spends, forecast, and get savings advice.")

total_in, total_out = vinc["amount"].sum(), vexp["amount"].sum()
net = total_in - total_out
rate = net / total_in * 100 if total_in else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Income", f"₹{total_in:,.0f}")
c2.metric("Expenses", f"₹{total_out:,.0f}")
c3.metric("Net savings", f"₹{net:,.0f}")
c4.metric("Savings rate", f"{rate:.1f}%", delta=f"{rate-goal:+.1f} vs goal")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🏷️ AI categorization", "🔍 Spending patterns", "🚨 Forecast & anomalies", "💡 Recommendations"])

# ------------------------------------------------------------------ Overview
with tab1:
    l, r = st.columns([3, 2])
    monthly = (view.assign(month=view["date"].dt.to_period("M").astype(str))
               .groupby(["month", "type"])["amount"].sum().reset_index())
    l.plotly_chart(px.bar(monthly, x="month", y="amount", color="type", barmode="group",
                          title="Income vs expenses by month",
                          color_discrete_map={"Income": "#2a9d8f", "Expense": "#e76f51"}),
                   width="stretch")
    catsum = vexp.groupby("category")["amount"].sum().reset_index()
    r.plotly_chart(px.pie(catsum, names="category", values="amount", hole=0.45,
                          title="Expense share by category"), width="stretch")
    st.subheader("Recent transactions")
    st.dataframe(view.sort_values("date", ascending=False).head(15)
                 [["date", "description", "category", "amount", "type"]],
                 width="stretch", hide_index=True)

# ------------------------------------------------------------------ Categorization
with tab2:
    st.subheader("NLP transaction categorization")
    st.write("A TF-IDF character n-gram model with logistic regression reads each description and predicts its category, "
             "with a confidence score. Try your own text:")
    txt = st.text_input("Try a description", "Uber ride to airport")
    if txt:
        p = get_categorizer().predict([txt]).iloc[0]
        st.success(f"Predicted category: **{p['category']}**  (confidence {p['confidence']:.0%})")
    ai_rows = view[view["confidence"] < 1.0]
    st.caption(f"{len(ai_rows)} of {len(view)} transactions were categorized by AI.")
    low = ai_rows[ai_rows["confidence"] < 0.6]
    if len(low):
        st.warning(f"{len(low)} low-confidence predictions — worth a manual check:")
        st.dataframe(low[["date", "description", "category", "confidence"]], width="stretch", hide_index=True)
    st.dataframe(ai_rows.sort_values("date", ascending=False).head(25)
                 [["date", "description", "category", "confidence"]]
                 .style.format({"confidence": "{:.0%}"}), width="stretch", hide_index=True)

# ------------------------------------------------------------------ Patterns
with tab3:
    st.subheader("Spending pattern analysis")
    a, b = st.columns(2)
    heat = vexp.assign(month=vexp["date"].dt.to_period("M").astype(str)).pivot_table(
        index="category", columns="month", values="amount", aggfunc="sum", fill_value=0)
    a.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Oranges",
                             title="Category × month heatmap"), width="stretch")
    dow = vexp.assign(day=vexp["date"].dt.day_name()).groupby("day")["amount"].sum().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index()
    b.plotly_chart(px.bar(dow, x="day", y="amount", title="Spending by day of week",
                          color_discrete_sequence=["#264653"]), width="stretch")
    daily = vexp.groupby("date")["amount"].sum().reset_index()
    daily["7-day avg"] = daily["amount"].rolling(7, min_periods=1).mean()
    st.plotly_chart(px.line(daily, x="date", y="7-day avg", title="Daily spend (7-day moving average)"),
                    width="stretch")
    top = vexp.groupby("category")["amount"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
    top.columns = ["Total ₹", "Avg per txn ₹", "Transactions"]
    st.dataframe(top.style.format("{:,.0f}"), width="stretch")

# ------------------------------------------------------------------ Forecast + anomalies
fc = forecast_next_month(exp)
an = detect_anomalies(exp)
with tab4:
    st.subheader("Spending forecast")
    hist = fc["history"].copy()
    hist["month"] = hist["date"].dt.strftime("%Y-%m")
    st.metric("Forecast for next month", f"₹{fc['next_month']:,.0f}",
              delta=f"₹{fc['slope']:+,.0f}/month trend", delta_color="inverse")
    st.plotly_chart(px.line(hist, x="month", y="amount", markers=True, title="Monthly spend history (current month may be partial)"),
                    width="stretch")
    st.subheader("Anomaly detection (Isolation Forest)")
    flagged = an[an["anomaly"]].sort_values("amount", ascending=False)
    if flagged.empty:
        st.success("No unusual transactions found.")
    else:
        st.warning(f"{len(flagged)} transaction(s) look unusually large for their category.")
        st.dataframe(flagged[["date", "description", "category", "amount"]], width="stretch", hide_index=True)

# ------------------------------------------------------------------ Recommendations
with tab5:
    st.subheader("Personalized recommendations")
    for rec in generate_recommendations(df, fc, an, goal):
        box = {"warning": st.warning, "success": st.success, "info": st.info}[rec["level"]]
        box(f"**{rec['title']}**\n\n{rec['text']}")
    st.divider()
    st.subheader("What-if: reduce a category")
    cats = sorted(vexp["category"].unique())
    if cats:
        pick = st.selectbox("Category", cats)
        cut = st.slider("Reduce spending by (%)", 0, 50, 15)
        n_m = max(len(sel), 1)
        monthly_cat = vexp.loc[vexp["category"] == pick, "amount"].sum() / n_m
        saved = monthly_cat * cut / 100
        st.write(f"Cutting **{pick}** by {cut}% saves about **₹{saved:,.0f}/month** (₹{saved*12:,.0f}/year).")
