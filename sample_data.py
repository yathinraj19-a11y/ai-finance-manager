"""Generates realistic sample transactions (INR) for demo purposes."""
import random
from datetime import date, timedelta

import pandas as pd

# (description templates, category, amount range)
SPENDING = {
    "Food & Dining": (["Swiggy order", "Zomato order", "Domino's Pizza", "Cafe Coffee Day", "Starbucks", "Restaurant dinner", "Biryani House"], (150, 900)),
    "Groceries": (["BigBasket", "Blinkit groceries", "DMart", "Reliance Fresh", "Zepto vegetables", "Nature's Basket"], (250, 1800)),
    "Transport": (["Uber ride", "Ola cab", "Namma Metro recharge", "Petrol pump", "Rapido bike", "BMTC bus pass"], (40, 1200)),
    "Shopping": (["Amazon purchase", "Flipkart order", "Myntra clothing", "Decathlon sports", "Croma electronics"], (300, 3000)),
    "Entertainment": (["Netflix subscription", "Spotify Premium", "BookMyShow tickets", "PVR Cinemas", "Steam game", "Hotstar"], (100, 900)),
    "Utilities & Bills": (["BESCOM electricity bill", "Airtel broadband", "Jio recharge", "Water bill", "Gas cylinder"], (250, 2200)),
    "Health": (["Apollo Pharmacy", "Practo consultation", "Cult.fit membership", "Diagnostic lab test"], (150, 1500)),
    "Education": (["Udemy course", "Coursera subscription", "Textbook store", "Exam fee"], (300, 3500)),
    "Travel": (["IRCTC ticket", "MakeMyTrip flight", "OYO hotel booking", "IndiGo airlines"], (500, 4500)),
}

RENT = ("House rent", "Rent", 14000)


def generate(months: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    end = date.today()
    start = (end.replace(day=1) - timedelta(days=30 * (months - 1))).replace(day=1)
    rows = []
    d = start
    while d <= end:
        if d.day == 1:
            rows.append((d, "Monthly salary credit", "Salary", 52000.0, "Income"))
            rows.append((d + timedelta(days=1), RENT[0], RENT[1], float(RENT[2]), "Expense"))
        if d.day == 15 and rng.random() < 0.6:
            rows.append((d, "Freelance project payment", "Freelance", float(rng.randint(3000, 9000)), "Income"))
        # ~0.8 everyday expenses/day
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            cat = rng.choices(list(SPENDING), weights=[22, 14, 16, 9, 7, 6, 5, 3, 2])[0]
            names, (lo, hi) = SPENDING[cat]
            rows.append((d, rng.choice(names), cat, round(rng.uniform(lo, hi), 0), "Expense"))
        d += timedelta(days=1)
    # a few unusual spikes so anomaly detection has something to find
    spikes = [("Amazon purchase", "Shopping", 24999.0), ("MakeMyTrip flight", "Travel", 18450.0),
              ("Apollo Pharmacy", "Health", 9800.0)]
    for desc, cat, amt in spikes:
        dt = end - timedelta(days=rng.randint(5, 80))
        rows.append((dt, desc, cat, amt, "Expense"))
    df = pd.DataFrame(rows, columns=["date", "description", "category", "amount", "type"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def training_data() -> pd.DataFrame:
    """Labelled descriptions used to train the categorization model."""
    rng = random.Random(7)
    rows = []
    for cat, (names, _) in SPENDING.items():
        for n in names:
            for suffix in ["", " UPI", " POS 4821", " online", " #" + str(rng.randint(100, 999)), " payment"]:
                rows.append((n + suffix, cat))
    rows += [("House rent", "Rent"), ("Rent payment landlord", "Rent"), ("PG rent", "Rent"),
             ("Monthly salary credit", "Salary"), ("Salary", "Salary"),
             ("Freelance project payment", "Freelance"), ("Client invoice payment", "Freelance")]
    return pd.DataFrame(rows, columns=["description", "category"])
