"""Create the course dataset: data/churn.csv

A made-up (synthetic) table of telecom customers. Each row is one customer.
The column "churned" is 1 if the customer left the company, 0 if they stayed.
The same seed always produces the same file, so everyone gets identical data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_CUSTOMERS = 2000
SEED = 42


def make_churn_data(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)                       # months as a customer
    monthly = rng.uniform(20, 120, n).round(2)            # monthly bill
    calls = rng.poisson(1.5, n)                           # calls to support
    contract = rng.choice(["month-to-month", "one-year", "two-year"],
                          n, p=[0.55, 0.25, 0.20])

    # Short tenure, high bills, many support calls and monthly contracts
    # make a customer more likely to leave.
    score = (-0.04 * tenure + 0.02 * monthly + 0.45 * calls
             + np.select([contract == "month-to-month", contract == "one-year"],
                         [1.0, -0.3], -1.2)
             - 1.2)
    churned = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)

    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "support_calls": calls,
        "contract": contract,
        "churned": churned,
    })


if __name__ == "__main__":
    out = Path("data") / "churn.csv"
    out.parent.mkdir(exist_ok=True)
    df = make_churn_data()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out} (churn rate {df['churned'].mean():.0%})")
