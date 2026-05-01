# generate_churn_data.py
import numpy as np
import pandas as pd

def generate_churn_like(n=2000, seed=42):
    np.random.seed(seed)
    age = np.random.randint(18, 80, size=n)
    tenure = np.random.randint(0, 72, size=n)
    monthly_charges = np.round(np.random.normal(70, 30, size=n).clip(10, 300),2)
    contract = np.random.choice(["month-to-month","one-year","two-year"], size=n, p=[0.6,0.25,0.15])
    internet = np.random.choice(["dsl","fiber","none"], size=n, p=[0.35,0.45,0.2])
    senior = (age>=60).astype(int)
    p = 0.18 + 0.18*(contract=="month-to-month").astype(float) + 0.12*(internet=="fiber").astype(float)
    p += 0.06*(tenure<6).astype(float) - 0.03*(monthly_charges<50).astype(float)
    p = np.clip(p, 0.01, 0.95)
    churn = np.random.binomial(1, p)
    df = pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n)],
        "age": age,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "contract": contract,
        "internet_service": internet,
        "senior_citizen": senior,
        "churn": churn
    })
    return df

if __name__ == "__main__":
    df = generate_churn_like(2000)
    df.to_csv("sample_churn.csv", index=False)
    print("sample_churn.csv generated (2000 rows).")
