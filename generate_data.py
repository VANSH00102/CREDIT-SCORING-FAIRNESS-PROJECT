import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

def rc(choices, probs, n=N):
    p = np.array(probs, dtype=float)
    p /= p.sum()
    return np.random.choice(choices, n, p=p)

checking    = rc(['A11','A12','A13','A14'], [0.27,0.27,0.06,0.39])
credit_hist = rc(['A30','A31','A32','A33','A34'], [0.04,0.05,0.53,0.09,0.29])
purpose     = rc(['A40','A41','A42','A43','A44','A45','A46','A48','A49','A410'],
                  [0.23,0.10,0.18,0.10,0.01,0.02,0.05,0.01,0.10,0.20])
savings     = rc(['A61','A62','A63','A64','A65'], [0.60,0.10,0.06,0.06,0.18])
employment  = rc(['A71','A72','A73','A74','A75'], [0.06,0.17,0.34,0.17,0.26])
personal    = rc(['A91','A92','A93','A94'], [0.05,0.31,0.55,0.09])
other_deb   = rc(['A101','A102','A103'], [0.91,0.04,0.05])
property_   = rc(['A121','A122','A123','A124'], [0.28,0.23,0.33,0.16])
inst_plans  = rc(['A141','A142','A143'], [0.14,0.05,0.81])
housing     = rc(['A151','A152','A153'], [0.18,0.71,0.11])
job         = rc(['A171','A172','A173','A174'], [0.02,0.20,0.63,0.15])
telephone   = rc(['A191','A192'], [0.60,0.40])
foreign     = rc(['A201','A202'], [0.96,0.04])

duration      = np.random.randint(4, 73, N)
credit_amount = np.random.randint(250, 18500, N)
inst_rate     = np.random.randint(1, 5, N)
residence     = np.random.randint(1, 5, N)
age           = np.clip(np.random.normal(35, 10, N).astype(int), 18, 75)
exist_cred    = rc([1,2,3,4], [0.63,0.22,0.11,0.04])
dependents    = rc([1,2], [0.85,0.15])

score = (
    (checking=='A11').astype(float)*1.5 +
    (checking=='A12').astype(float)*0.7 -
    (checking=='A14').astype(float)*1.1 +
    (credit_hist=='A34').astype(float)*1.8 +
    (credit_hist=='A33').astype(float)*0.9 -
    (savings=='A64').astype(float)*1.0 -
    (savings=='A65').astype(float)*0.3 +
    (employment=='A71').astype(float)*0.9 -
    (employment=='A75').astype(float)*0.6 +
    duration/72.0*2.0 +
    credit_amount/18500.0*1.5 +
    inst_rate/4.0*0.5 +
    np.random.normal(0, 1.1, N)
)
target = (score >= np.percentile(score, 70)).astype(int) + 1

df = pd.DataFrame({
    'checking_account': checking,    'duration': duration,
    'credit_history':   credit_hist, 'purpose':  purpose,
    'credit_amount':    credit_amount,'savings':  savings,
    'employment':       employment,   'installment_rate': inst_rate,
    'personal_status':  personal,     'other_debtors': other_deb,
    'residence_since':  residence,    'property': property_,
    'age':              age,          'installment_plans': inst_plans,
    'housing':          housing,      'existing_credits': exist_cred,
    'job':              job,          'dependents': dependents,
    'telephone':        telephone,    'foreign_worker': foreign,
    'credit_risk':      target
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/german.data', sep=' ', index=False, header=False)
print(f"Dataset created — {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Good credit: {(target==1).sum()}  Bad credit: {(target==2).sum()}")
print("german.data saved inside data/ folder successfully!")