from ucimlrepo import fetch_ucirepo
import pandas as pd

heart = fetch_ucirepo(id=45)
X = heart.data.features
y = heart.data.targets

df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

# unify target to 0/1 if "num" field exists
if 'num' in df.columns and 'target' not in df.columns:
    df['target'] = (df['num'] > 0).astype(int)

df.to_csv("data/heart_disease.csv", index=False)
print("Saved data/heart_disease.csv with shape:", df.shape)
