import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="centered")
st.title("Heart Disease – Risk Estimator")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    # final model pipeline (preprocessing + classifier)
    model = joblib.load("models/final_model.pkl")

    # meta is optional (versions, features, etc.)
    meta = {}
    if os.path.exists("models/final_model_meta.json"):
        try:
            meta = json.load(open("models/final_model_meta.json", "r"))
        except Exception:
            pass

    # template row to get the exact training schema (selected dataset)
    try:
        template = pd.read_csv("data/heart_selected.csv", nrows=1).drop(columns=["target"])
    except FileNotFoundError:
        st.error("Missing data/heart_selected.csv — run Notebook 03 to generate it.")
        st.stop()

    cols = list(template.columns)
    base = pd.Series(0.0, index=cols)   # start with all zeros (works for one-hots)
    return model, meta, template, base, cols

pipe, meta, template, base_row, cols = load_artifacts()

# ---------- UI inputs ----------
st.subheader("Enter patient data")

c1, c2 = st.columns(2)
with c1:
    age      = st.number_input("Age", min_value=20, max_value=90, value=55)
    trestbps = st.number_input("Resting BP (trestbps)", min_value=80, max_value=220, value=130)
    chol     = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=240)
with c2:
    thalach  = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=220, value=150)
    oldpeak  = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.5, value=1.0, step=0.1)

sex   = st.selectbox("Sex", ["female", "male"])                             
cp    = st.selectbox("Chest pain type (cp)", ["1.0","2.0","3.0","4.0"])     
exang = st.selectbox("Exercise-induced angina (exang)", ["0.0","1.0"])      
slope = st.selectbox("Slope of ST segment (slope)", ["0.0","1.0","2.0"])
thal  = st.selectbox("Thalassemia (thal)", ["3.0","6.0","7.0"])
ca    = st.selectbox("Major vessels colored (ca)", ["0.0","1.0","2.0","3.0"])

# ---------- Build a feature vector that matches training columns ----------
row = base_row.copy()

# numeric features
for k, v in {"age": age, "trestbps": trestbps, "chol": chol,
             "thalach": thalach, "oldpeak": oldpeak}.items():
    if k in row.index:
        row[k] = float(v)

def set_onehot(prefix: str, level: str):
    fam = [c for c in row.index if c.startswith(prefix + "_")]
    for c in fam:
        row[c] = 0.0
    col = f"{prefix}_{level}"
    if col in row.index:
        row[col] = 1.0

set_onehot("sex",   "1.0" if sex == "male" else "0.0")
set_onehot("cp",    cp)
set_onehot("exang", exang)
set_onehot("slope", slope)
set_onehot("thal",  thal)
set_onehot("ca",    ca)

X_one = pd.DataFrame([row.values], columns=row.index)

# ---------- Predict ----------
proba = float(pipe.predict_proba(X_one)[0, 1])

threshold = 0.5
try:
    with open("results/best_thresholds.json", "r") as f:
        thr_map = json.load(f)
        # prefer key for tuned logreg; otherwise just take the first value
        if "logreg_tuned" in thr_map:
            threshold = float(thr_map["logreg_tuned"])
        elif len(thr_map):
            threshold = float(list(thr_map.values())[0])
except Exception:
    pass

pred = int(proba >= threshold)

st.markdown(f"### Predicted probability of disease: **{proba:.3f}**")
st.markdown(f"Decision @ threshold {threshold:.2f}: **{pred}**  (1 = disease)")

try:
    from sklearn.linear_model import LogisticRegression
    clf = pipe.named_steps.get("clf", None)
    pre = pipe.named_steps.get("pre", None)
    if isinstance(clf, LogisticRegression) and pre is not None:
        # transform once with the preprocessor to align with the model's coefficient space
        Xz = pre.transform(X_one)  # numpy array
        w  = clf.coef_.ravel()
        contrib = pd.Series((w * Xz.ravel()), index=cols).sort_values(key=np.abs, ascending=False)
        st.subheader("Top feature contributions (signed, scaled space)")
        st.dataframe(contrib.head(10).to_frame("contribution"))
except Exception:
    pass

with st.expander("Model metadata"):
    st.json(meta or {"note": "No meta file found (optional)."})
st.caption("Note: This app expects the feature-selected schema. If a one-hot column was not selected during feature selection, it will simply be absent (handled automatically).")
