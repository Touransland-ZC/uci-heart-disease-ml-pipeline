# UCI Heart Disease — Full ML Pipeline 🫀

End-to-end machine learning pipeline on the **UCI Heart Disease (Cleveland)** dataset.  
Covers data access → EDA → PCA → feature selection → supervised learning (baseline & tuning) → unsupervised clustering → export & simple UI.

**Best model:** Tuned Logistic Regression (feature-selected matrix)  
**Test AUROC:** ~**0.968** · **Recall:** ~0.93 · **Precision:** ~0.81 · **F1:** ~0.87

---

##  Setup

### Option A — Windows (venv)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
### Option B — Linux/Mac
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
##  Project Structure (key items)
```bash
data/
  heart_disease.csv
  heart_disease_clean.csv
  heart_selected.csv
deployment/
  ngrok_setup.txt
models/
  final_model.pkl               # full pipeline (preprocess + model)
results/
  final_tuned_metrics.csv
  tuning_comparison_baseline_vs_tuned.csv
  evaluation_metrics.txt
  best_thresholds.json
  unsupervised_cluster_assignments.csv
notebooks/
  01_data_preprocessing.ipynb
  02_pca_analysis.ipynb
  03_feature_selection.ipynb
  04_supervised_learning.ipynb
  05_unsupervised_learning.ipynb
  06_hyperparameter_tuning.ipynb
ui/
  app.py                        # Streamlit demo
```

