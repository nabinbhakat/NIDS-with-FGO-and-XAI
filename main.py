import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Paths setup
def get_paths():
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, 'data')
    models_dir = os.path.join(base, 'models')
    plots_dir = os.path.join(base, 'plots')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return data_dir, models_dir, plots_dir

data_dir, models_dir, plots_dir = get_paths()

# Load & clean
df_train = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_training.csv')).dropna()
df_test  = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_testing.csv')).dropna()

# Features & labels
X_train_df = df_train.drop(columns=['attack_cat', 'label'])
y_train     = df_train['label'].values
X_test_df  = df_test.drop(columns=['attack_cat', 'label'])
y_test      = df_test['label'].values

# ColumnTransformer persistence
col_path = os.path.join(models_dir, 'col_transformer.pkl')
if os.path.exists(col_path):
    col_trans = joblib.load(col_path)
else:
    num_cols = X_train_df.select_dtypes(include=['int64','float64']).columns
    cat_cols = X_train_df.select_dtypes(include=['object','bool']).columns
    col_trans = ColumnTransformer([
        ('ohe', OneHotEncoder(drop='first', sparse=False), cat_cols),
        ('scale', StandardScaler(), num_cols)
    ])
    col_trans.fit(X_train_df)
    joblib.dump(col_trans, col_path)

# Transform data
X_train = col_trans.transform(X_train_df)
X_test  = col_trans.transform(X_test_df)

# FGO routines
def fitness(sol, X, y, clf, alpha=0.01):
    sel = sol > 0.5
    if not sel.any(): return 1.0
    X_sel = X[:, sel]
    scores = cross_val_score(clf, X_sel, y, cv=5, scoring='accuracy', error_score='raise')
    return (1 - np.mean(scores)) + alpha * (sel.sum()/len(sol))

def run_fgo(X, y, clf, pop_size=20, max_iter=10, exp_w=0.9, explo_w=0.1):
    n_feat = X.shape[1]
    pop = np.random.rand(pop_size, n_feat)
    best_sol, best_fit = None, float('inf')
    for t in range(max_iter):
        fits = np.array([fitness(p, X, y, clf) for p in pop])
        idx = fits.argmin()
        if fits[idx] < best_fit:
            best_fit = fits[idx]; best_sol = pop[idx].copy()
        E = exp_w * (1 - t/max_iter) + explo_w
        for i in range(pop_size):
            r = np.random.rand(n_feat)
            pop[i] += E * r * (best_sol - pop[i]) + 0.01 * np.random.randn(n_feat)
            pop[i] = np.clip(pop[i], 0, 1)
        print(f"FGO iter {t+1}/{max_iter} best={best_fit:.4f} features={int((best_sol>0.5).sum())}")
    return best_sol > 0.5

# Feature mask load/run
mask_file = os.path.join(models_dir, 'selected_features.npy')
if os.path.exists(mask_file):
    feat_mask = np.load(mask_file)
else:
    feat_mask = run_fgo(X_train, y_train, DecisionTreeClassifier(random_state=42))
    np.save(mask_file, feat_mask)

# Apply mask
X_train_sel = X_train[:, feat_mask]
X_test_sel  = X_test[:, feat_mask]

# Model training/evaluation
metrics_file = os.path.join(models_dir, 'metrics.pkl')
if os.path.exists(metrics_file):
    evaluations = joblib.load(metrics_file)
else:
    models = {
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'knn':           KNeighborsClassifier(),
        'mlp':           MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
        'xgboost':       xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    evaluations = {}
    for name, mdl in models.items():
        mdl.fit(X_train_sel, y_train)
        joblib.dump(mdl, os.path.join(models_dir, f"{name}.pkl"))
        y_pred = mdl.predict(X_test_sel)
        evaluations[name] = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall':    recall_score(y_test, y_pred, average='weighted'),
            'f1_score':  f1_score(y_test, y_pred, average='weighted')
        }
    joblib.dump(evaluations, metrics_file)

# Report
for nm, m in evaluations.items():
    print(f"{nm}: " + ", ".join([f"{k}={v:.4f}" for k,v in m.items()]))

# Generate or load SHAP plots
for name in evaluations:
    plot_path = os.path.join(plots_dir, f"{name}_shap.png")
    if not os.path.exists(plot_path):
        clf = joblib.load(os.path.join(models_dir, f"{name}.pkl"))
        idx = np.random.choice(X_test_sel.shape[0], size=100, replace=False)
        X_samp = X_test_sel[idx]
        if name=="knn" or name=="mlp":
            explainer=shap.KernelExplainer(clf.predict_proba,X_samp)
            vals=explainer.shap_values(X_samp)
        else :    
            vals = shap.TreeExplainer(clf).shap_values(X_samp)
        vals=vals[:,:,0]
        fig=plt.figure()
        shap.summary_plot(vals, X_samp, feature_names=X_train_df.columns[feat_mask])
        fig.savefig(plot_path)
        print(f"Saved SHAP plot for {name}")
