import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import lime
from lime import lime_tabular
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from PIL import Image
import matplotlib.pyplot as plt

# Page config and title
st.set_page_config(page_title='Network Intrusion Detection System', layout='centered')
st.title('Network Intrusion Detection System using Explainable AI')

# Paths
base_dir = os.path.dirname(__file__)
models_dir = os.path.join(base_dir, 'models')
plots_dir = os.path.join(base_dir, 'plots')

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Load stored metrics
metrics = joblib.load(os.path.join(models_dir, 'metrics.pkl'))
model_names = list(metrics.keys())

# Model selection dropdown
selected = st.selectbox('Select Model', model_names)

# Display performance metrics
m = metrics[selected]
st.subheader('Performance Metrics')
st.write(f"Accuracy: {m['accuracy']:.4f}")
st.write(f"Precision: {m['precision']:.4f}")
st.write(f"Recall: {m['recall']:.4f}")
st.write(f"F1-score: {m['f1_score']:.4f}")

# Generate or load SHAP plot
if st.button('Generate SHAP plot'):
    plot_path = os.path.join(plots_dir, f"{selected}_shap.png")
    if os.path.exists(plot_path):
        img = Image.open(plot_path)
        st.image(img, caption=f"SHAP summary for {selected}")
    else:
        with st.spinner('Generating SHAP plot...'):
            # re-load test data & transformer & mask
            data_dir = os.path.join(base_dir, 'data')
            df_test = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_testing.csv')).dropna()
            df_train = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_training.csv')).dropna()
            X_train_df = df_train.drop(columns=['attack_cat', 'label'])
            X_df = df_test.drop(columns=['attack_cat','label'])
            col_trans = joblib.load(os.path.join(models_dir,'col_transformer.pkl'))
            X = col_trans.transform(X_df)
            mask = np.load(os.path.join(models_dir,'selected_features.npy'))
            X_sel = X[:, mask]
            # load model
            clf = joblib.load(os.path.join(models_dir, f"{selected}.pkl"))
            
            # Improved error handling for SHAP plot generation
            try:
                idx = np.random.choice(X_sel.shape[0], size=min(100, X_sel.shape[0]), replace=False)
                X_samp = X_sel[idx]
                
                if selected=="knn" or selected=="mlp":
                    explainer=shap.KernelExplainer(clf.predict_proba,X_samp)
                    vals=explainer.shap_values(X_samp)
                else:    
                    explainer = shap.TreeExplainer(clf)
                    vals = explainer.shap_values(X_samp)
                
                # Properly handle different SHAP value formats
                if isinstance(vals, list):
                    # For multi-class output, focus on class 1 (Attack)
                    vals = vals[1]
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(vals, X_samp, feature_names=X_train_df.columns[mask], show=False)
                
                # save and display
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                img = Image.open(plot_path)
                st.image(img, caption=f"SHAP summary for {selected}")
            except Exception as e:
                st.error(f"Error generating SHAP plot: {e}")
          
# Generate live LIME plot
if st.button('Generate LIME plot'):
    with st.spinner('Generating LIME plot...'):
        try:
            # Reload data and preprocessing components
            data_dir = os.path.join(base_dir, 'data')
            df_test = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_testing.csv')).dropna()
            df_train = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_training.csv')).dropna()
            X_train_df = df_train.drop(columns=['attack_cat','label'])
            X_test_df = df_test.drop(columns=['attack_cat','label'])
            col_trans = joblib.load(os.path.join(models_dir,'col_transformer.pkl'))
            X_train_trans = col_trans.transform(X_train_df)
            X_test_trans = col_trans.transform(X_test_df)
            mask = np.load(os.path.join(models_dir,'selected_features.npy'))
            X_test_sel = X_test_trans[:, mask]
            X_train_sel = X_train_trans[:,mask]

            # Modified model loading with compatibility hacks
            clf = joblib.load(os.path.join(models_dir, f"{selected}.pkl"))

            def remove_monotonic_constraint(model):
                if hasattr(model, 'monotonic_cst'):
                    del model.monotonic_cst
                # Recursively patch ensemble models
                if hasattr(model, 'estimators_'):
                    for estimator in model.estimators_:
                        remove_monotonic_constraint(estimator)
                return model
            
            if any(x in selected.lower() for x in ['tree', 'forest', 'ensemble']):
                clf = remove_monotonic_constraint(clf)
            
            # Create prediction wrapper
            def predict_fn(X):
                X = np.nan_to_num(X, nan=0.0)
                return clf.predict_proba(X)

            # Build LIME explainer
            feat_names = X_train_df.columns[mask]
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_sel,
                feature_names=feat_names.tolist(),
                class_names=['Normal','Attack'],
                mode='classification',
                discretize_continuous=True,
                verbose=False
            )

            # Generate explanation
            random_idx = np.random.randint(0, X_test_sel.shape[0])
            exp = lime_explainer.explain_instance(
                data_row=X_test_sel[random_idx],
                predict_fn=predict_fn,
                num_features=min(10, X_test_sel.shape[1]),
                top_labels=1
            )

            # Visualization
            try:
                lime_html = exp.as_html()
                st.components.v1.html(lime_html, height=800)
            except Exception as e:
                st.warning(f"HTML rendering failed: {e}. Using fallback visualization.")
                fig, ax = plt.subplots(figsize=(10, 6))
                exp.as_pyplot_figure(label=1)
                st.pyplot(fig)
                st.dataframe(pd.DataFrame(exp.as_list(label=1), columns=["Feature", "Impact"]))

        except Exception as e:
            st.error(f"Error generating LIME explanation: {e}")