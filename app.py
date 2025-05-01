import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import lime
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
            idx = np.random.choice(X_sel.shape[0], size=min(100, X_sel.shape[0]), replace=False)
            X_samp = X_sel[idx]
            if selected=="knn" or selected=="mlp":
                explainer=shap.KernelExplainer(clf.predict_proba,X_samp)
                vals=explainer.shap_values(X_samp)
            else :    
                vals = shap.TreeExplainer(clf).shap_values(X_samp)
            vals=vals[:,:,0]
            fig=plt.figure()
            shap.summary_plot(vals, X_samp, feature_names=X_train_df.columns[mask])
            # save and display
            fig.savefig(plot_path)
            img = Image.open(plot_path)
            st.image(img, caption=f"SHAP summary for {selected}")

# Generate live LIME plot
if st.button('Generate LIME plot'):
    with st.spinner('Generating LIME plot...'):
        # reload test data & transformer & mask
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
        # load model
        clf = joblib.load(os.path.join(models_dir, f"{selected}.pkl"))
        # build explainer and explain first instance
        feat_names = X_train_df.columns[mask]
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_sel,
            feature_names=feat_names,
            class_names=['Normal','Attack'],
            mode='classification'
        )
        random_idx = np.random.randint(0, X_test_sel.shape[0])
        exp = lime_explainer.explain_instance(
            X_test_sel[random_idx], 
            clf.predict_proba, 
            num_features=min(10, X_test_sel.shape[1])
        )

        # Save as HTML and render in Streamlit
        lime_html = exp.as_html()
        st.components.v1.html(lime_html, height=800)