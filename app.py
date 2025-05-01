# import os
# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import shap
# import lime
# from lime import lime_tabular
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from PIL import Image
# import matplotlib.pyplot as plt

# # Page config and title
# st.set_page_config(page_title='Network Intrusion Detection System', layout='centered')
# st.title('Network Intrusion Detection System using Explainable AI')

# # Paths
# base_dir = os.path.dirname(__file__)
# models_dir = os.path.join(base_dir, 'models')
# plots_dir = os.path.join(base_dir, 'plots')

# # Create directories if they don't exist
# os.makedirs(models_dir, exist_ok=True)
# os.makedirs(plots_dir, exist_ok=True)

# # Load stored metrics
# metrics = joblib.load(os.path.join(models_dir, 'metrics.pkl'))
# model_names = list(metrics.keys())

# # Model selection dropdown
# selected = st.selectbox('Select Model', model_names)

# # Display performance metrics
# m = metrics[selected]
# st.subheader('Performance Metrics')
# st.write(f"Accuracy: {m['accuracy']:.4f}")
# st.write(f"Precision: {m['precision']:.4f}")
# st.write(f"Recall: {m['recall']:.4f}")
# st.write(f"F1-score: {m['f1_score']:.4f}")

# # Generate or load SHAP plot
# if st.button('Generate SHAP plot'):
#     plot_path = os.path.join(plots_dir, f"{selected}_shap.png")
#     if os.path.exists(plot_path):
#         img = Image.open(plot_path)
#         st.image(img, caption=f"SHAP summary for {selected}")
#     else:
#         with st.spinner('Generating SHAP plot...'):
#             # re-load test data & transformer & mask
#             data_dir = os.path.join(base_dir, 'data')
#             df_test = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_testing.csv')).dropna()
#             df_train = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_training.csv')).dropna()
#             X_train_df = df_train.drop(columns=['attack_cat', 'label'])
#             X_df = df_test.drop(columns=['attack_cat','label'])
#             col_trans = joblib.load(os.path.join(models_dir,'col_transformer.pkl'))
#             X = col_trans.transform(X_df)
#             mask = np.load(os.path.join(models_dir,'selected_features.npy'))
#             X_sel = X[:, mask]
#             # load model
#             clf = joblib.load(os.path.join(models_dir, f"{selected}.pkl"))
            
#             # Improved error handling for SHAP plot generation
#             try:
#                 idx = np.random.choice(X_sel.shape[0], size=min(100, X_sel.shape[0]), replace=False)
#                 X_samp = X_sel[idx]
                
#                 if selected=="knn" or selected=="mlp":
#                     explainer=shap.KernelExplainer(clf.predict_proba,X_samp)
#                     vals=explainer.shap_values(X_samp)
#                 else:    
#                     explainer = shap.TreeExplainer(clf)
#                     vals = explainer.shap_values(X_samp)
                
#                 # Properly handle different SHAP value formats
#                 if isinstance(vals, list):
#                     # For multi-class output, focus on class 1 (Attack)
#                     vals = vals[1]
                
#                 plt.figure(figsize=(10, 6))
#                 shap.summary_plot(vals, X_samp, feature_names=X_train_df.columns[mask], show=False)
                
#                 # save and display
#                 plt.savefig(plot_path, bbox_inches='tight')
#                 plt.close()
#                 img = Image.open(plot_path)
#                 st.image(img, caption=f"SHAP summary for {selected}")
#             except Exception as e:
#                 st.error(f"Error generating SHAP plot: {e}")

# # Generate live LIME plot
# if st.button('Generate LIME plot'):
#     with st.spinner('Generating LIME plot...'):
#         try:
#             # reload test data & transformer & mask
#             data_dir = os.path.join(base_dir, 'data')
#             df_test = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_testing.csv')).dropna()
#             df_train = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_training.csv')).dropna()
#             X_train_df = df_train.drop(columns=['attack_cat','label'])
#             X_test_df = df_test.drop(columns=['attack_cat','label'])
#             col_trans = joblib.load(os.path.join(models_dir,'col_transformer.pkl'))
#             X_train_trans = col_trans.transform(X_train_df)
#             X_test_trans = col_trans.transform(X_test_df)
#             mask = np.load(os.path.join(models_dir,'selected_features.npy'))
#             X_test_sel = X_test_trans[:, mask]
#             X_train_sel = X_train_trans[:,mask] 
            
#             # load model
#             clf = joblib.load(os.path.join(models_dir, f"{selected}.pkl"))
            
#             # FIX: Create a wrapper for the model predict function to handle missing values
#             def predict_fn(X):
#                 # Replace any NaN values with 0
#                 X = np.nan_to_num(X, nan=0.0)
#                 return clf.predict_proba(X)
            
#             # build explainer and explain first instance
#             feat_names = X_train_df.columns[mask]
#             lime_explainer = lime_tabular.LimeTabularExplainer(
#                 training_data=X_train_sel,
#                 feature_names=feat_names,
#                 class_names=['Normal','Attack'],
#                 mode='classification',
#                 discretize_continuous=True  # FIX: Added to handle continuous features better
#             )
            
#             random_idx = np.random.randint(0, X_test_sel.shape[0])
            
#             # FIX: Use our wrapped predict function instead of directly using clf.predict_proba
#             exp = lime_explainer.explain_instance(
#                 X_test_sel[random_idx], 
#                 predict_fn,  # FIX: Using wrapped function that handles NaN values
#                 num_features=min(10, X_test_sel.shape[1])
#             )
            
#             # FIX: Try different visualization options based on what works
#             try:
#                 # First try: using HTML component
#                 lime_html = exp.as_html()
#                 st.components.v1.html(lime_html, height=800)
#             except Exception as e:
#                 # Fallback: use matplotlib figure
#                 st.warning(f"HTML rendering failed: {e}. Using matplotlib visualization instead.")
#                 fig = exp.as_pyplot_figure(label=1)  # For the "Attack" class
#                 st.pyplot(fig)
                
#                 # Also show as table for clarity
#                 st.subheader("Feature Importance")
#                 feature_importance = exp.as_list(label=1)
#                 fi_df = pd.DataFrame(feature_importance, columns=["Feature", "Weight"])
#                 st.dataframe(fi_df)
        
#         except Exception as e:
#             st.error(f"Error generating LIME explanation: {e}")
            
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
data_dir = os.path.join(base_dir, 'data')

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Check if files exist before trying to load them
def safe_load_data():
    try:
        # Load stored metrics
        metrics = joblib.load(os.path.join(models_dir, 'metrics.pkl'))
        return metrics
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        st.stop()

metrics = safe_load_data()
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
            try:
                # Load necessary data
                col_trans = joblib.load(os.path.join(models_dir, 'col_transformer.pkl'))
                mask = np.load(os.path.join(models_dir, 'selected_features.npy'))
                clf = joblib.load(os.path.join(models_dir, f"{selected}.pkl"))
                
                # Load cached data or compute features
                feature_names_file = os.path.join(models_dir, 'feature_names.joblib')
                if os.path.exists(feature_names_file):
                    feature_names = joblib.load(feature_names_file)
                else:
                    # If we can't load the feature names from file, use generic names
                    feature_names = [f"feature_{i}" for i in range(sum(mask))]
                
                # Use cached preprocessed samples or create new ones
                samples_file = os.path.join(models_dir, 'sample_data.npy')
                if os.path.exists(samples_file):
                    X_samp = np.load(samples_file)
                else:
                    # Load test data if available
                    try:
                        df_test = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_testing.csv')).dropna()
                        X_df = df_test.drop(columns=['attack_cat', 'label'])
                        X = col_trans.transform(X_df)
                        X_sel = X[:, mask]
                        idx = np.random.choice(X_sel.shape[0], size=min(100, X_sel.shape[0]), replace=False)
                        X_samp = X_sel[idx]
                        np.save(samples_file, X_samp)
                    except Exception as e:
                        st.error(f"Error processing test data: {e}")
                        # Create random data as fallback
                        X_samp = np.random.rand(100, sum(mask))
                
                # Generate SHAP values based on model type
                if selected == "knn" or selected == "mlp":
                    explainer = shap.KernelExplainer(clf.predict_proba, X_samp)
                    shap_values = explainer.shap_values(X_samp)
                else:
                    explainer = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_samp)
                
                # Process SHAP values based on their format
                if isinstance(shap_values, list):
                    # For multi-class output, focus on class 1 (Attack)
                    shap_values = shap_values[1]
                
                # Create and save plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_samp, feature_names=feature_names, show=False)
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                
                # Display the saved image
                img = Image.open(plot_path)
                st.image(img, caption=f"SHAP summary for {selected}")
                
            except Exception as e:
                st.error(f"Error generating SHAP plot: {e}")

# Generate live LIME plot
if st.button('Generate LIME plot'):
    with st.spinner('Generating LIME plot...'):
        try:
            # Load necessary data
            col_trans = joblib.load(os.path.join(models_dir, 'col_transformer.pkl'))
            mask = np.load(os.path.join(models_dir, 'selected_features.npy'))
            clf = joblib.load(os.path.join(models_dir, f"{selected}.pkl"))
            
            # Load or create sample data for explanation
            samples_file = os.path.join(models_dir, 'sample_data.npy')
            if os.path.exists(samples_file):
                X_train_sel = np.load(samples_file)
                X_test_sel = X_train_sel  # Use same data for test
            else:
                try:
                    # Try to load from raw data files
                    df_test = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_testing.csv')).dropna()
                    df_train = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_training.csv')).dropna()
                    
                    # Process training data
                    X_train_df = df_train.drop(columns=['attack_cat', 'label'])
                    X_train_trans = col_trans.transform(X_train_df)
                    X_train_sel = X_train_trans[:, mask]
                    
                    # Process test data
                    X_test_df = df_test.drop(columns=['attack_cat', 'label'])
                    X_test_trans = col_trans.transform(X_test_df)
                    X_test_sel = X_test_trans[:, mask]
                    
                    # Save sample data for future use
                    np.save(samples_file, X_train_sel)
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    # Create random data as fallback
                    n_features = sum(mask) if isinstance(mask, np.ndarray) else 10
                    X_train_sel = np.random.rand(100, n_features)
                    X_test_sel = np.random.rand(10, n_features)
            
            # Load or generate feature names
            feature_names_file = os.path.join(models_dir, 'feature_names.joblib')
            if os.path.exists(feature_names_file):
                feat_names = joblib.load(feature_names_file)
            else:
                # Generate generic feature names
                feat_names = [f"feature_{i}" for i in range(X_train_sel.shape[1])]
            
            # Critical fix: Create a wrapper for the model predict function to handle missing values
            def predict_fn(X):
                # Replace any NaN values with 0
                X_clean = np.nan_to_num(X, nan=0.0)
                return clf.predict_proba(X_clean)
            
            # Build LIME explainer with safeguards
            lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_train_sel,
                feature_names=feat_names,
                class_names=['Normal', 'Attack'],
                mode='classification',
                discretize_continuous=True
            )
            
            # Choose a random sample to explain
            random_idx = np.random.randint(0, len(X_test_sel))
            
            # Generate explanation using our wrapped predict function
            exp = lime_explainer.explain_instance(
                X_test_sel[random_idx],
                predict_fn,  # Use wrapped function that handles NaN values
                num_features=min(10, X_test_sel.shape[1])
            )
            
            # Multiple visualization options based on what works in deployment
            try:
                # Try HTML rendering first
                lime_html = exp.as_html()
                st.components.v1.html(lime_html, height=800)
            except Exception as e:
                # Fall back to matplotlib visualization
                st.warning(f"HTML rendering failed: {e}. Using matplotlib visualization instead.")
                fig = exp.as_pyplot_figure(label=1)  # For the "Attack" class
                st.pyplot(fig)
                
                # Also show as table for clarity
                st.subheader("Feature Importance")
                feature_importance = exp.as_list(label=1)
                fi_df = pd.DataFrame(feature_importance, columns=["Feature", "Weight"])
                st.dataframe(fi_df)
        
        except Exception as e:
            st.error(f"Error generating LIME explanation: {e}")
