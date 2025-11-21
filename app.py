import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ---------- Config / Paths (uses the dataset path you created earlier) ----------
DATA_PATH = "/mnt/data/retail_churn_dataset.csv"            # dataset path (provided)
MODEL_PATH = "/mnt/data/retail_churn_model_pipeline.pkl"    # model save/load path

st.set_page_config(page_title="Retail Customer Churn Predictor", layout="wide")

# ---------- Utilities ----------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def build_and_train_model(df, save_path=MODEL_PATH, random_state=42):
    # Prepare data (same features as training script)
    target = "churn"
    id_col = "customer_id"
    X = df.drop(columns=[target, id_col])
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Preprocessing
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=250, max_depth=12, random_state=random_state, n_jobs=-1))
    ])

    # CV (quick)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    pipeline.fit(X_train, y_train)

    # Eval
    y_proba = pipeline.predict_proba(X_test)[:,1]
    test_auc = roc_auc_score(y_test, y_proba)

    # Save
    joblib.dump(pipeline, save_path, compress=3)

    return pipeline, cv_scores, test_auc

def predict_single(model, input_df):
    proba = model.predict_proba(input_df)[:,1]
    pred = model.predict(input_df)
    return pred, proba

def get_feature_names(model, X_sample):
    # Build feature names from pipeline (numeric + onehot)
    pre = model.named_steps["preprocessor"]
    numeric_features = X_sample.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_sample.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = []
    if len(categorical_features) > 0:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        cat_cols = list(ohe.get_feature_names_out(categorical_features))
    return numeric_features + cat_cols

# ---------- App UI ----------
st.title("📈 Retail Customer Churn Predictor")

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### Model control")
    model = load_model()
    if model is not None:
        st.success("✅ Loaded model from disk")
        st.write(f"Model path: `{MODEL_PATH}`")
        if st.button("Remove saved model"):
            try:
                os.remove(MODEL_PATH)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Could not remove model: {e}")
    else:
        st.warning("No saved model found.")
        if os.path.exists(DATA_PATH):
            if st.button("Train model using dataset (fast)"):
                with st.spinner("Training model — this may take a minute..."):
                    df = pd.read_csv(DATA_PATH)
                    trained_model, cv_scores, test_auc = build_and_train_model(df)
                    st.success("Model trained and saved.")
                    st.write("CV ROC-AUC:", cv_scores.round(3).tolist())
                    st.write("Test ROC-AUC:", round(test_auc, 4))
                    st.experimental_rerun()
        else:
            st.info("Dataset not found at the expected path:")
            st.code(DATA_PATH)

with col1:
    st.markdown("### Single customer prediction")
    st.markdown("Fill the customer's details and click Predict")
    # If dataset exists, infer fields; otherwise provide defaults
    if os.path.exists(DATA_PATH):
        sample_df = pd.read_csv(DATA_PATH).drop(columns=["churn","customer_id"]).iloc[0:1]
        # Generate inputs dynamically
        user_inputs = {}
        for col in sample_df.columns:
            val = sample_df.iloc[0][col]
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                user_inputs[col] = st.number_input(col, value=float(val))
            else:
                opts = sorted(pd.read_csv(DATA_PATH)[col].dropna().unique().tolist())
                user_inputs[col] = st.selectbox(col, opts, index=0)
        if st.button("Predict single customer"):
            if load_model() is None:
                st.error("No model available. Train it first using the button on the right.")
            else:
                model = load_model()
                input_df = pd.DataFrame([user_inputs])
                pred, proba = predict_single(model, input_df)
                label = "Churn" if pred[0]==1 else "No Churn"
                st.metric("Prediction", label)
                st.metric("Churn probability", f"{proba[0]:.3f}")
                # Show simple feature importance (tree importances on preprocessed features)
                try:
                    feat_names = get_feature_names(model, input_df)
                    importances = model.named_steps["clf"].feature_importances_
                    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(10)
                    st.write("Top feature importances (approx):")
                    st.bar_chart(fi)
                except Exception:
                    st.info("Feature importance not available for this model.")
    else:
        st.info("Dataset not present. Upload dataset (CSV) below or put it at:")
        st.code(DATA_PATH)

st.markdown("---")
st.markdown("### Batch predictions (CSV upload)")
st.markdown("Upload a CSV with same columns as training data (customer_id and churn will be ignored).")

uploaded = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
if uploaded is not None:
    try:
        df_batch = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(df_batch.head())
        if load_model() is None:
            st.error("No model available. Train first or upload a model file.")
        else:
            model = load_model()
            X_batch = df_batch.drop(columns=[c for c in ["churn","customer_id"] if c in df_batch.columns], errors="ignore")
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)[:,1]
            out = df_batch.copy()
            out["pred_churn"] = preds
            out["churn_proba"] = probs
            st.success("Predictions ready.")
            st.dataframe(out.head(50))
            st.download_button("Download predictions CSV", out.to_csv(index=False).encode('utf-8'), file_name="predictions.csv")
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")

st.markdown("---")
st.markdown("#### Advanced / dev")
with st.expander("Upload a trained pipeline (.pkl) to use directly"):
    upload_model = st.file_uploader("Upload joblib .pkl pipeline", type=["pkl"])
    if upload_model is not None:
        bytes_data = upload_model.read()
        with open(MODEL_PATH, "wb") as f:
            f.write(bytes_data)
        st.success(f"Saved uploaded model to {MODEL_PATH}. Reloading app...")
        st.experimental_rerun()

st.markdown("App created by assistant — uses dataset path:")
st.code(DATA_PATH)