import streamlit as st
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="SHAP Demo", layout="wide")
st.title("SHAP Values Demo")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    target_col = st.selectbox("Select Target Column", data.columns)

    if st.button("Train Model & Show SHAP"):
        X = data.drop(columns=[target_col])
        y = data[target_col]

        X = X.select_dtypes(include=["number"])

        model = RandomForestClassifier(random_state=1)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        st.subheader("Feature Importance")
        try:
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(bbox_inches="tight")
        except Exception as e:
            st.error(f"SHAP plot error: {e}")
else:
    st.info("Upload a CSV file to begin.")
