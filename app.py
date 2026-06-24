
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="SHAP Explanation App", layout="wide")
st.title("SHAP Values Explorer")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.write("Preview Data", data.head())

    target = st.selectbox("Select target column", data.columns)

    X = data.drop(columns=[target])
    X = X.select_dtypes(include=[np.number])

    y = data[target]
    if y.dtype == object:
        y = y.astype("category").cat.codes

    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    model = RandomForestClassifier(random_state=1)
    model.fit(train_X, train_y)

    st.success("Model trained successfully")

    row_id = st.number_input(
        "Row to explain", min_value=0, max_value=len(val_X)-1, value=0
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(val_X.iloc[[row_id]])

    st.subheader("Prediction")
    st.write(model.predict_proba(val_X.iloc[[row_id]]))

    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(
        explainer.shap_values(val_X),
        val_X,
        show=False
    )
    st.pyplot(fig)

    st.subheader("SHAP Waterfall Plot (Selected Row)")
    try:
        fig2 = plt.figure()
        sv = explainer(val_X.iloc[[row_id]])
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Waterfall plot unavailable: {e}")
else:
    st.info("Upload a CSV file to begin.")
