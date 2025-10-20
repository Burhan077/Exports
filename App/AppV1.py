import os
import pickle
import pandas as pd
import streamlit as st
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "Training Data", "model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found!\nExpected at: {MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
st.set_page_config(page_title="Customs Value Predictor", page_icon="💼")
st.title("Customs Value Predictor")
st.caption("Predict shipment customs value in BWP using export data features.")
st.subheader("Enter Shipment Details")
gross_weight = st.number_input("Gross Weight (KGM)", min_value=0.0, step=0.1)
net_weight = st.number_input("Net Weight (KGM)", min_value=0.0, step=0.1)
quantity = st.number_input("Quantity", min_value=0.0, step=0.1)
destination = st.text_input("Destination Country", value="Namibia")
hs_code = st.text_input("HS CODE", value="2012010")
input_data = pd.DataFrame({
    "GROSS WEIGHT": [gross_weight],
    "NET WEIGHT": [net_weight],
    "QUANTITY": [quantity],
    "Destination Country": [destination],
    "HS CODE": [hs_code]
})
expected_columns = [
    'TYPE','CPC DESCRIPTION','Origin Country','Destination Country','HS CODE',
    'PACKAGE TYPE','QUANTITY UOM','QUANTITY','NO OF PACKAGE TYPE','GROSS WEIGHT',
    'GROSS WEIGHT UOM','NET WEIGHT','NET WEIGHT UOM','REGIME','DECLARATION OFFICE',
    'PORT OF ENTRY','ITEM NO','CPC GROUP CODE','CPC CODE','CPC GROUP DESCRIPTION',
    'Chapter','HEADING','SUB HEADING','MONTH','YEAR','INVOICE AMOUNT BWP','FREIGHT BWP'
]
numeric_defaults = {
    "QUANTITY": 0.0,
    "NO OF PACKAGE TYPE": 0.0,
    "GROSS WEIGHT": gross_weight,
    "NET WEIGHT": net_weight,
    "ITEM NO": 0.0,
    "INVOICE AMOUNT BWP": 0.0,
    "FREIGHT BWP": 0.0
}
categorical_defaults = {
    "TYPE": "Export",
    "CPC DESCRIPTION": "Direct Exports of home produced goods",
    "Origin Country": "BOTSWANA",
    "Destination Country": destination,
    "HS CODE": hs_code,
    "PACKAGE TYPE": "EACH",
    "QUANTITY UOM": "KGM",
    "GROSS WEIGHT UOM": "KGM",
    "NET WEIGHT UOM": "KGM",
    "REGIME": "1",
    "DECLARATION OFFICE": "Kasane Inland Office",
    "PORT OF ENTRY": "Mpacha",
    "CPC GROUP CODE": "1000",
    "CPC CODE": "10000",
    "CPC GROUP DESCRIPTION": "Direct permanent export",
    "Chapter": "2",
    "HEADING": "201",
    "SUB HEADING": "20120",
    "MONTH": "November",
    "YEAR": 2019
}
for col in expected_columns:
    if col not in input_data.columns:
        if col in numeric_defaults:
            input_data[col] = numeric_defaults[col]
        elif col in categorical_defaults:
            input_data[col] = categorical_defaults[col]
        else:
            # Default fallback
            input_data[col] = 0 if "BWP" in col or "WEIGHT" in col or "QUANTITY" in col else "Unknown"
input_data = input_data[expected_columns]
if st.button("Predict Customs Value"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Customs Value: {prediction:,.2f} BWP")
    except Exception as e:
        st.error(f"Prediction failed:\n{e}")
st.markdown("---")
