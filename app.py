import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
try:
    model = joblib.load("model.pkl")

    df = pd.read_csv("Laptop_Cleaned.csv")
    df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore", inplace=True)

except Exception as e:
    st.error(f"Error Loading Files : {e}")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("💻 About")

st.sidebar.info("""
### Laptop Price Prediction

Machine Learning Model:
- Linear Regression

Framework:
- Streamlit

Developed By:
Vaishnavi Chandak
""")

# -----------------------------
# Title
# -----------------------------
st.title("💻 Laptop Price Predictor")

st.write("Enter laptop specifications to predict its estimated price.")

st.markdown("---")

# -----------------------------
# Input Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    brand = st.selectbox(
        "Brand",
        sorted(df["brand"].dropna().unique())
    )

    name = st.selectbox(
        "Laptop Name",
        sorted(df["name"].dropna().unique())
    )

    processor = st.selectbox(
        "Processor",
        sorted(df["processor"].dropna().unique())
    )

    CPU = st.selectbox(
        "CPU",
        sorted(df["CPU"].dropna().unique())
    )

    spec_rating = st.slider(
        "Specification Rating",
        0.0,
        100.0,
        70.0
    )

    Ram = st.selectbox(
        "RAM (GB)",
        sorted(df["Ram"].dropna().unique())
    )

    Ram_type = st.selectbox(
        "RAM Type",
        sorted(df["Ram_type"].dropna().unique())
    )

with col2:

    ROM = st.selectbox(
        "Storage",
        sorted(df["ROM"].dropna().unique())
    )

    ROM_type = st.selectbox(
        "Storage Type",
        sorted(df["ROM_type"].dropna().unique())
    )

    GPU = st.selectbox(
        "GPU",
        sorted(df["GPU"].dropna().unique())
    )

    display_size = st.number_input(
        "Display Size",
        min_value=10.0,
        max_value=20.0,
        value=15.6
    )

    resolution_width = st.number_input(
        "Resolution Width",
        value=1920
    )

    resolution_height = st.number_input(
        "Resolution Height",
        value=1080
    )

    OS = st.selectbox(
        "Operating System",
        sorted(df["OS"].dropna().unique())
    )

    warranty = st.number_input(
        "Warranty (Years)",
        min_value=0,
        max_value=5,
        value=1
    )

st.markdown("---")

# -----------------------------
# Prediction
# -----------------------------
if st.button("💰 Predict Price", use_container_width=True):

    input_df = pd.DataFrame({

        "brand":[brand],
        "name":[name],
        "spec_rating":[spec_rating],
        "processor":[processor],
        "CPU":[CPU],
        "Ram":[Ram],
        "Ram_type":[Ram_type],
        "ROM":[ROM],
        "ROM_type":[ROM_type],
        "GPU":[GPU],
        "display_size":[display_size],
        "resolution_width":[resolution_width],
        "resolution_height":[resolution_height],
        "OS":[OS],
        "warranty":[warranty]

    })

    try:

        prediction = model.predict(input_df)

        st.success(f"## 💰 Estimated Laptop Price : ₹ {prediction[0]:,.2f}")

        st.balloons()

    except Exception as e:

        st.error(f"Prediction Error : {e}")

        st.write("Input Data")
        st.write(input_df)

        st.write("Data Types")
        st.write(input_df.dtypes)

st.markdown("---")

st.caption("Developed by Vaishnavi Chandak | Streamlit | Machine Learning")