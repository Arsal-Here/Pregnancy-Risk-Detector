import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load Lottie animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load model
model = joblib.load("model.pkl")  # Replace with your actual model
df = pd.read_csv("dataset.csv")
# Page Config
st.set_page_config(page_title="Pregnancy Risk Detector", page_icon="🩺", layout="wide")

# Custom CSS for vibrant styling
st.markdown("""
    <style>
        .main {
            background-color: #fff7f9;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #d63384;
            text-align: center;
        }
        .stButton > button {
            background-color: #ff4d6d;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1.5rem;
            border-radius: 10px;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #c9184a;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
lottie_animation = load_lottiefile("Pregnant.json")  # Download from lottiefiles.com
with st.container():
    st.markdown("<h1> <p style='font-size:70px; color:#FF69B4'> Pregnancy Risk Level Detector</h1>", unsafe_allow_html=True)
    
left,right = st.columns(2)
with left:
    st.markdown("<p style='font-size:35px'>This Risk Level Detector uses Machine Learning to predict High Risk Pregnancies considering various factors and boasts a 89% accuracy rate </p>",unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px'>The data used for this model is a sample dataset from the website below</p> ",unsafe_allow_html=True)
    st.markdown("https://www.kaggle.com/datasets/vmohammedraiyyan/maternal-health-and-high-risk-pregnancy-dataset")

with right:
    st_lottie(lottie_animation, height=300, speed=1, key="pregnancy")


#middle part
left,right = st.columns(2)
with left:
    fig,ax = plt.subplots()
    sns.scatterplot(x=df['Age'],y=df['BMI'],hue=df['Risk Level'])
    st.pyplot(fig)

with right:
    st.markdown("""
        <ul style='color:#D3D3D3; font-size:25px;'>
          <li><strong>Approximately 20 million high‑risk pregnancies occur annually worldwide</strong>, representing about <strong>6–7% of all births</strong>, particularly in low‑ and lower‑middle‑income countries</li>
          <li><strong>Hypertensive disorders of pregnancy (e.g., preeclampsia)</strong> affect <strong>5–10% of pregnancies globally</strong>, contributing to maternal deaths and adverse perinatal outcomes.</li>
          <li><strong>Gestational diabetes affects about 16.7% of all births globally, with higher rates in low- and middle-income countries (~14–15%)</strong></li>
        </ul>
        """, unsafe_allow_html=True)


st.subheader("🩺 Predict the risk level based on pregnancy-related health metrics")
prediction = 0
# Input Form
with st.form("risk_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=19, max_value=50, step=1)
        systolic = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=200)
        diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140)
        bs = st.number_input("Blood Sugar (mmol/L)", min_value=3.0, max_value=11.0)
        temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=105.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)

    with col2:
        prev_comp = st.selectbox("Previous Complications", ["No", "Yes"])
        pre_diabetes = st.selectbox("Preexisting Diabetes", ["No", "Yes"])
        gest_diabetes = st.selectbox("Gestational Diabetes", ["No", "Yes"])
        mental_health = st.selectbox("Mental Health Issues", ["No", "Yes"])
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=50, max_value=160)

    submit = st.form_submit_button("🔍 Predict Risk Level")

    if submit:
        # Prepare input
        input_df = pd.DataFrame([[
            age, systolic, diastolic, bs, temp, bmi,
            1 if prev_comp == "Yes" else 0,
            1 if pre_diabetes == "Yes" else 0,
            1 if gest_diabetes == "Yes" else 0,
            1 if mental_health == "Yes" else 0,
            heart_rate
        ]], columns=[
            'Age','Systolic BP','Diastolic','BS','Body Temp','BMI',
            'Previous Complications','Preexisting Diabetes',
            'Gestational Diabetes','Mental Health','Heart Rate'
        ])

        # Make prediction
        prediction = model.predict(input_df)[0]

        result = "High Risk Pregnancy" if prediction==1 else "Low Risk Pregnancy"

        # Output
        st.success(f"*Predicted Risk Level:* **{(result) }**")

if prediction: # high risk pregnancy
    st.markdown("<h3 style='color:#87CEEB;'>What to Do After Knowing You Have a High-Risk Pregnancy</h3>", unsafe_allow_html=True)
    l_col,r_col = st.columns(2)
    with l_col:
        with st.container():
            st.markdown("""
            
                        
            - **Follow Doctor’s Advice**
              - Regular checkups with a specialist (OB-GYN or maternal-fetal medicine doctor)
              - Take all medications as prescribed

            - **Adopt a Healthy Lifestyle**
              - Eat a balanced diet, stay hydrated
              - Get enough sleep and do safe exercise if allowed

            - **Avoid Harmful Habits**
              - No smoking, alcohol, drugs, or self-medicating
            """)
    with r_col:
        with st.container():
            st.markdown("""
            - **Monitor Symptoms Closely**
              - Track blood pressure, blood sugar, weight, and baby’s movements
              - Watch for warning signs (e.g., bleeding, swelling, vision changes)

            - **Build a Support System**
              - Get help from family, friends, or support groups
              - Seek mental health support if needed

            - **Prepare for Delivery Early**
              - Plan with your doctor for possible early delivery or C-section
              - Choose a hospital with a NICU if needed
            """)





