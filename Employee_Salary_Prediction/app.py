import streamlit as st
import joblib
import numpy as np
import base64

# Function to set background image using base64 encoding
def set_bg_image(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background (make sure img.jpg is in the same directory)
set_bg_image("img.jpg")

# Load trained model
model = joblib.load("linearmodel.pkl")

# Title and Description
st.title("💼 Salary Prediction App")
st.divider()
st.write("Estimate your monthly salary based on your **experience** and **job rate**.")

# Inputs
months = st.number_input("Enter your experience (in days)", value=1, step=1, min_value=0)
job_rate = st.slider("Enter your job rate (1: Intern ➜ 5: Lead)", min_value=1, max_value=5, value=3)

# Prepare input for model
X = np.array([[months]])

st.divider()

# Predict Button
predict = st.button("📈 Predict Salary")
st.divider()

# Output
if predict:
    base_prediction = model.predict(X)[0]

    # Bonus logic
    bonus = 0
    if months >= 12 and job_rate > 2:
        bonus = (job_rate - 2) * 10000  # ₹10,000 per level above 2

    final_salary = base_prediction + bonus

    st.balloons()
    st.write(f"💰 **Estimated Monthly Salary:** ₹{final_salary:,.2f}")
    st.write(f"📅 **Estimated Annual Salary:** ₹{final_salary * 12:,.2f}")
    st.info("📌 Adjusted salary includes job rate bonus for experienced employees.")
else:
    st.write("Press the **Predict Salary** button to view your estimated salary.")
