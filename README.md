# 💰 Developer Salary Predictor

An interactive web app that estimates a developer's monthly and annual salary from their profile.

🚀 **[Live Demo](https://soumya-salary-predictor.streamlit.app/)**

## Features
- Enter country, education, developer type, work history, industry, years of experience, age and developer level
- Get an estimated monthly and annual salary instantly
- Salary breakdown showing each step of the calculation
- Bar chart comparing the model prediction with the final estimate
- Clean, responsive UI built with Streamlit

## How it works
1. A linear regression model (trained on years of experience) gives a base prediction.
2. The estimate is built on the developer level base salary plus an experience bonus.
3. Country, education, developer type, work history, industry and age adjust the result through multipliers.

The multipliers are assumptions defined at the top of `app.py`, so they can be tuned easily.

## Tech stack
- Python
- Streamlit
- scikit-learn
- NumPy
- Joblib

## Run locally
```bash
git clone https://github.com/Soumyasriau/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction/Employee-Salary-Prediction-main/Employee_Salary_Prediction
pip install -r requirements.txt
streamlit run app.py
```

## Note
This is an estimate, not a guaranteed salary. Actual salaries vary by location, company, skills and market conditions.
