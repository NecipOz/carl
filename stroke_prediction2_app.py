import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import pickle

# Streamlit UI for user inputs
st.title("Stroke Prediction App")

# User inputs for the features
age = st.slider("Age", 18, 100, 45)
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])

# Convert categorical inputs to numerical values for the model
heart_disease = int(heart_disease == "Yes")
hypertension = int(hypertension == "Yes")
#smoking_status = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}[smoking_status]
#residence_type = int(residence_type == "Urban")
#work_type = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}[work_type]
#ever_married = int(ever_married == "Yes")

# Create a DataFrame with user input values
user_data = pd.DataFrame([[age, avg_glucose_level, bmi, heart_disease, hypertension, smoking_status, 
                           residence_type, work_type, ever_married]], 
                         columns=['age', 'avg_glucose_level', 'bmi', 'heart_disease', 'hypertension', 
                                  'smoking_status', 'Residence_type', 'work_type', 'ever_married'])

# Load the pre-trained model pipeline
with open("stroke_model.pkl", "rb") as file:
    model_pipeline = pickle.load(file)

# Display the raw user data for debugging
st.write("User Input Data:")
st.write(user_data)

# Make prediction
if st.button("Predict Stroke Risk"):
    try:
        # Use the model pipeline to make a prediction
        prediction = model_pipeline.predict(user_data)
        if prediction[0] == 1:
            st.write("The model predicts a high risk of stroke.")
        else:
            st.write("The model predicts a low risk of stroke.")
    except Exception as e:
        st.write(f"Error in prediction: {e}")

# Additional information
st.write("""
**Note**: This prediction is based on statistical data and a machine learning model. It does not replace professional medical advice.
""")