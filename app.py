import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd 
import pickle

# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)

# Streamlit app
# Set the title of the app
st.title("Customer Churn Prediction")

# Create input fields for user to enter customer data
geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", 18, 92)
balance= st.number_input("Balance", min_value=0.0)
credit_score = st.number_input("Credit Score")
num_of_products = st.slider("Number of Products", 1, 4)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
tenure = st.slider("Tenure", 0, 10) 
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])


# Prepare the input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One hot encode the 'Geography' feature
geography_encoded = onehot_encoder.transform([[geography]]).toarray()
input_data = pd.concat([input_data, pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))], axis=1)  

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict the churn probability
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.write(f"Churn Probability: {prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")