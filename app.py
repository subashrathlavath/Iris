import streamlit as st
import numpy as np
import pickle
import joblib
from sklearn.tree import DecisionTreeClassifier 
# Load the trained model
model = joblib.load("model.pkl")
#print(type(model))  # Should print <class 'sklearn.tree.DecisionTreeClassifier'>


# Streamlit UI
st.title("🌱 Iris Flower Classification - Decision Tree Model")

# Input fields for user to enter feature values
sepal_length = st.number_input("Sepal Length", min_value=4.0, max_value=8.0, value=5.1)
sepal_width = st.number_input("Sepal Width", min_value=2.0, max_value=5.0, value=3.5)
petal_length = st.number_input("Petal Length", min_value=1.0, max_value=7.0, value=1.4)
petal_width = st.number_input("Petal Width", min_value=0.1, max_value=2.5, value=0.2)

# Prediction Button
if st.button("Predict"):
    # Prepare input features
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Display result
    st.success(f"🌸 Predicted Flower Type: {prediction[0]}")
