import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

# =========================
# CHECK CURRENT DIRECTORY
# =========================
print("Current Working Directory:")
print(os.getcwd())

print("\nFiles Present:")
print(os.listdir())

# =========================
# LOAD MODEL
# =========================

# OPTION 1:
# If model file is in same folder as app.py

model = load_model("lstm_model.h5")


# OPTION 2:
# If model file is in another folder
# Uncomment below and use full path

# model = load_model(
#     "/home/intellect/Documents/p2project/models/lstm_model.h5"
# )

# =========================
# STREAMLIT UI
# =========================

st.title("⚡ Electricity Consumption Forecasting")

st.write("LSTM Model Loaded Successfully ✅")

# Example input
sample_input = np.random.rand(1, 24, 1)

# Prediction
prediction = model.predict(sample_input)

st.subheader("Prediction Output")
st.write(prediction)