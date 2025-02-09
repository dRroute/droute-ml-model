from django.shortcuts import render

# Create your views here.
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

# Get the current script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the model path dynamically
model_path = os.path.join(base_dir, "..", "fare_prediction_model.pkl")  # Adjust according to folder structure

# Ensure the file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")

# Define your column names (same as your training data)
column_names = [
    "Source_Latitude", "Source_Longitude", "Destination_Latitude", "Destination_Longitude",
    "Distance_km", "Weight_kg", "Volume_cuft", "Pickup_Hour", "Delivery_Hour", "Day_of_Week", "Goods_Type"
]

@api_view(['POST'])
def predict_fare(request):
    try:
        # Extract input data from request
        data = request.data

        # Process the input data into the format your model expects
        pickup_datetime = datetime.strptime(data["Pickup_Time"], "%d-%m-%Y %H:%M")
        pickup_hour = pickup_datetime.hour
        day_of_week = pickup_datetime.weekday()  # Monday = 0, Sunday = 6

        delivery_hour = int(data["Delivery_Time"].split(":")[0])  # Extracting hour from "54:00.4"

        # Prepare the input for prediction
        formatted_input = np.array([[
            data["Source_Latitude"],
            data["Source_Longitude"],
            data["Destination_Latitude"],
            data["Destination_Longitude"],
            data["Distance_km"],
            data["Weight_kg"],
            data["Volume_cuft"],
            pickup_hour,
            delivery_hour,
            day_of_week,
            data["Goods_Type"]
        ]])

        # Convert to DataFrame
        formatted_input_df = pd.DataFrame(formatted_input, columns=column_names)

        # Predict the fare
        predicted_fare = model.predict(formatted_input_df)
        return Response({"predicted_fare": predicted_fare[0]}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
