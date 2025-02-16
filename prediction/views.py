import os
import gdown
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

# Define the Google Drive file ID (Extracted from your link)
file_id = "1fsOHsiXtnrqePv_ooZq8JYFiyts5EGKO"
download_url = f"https://drive.google.com/uc?id={file_id}"

# Set the path where the model should be saved
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "fare_prediction_model.pkl")

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    print("Downloading the model file...")
    gdown.download(download_url, model_path, quiet=False)
    print("Model downloaded successfully!")

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")

# Define column names (same as training data)
column_names = [
    "Source_Latitude", "Source_Longitude", "Destination_Latitude", "Destination_Longitude",
    "Distance_km", "Weight_kg", "Volume_cuft", "Pickup_Hour", "Delivery_Hour", "Day_of_Week", "Goods_Type"
]

@api_view(['POST'])
def predict_fare(request):
    try:
        data = request.data

        # Convert Pickup_Time to datetime
        pickup_datetime = datetime.strptime(data["Pickup_Time"], "%d-%m-%Y %H:%M")
        pickup_hour = pickup_datetime.hour
        day_of_week = pickup_datetime.weekday()

        # Extract Delivery Hour
        delivery_hour = int(data["Delivery_Time"].split(":")[0])

        # Prepare input for prediction
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

        formatted_input_df = pd.DataFrame(formatted_input, columns=column_names)

        # Predict fare
        predicted_fare = model.predict(formatted_input_df)
        return Response({"predicted_fare": predicted_fare[0]}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
