import mlflow
import mlflow.sklearn
import numpy as np

def load_production_model(model_name, version="Latest"):
    """Load the production version of a model"""
    
    TRACKING_URI = "http://127.0.0.1:8080"  
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model

# Load production model
production_model = load_production_model("RandomForest_model", version="Latest")

# Make predictions
sample_data = np.array([[-1.418269, 0.865052, -0.773136, -1.224745, 0.447214]])
predictions = production_model.predict(sample_data)
print(f"Predictions: {predictions}")