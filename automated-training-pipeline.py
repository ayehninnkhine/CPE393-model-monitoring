import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI (use local directory or remote server)
TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("customer-churn-prediction")
client = MlflowClient(TRACKING_URI)

def prepare_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic features
    data = {
        'tenure': np.random.randint(0, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice([0, 1, 2], n_samples),
        'payment_method': np.random.choice([0, 1, 2, 3], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with some logic
    df['churn'] = ((df['tenure'] < 12) | 
                   (df['monthly_charges'] > 80)).astype(int)
    
    return df

def automated_training_pipeline():
    """Automated pipeline for model retraining"""
    
    print("Starting automated training pipeline...")
    
    # 1. Load new data
    df = prepare_data()  # In production, load from database
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Train new model
    rf_params = {"n_estimators": 30, "max_depth": 10, "random_state": 42}
    rf_model = RandomForestClassifier(**rf_params)

    rf_model.fit(X_train_scaled, y_train)

    # 3. Compare with production model
    production_model = load_production_model("RandomForest_model", version="Latest")
    y_pred_prod = production_model.predict(X_test_scaled)
    prod_accuracy = accuracy_score(y_test, y_pred_prod)
    y_pred_new = rf_model.predict(X_test_scaled)
    new_accuracy = accuracy_score(y_test, y_pred_new)
    
    print(f"Production model accuracy: {prod_accuracy:.4f}")
    print(f"New model accuracy: {new_accuracy:.4f}")
    
    # 4. Promote if better
    if new_accuracy >= prod_accuracy:
        signature = infer_signature(X_train_scaled, y_train)
        mlflow.sklearn.log_model(
            rf_model, 
            "model",
            signature=signature,
            registered_model_name=f"RandomForest_model"
        )
        
        print("Production model accuracy", prod_accuracy)
        print("New model accuracy", new_accuracy)
        
        print("New model version registered.")
    else:
        print("Production model retained.")
    
def load_production_model(model_name, version="Latest"):
    """Load the production version of a model"""
    
    TRACKING_URI = "http://127.0.0.1:8080"  
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model


# Run pipeline
automated_training_pipeline()