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

# Set MLflow tracking URI (use local directory or remote server)
TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("customer-churn-prediction")

## 2. Data Preparation


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

# Load and split data
df = prepare_data()
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


## 3. Training with Experiment Tracking

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test, params=None):
    """Train model and log everything to MLflow"""
    
    with mlflow.start_run(run_name=model_name):
        
        # Log parameters
        if params:
            mlflow.log_params(params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model with signature
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            registered_model_name=f"{model_name}_model"
        )
        
        # Log additional artifacts
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        
        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return mlflow.active_run().info.run_id

# Train multiple models
print("Training Models...")

# Random Forest
rf_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}
rf_model = RandomForestClassifier(**rf_params)
rf_run_id = train_and_log_model(
    rf_model, 
    "RandomForest", 
    X_train_scaled, 
    y_train, 
    X_test_scaled, 
    y_test,
    rf_params
)

# Logistic Regression
lr_params = {
    "C": 1.0,
    "random_state": 42,
    "max_iter": 1000
}
lr_model = LogisticRegression(**lr_params)
lr_run_id = train_and_log_model(
    lr_model, 
    "LogisticRegression", 
    X_train_scaled, 
    y_train, 
    X_test_scaled, 
    y_test,
    lr_params
)