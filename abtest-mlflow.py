import hashlib
import random
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(TRACKING_URI)

class ABTestFramework:

    """A/B testing framework for model comparison"""
    
    
    def __init__(self, model_a_name, model_b_name, traffic_split=0.5):
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.traffic_split = traffic_split
        
        self.model_a = mlflow.sklearn.load_model(f"models:/{self.model_a_name}/{"Latest"}")
        self.model_b = mlflow.sklearn.load_model(f"models:/{self.model_b_name}/{"Latest"}")
        
        self.results = {
            'model_a': {'predictions': [], 'actual': [], 'latencies': []},
            'model_b': {'predictions': [], 'actual': [], 'latencies': []}
        }
    
    def assign_variant(self, user_id):
        """Assign user to model variant (consistent hashing)"""
        
        hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        return 'model_a' if (hash_value % 100) < (self.traffic_split * 100) else 'model_b'
    
    def predict(self, user_id, features):
        """Make prediction using assigned model variant"""
        
        variant = self.assign_variant(user_id)
        model = self.model_a if variant == 'model_a' else self.model_b
        
        import time
        start_time = time.time()
        prediction = model.predict(features)
        latency = time.time() - start_time
        
        # Log prediction
        with mlflow.start_run(run_name=f"ab_test_prediction_{variant}"):
            mlflow.set_tag("variant", variant)
            mlflow.set_tag("user_id", user_id)
            mlflow.log_metric("prediction_latency", latency)
        
        return prediction, variant, latency
    
    def record_outcome(self, variant, prediction, actual, latency):
        """Record prediction outcome for analysis"""
        
        self.results[variant]['predictions'].append(prediction)
        self.results[variant]['actual'].append(actual)
        self.results[variant]['latencies'].append(latency)
    
    def analyze_results(self):
        """Analyze A/B test results"""
        
        results_summary = {}
        
        for variant in ['model_a', 'model_b']:
            if len(self.results[variant]['predictions']) > 0:
                accuracy = accuracy_score(
                    self.results[variant]['actual'],
                    self.results[variant]['predictions']
                )
                avg_latency = np.mean(self.results[variant]['latencies'])
                
                results_summary[variant] = {
                    'accuracy': accuracy,
                    'avg_latency': avg_latency,
                    'sample_size': len(self.results[variant]['predictions'])
                }
                
                print(f"\n{variant} Results:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Avg Latency: {avg_latency:.4f}s")
                print(f"  Sample Size: {results_summary[variant]['sample_size']}")
        
        # Log results to MLflow
        with mlflow.start_run(run_name="ab_test_results"):
            for variant, metrics in results_summary.items():
                mlflow.log_metric(f"{variant}_accuracy", metrics['accuracy'])
                mlflow.log_metric(f"{variant}_latency", metrics['avg_latency'])
                mlflow.log_metric(f"{variant}_samples", metrics['sample_size'])
            
            mlflow.log_dict(results_summary, "ab_test_summary.json")
        
        return results_summary
    
    def determine_winner(self, min_samples=100, confidence_level=0.95):
        """Determine winning model using statistical test"""
        
        from scipy import stats
        
        if (len(self.results['model_a']['predictions']) < min_samples or 
            len(self.results['model_b']['predictions']) < min_samples):
            print(f"Insufficient samples for statistical significance (need {min_samples})")
            return None
        
        # Convert to numpy arrays
        a_correct = np.array(self.results['model_a']['predictions']) == np.array(self.results['model_a']['actual'])
        b_correct = np.array(self.results['model_b']['predictions']) == np.array(self.results['model_b']['actual'])
        
        # Perform two-sample t-test
        t_stat, p_value = stats.ttest_ind(a_correct, b_correct)
        
        print(f"\nStatistical Test Results:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < (1 - confidence_level):
            winner = 'model_a' if np.mean(a_correct) > np.mean(b_correct) else 'model_b'
            print(f"  Winner: {winner} (statistically significant)")
            
            with mlflow.start_run(run_name="ab_test_winner"):
                mlflow.set_tag("winner", winner)
                mlflow.log_metric("p_value", p_value)
                mlflow.log_metric("t_statistic", t_stat)
            
            return winner
        else:
            print("  No statistically significant difference")
            return None

# Example A/B test
ab_test = ABTestFramework("RandomForest_model", "LogisticRegression_model", traffic_split=0.5)

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

df = prepare_data()  # In production, load from database
X = df.drop('churn', axis=1)
y = df['churn']
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Simulate predictions
for i in range(500):
    user_id = f"user_{i}"
    features = X_test_scaled[i % len(X_test_scaled)].reshape(1, -1)
    actual = y_test.iloc[i % len(y_test)]
    
    prediction, variant, latency = ab_test.predict(user_id, features)
    ab_test.record_outcome(variant, prediction[0], actual, latency)

# Analyze results
results = ab_test.analyze_results()
winner = ab_test.determine_winner()