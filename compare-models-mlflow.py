import mlflow

def compare_models(experiment_name):
    """Compare all models in an experiment"""
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Display comparison
    comparison = runs[['run_id', 'params.n_estimators', 'metrics.accuracy', 
                       'metrics.f1_score', 'start_time']].sort_values(
        by='metrics.f1_score', ascending=False
    )
    
    print("\n=== Model Comparison ===")
    print(comparison)
    
    return comparison

# Compare all models
compare_models("customer-churn-prediction")
