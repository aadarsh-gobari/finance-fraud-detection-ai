import joblib
import numpy as np

# Load model
model = joblib.load("../models/fraud_xgboost_model.pkl")

def predict_fraud(transaction_features):
    """
    Predict fraud probability for a single transaction.
    Input: 1D numpy array of transaction features
    Output: Fraud probability
    """
    transaction_features = transaction_features.reshape(1, -1)
    probability = model.predict_proba(transaction_features)[0][1]
    return probability

# Example usage
if __name__ == "__main__":
    sample_transaction = np.zeros((30,))  # replace with real feature values
    risk_score = predict_fraud(sample_transaction)
    print(f"Fraud Risk Score: {risk_score:.4f}")
