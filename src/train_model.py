import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'model.pkl')

if __name__ == "__main__":
    train_model('data/processed_data.csv')

