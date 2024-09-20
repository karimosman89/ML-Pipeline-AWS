# Import necessary libraries
import pandas as pd
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn import SKLearn

# Set up AWS and SageMaker
role = get_execution_role()
s3_bucket = 'your-s3-bucket-name'
s3_prefix = 'customer-churn'

# Load data
data = pd.read_csv('telco_churn.csv')
data.to_csv('processed_data.csv', index=False)

# Upload to S3
s3_client = boto3.client('s3')
s3_client.upload_file('processed_data.csv', s3_bucket, f'{s3_prefix}/processed_data.csv')

# Define a Scikit-Learn script for training
sklearn_script = '''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

def train():
    # Load data
    data = pd.read_csv('/opt/ml/input/data/train/processed_data.csv')
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, '/opt/ml/model/model.joblib')

if __name__ == '__main__':
    train()
'''

# Save the training script to a file
with open('train.py', 'w') as f:
    f.write(sklearn_script)

# Define the estimator
estimator = SKLearn(
    entry_point='train.py',
    role=role,
    train_instance_type='ml.m5.large',
    source_dir='.',
    output_path=f's3://{s3_bucket}/{s3_prefix}/model',
)

# Train the model
estimator.fit({'train': f's3://{s3_bucket}/{s3_prefix}/processed_data.csv'})
