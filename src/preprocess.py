import pandas as pd

def preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    # Basic preprocessing steps
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

if __name__ == "__main__":
    df = preprocess_data('data/customer_data.csv')
    df.to_csv('data/processed_data.csv', index=False)

