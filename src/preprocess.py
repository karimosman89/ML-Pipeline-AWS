import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('data/customer_data.csv')

# Handle categorical data with LabelEncoder or pd.get_dummies()
categorical_columns = ['State', 'Gender']  # Add relevant columns here

# Option 1: Use LabelEncoder for ordinal encoding
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Option 2: Use one-hot encoding if categorical data has no ordinal relationship
df = pd.get_dummies(df, columns=categorical_columns)

# Save the processed data
df.to_csv('data/processed_data.csv', index=False)
