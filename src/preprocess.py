import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('data/customer_data.csv')

# Print the column names to debug the issue
print("Columns in the dataset before preprocessing:", df.columns)

# Handle common column name variations (e.g., lower/upper case, spaces)
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
df.columns = df.columns.str.lower()  # Convert column names to lowercase for consistency

# Specify the correct column names for categorical encoding
categorical_columns = ['state', 'international plan', 'voice mail plan']  # Correct column names in lowercase

# Use LabelEncoder for ordinal encoding
label_encoder = LabelEncoder()
for column in categorical_columns:
    if column in df.columns:
        df[column] = label_encoder.fit_transform(df[column])
    else:
        print(f"Warning: '{column}' column not found in the dataset!")

# Ensure the 'Churn' column is retained in the processed data
# Assuming 'Churn' is your target column, retain it during processing
df['Churn'] = df['churn']  # Ensure 'Churn' is included in the processed data

# Save the processed data
df.to_csv('data/processed_data.csv', index=False)

print("Columns in the dataset after preprocessing:", df.columns)


