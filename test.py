import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression

# Load Dataset
file_path = 'HDI.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Dataset Dimensions
print(f"\nDimensions of the dataset: {data.shape}")

# Dataset Summary
print("\nSummary of the dataset:")
print(data.info())
print("\nStatistical Summary of Numerical Columns:")
print(data.describe())

# Handling Missing Values
print("\nHandling Missing Values:")
missing_values = data.isnull().sum()
print(f"Missing values in each column before imputation:\n{missing_values}")

# Impute numerical columns with their mean
data.fillna(data.mean(numeric_only=True), inplace=True)

# Impute categorical columns with their mode
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

missing_values_after = data.isnull().sum()
print(f"Missing values in each column after imputation:\n{missing_values_after}")

# Detecting and Dropping Duplicates
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
data = data.drop_duplicates()
print("Duplicates removed.")

# Univariate Analysis
print("\nUnivariate Analysis:")
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    print(f"\nAnalysis for {column}:")
    print(data[column].describe())

# Correlation Matrix
print("\nCorrelation Matrix:")
numeric_data = data.select_dtypes(include=['float64', 'int64'])
numeric_data.fillna(numeric_data.mean(), inplace=True)
correlation_matrix = numeric_data.corr()
print(correlation_matrix)

# Group-wise Analysis
for cat_col in categorical_columns:
    print(f"\nGroup-wise Statistics for {cat_col}:")
    group_stats = data.groupby(cat_col)[numeric_data.columns].mean()
    print(group_stats)

# Feature Engineering
print("\nFeature Engineering:")
# Log Transform (For Skewed Data)
for col in numeric_data.columns:
    data[f"log_{col}"] = np.log1p(data[col].replace(0, np.nan)).fillna(0)  # Log transform avoids log(0)
    data[f"sqrt_{col}"] = np.sqrt(data[col])

# Interaction Features
if len(numeric_data.columns) >= 2:
    data['interaction_feature'] = data[numeric_data.columns[0]] * data[numeric_data.columns[1]]
    print("Interaction feature created by multiplying the first two numerical columns.")

# Encoding Categorical Variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[f"{col}_encoded"] = le.fit_transform(data[col])
    label_encoders[col] = le
    print(f"Label encoding applied to: {col}")

# One-Hot Encoding
one_hot_encoded_data = pd.get_dummies(data[categorical_columns], drop_first=True)
data = pd.concat([data, one_hot_encoded_data], axis=1)
print("One-hot encoding applied to categorical variables.")

# Clean and Scale Numeric Data
print("\nCleaning and Scaling Numeric Data:")
# Replace Inf/-Inf with NaN
data[numeric_data.columns] = data[numeric_data.columns].replace([np.inf, -np.inf], np.nan)

# Impute missing values (if any remain) with column mean
data[numeric_data.columns] = data[numeric_data.columns].fillna(data[numeric_data.columns].mean())

# Standardization and Normalization
scaler = StandardScaler()
standardized = scaler.fit_transform(data[numeric_data.columns])

normalized_data = MinMaxScaler()
normalized = normalized_data.fit_transform(data[numeric_data.columns])

# Save Scaled Data
standardized_df = pd.DataFrame(standardized, columns=numeric_data.columns)
normalized_df = pd.DataFrame(normalized, columns=numeric_data.columns)

standardized_df.to_csv('standardized_data.csv', index=False)
normalized_df.to_csv('normalized_data.csv', index=False)
print("\nStandardized and Normalized datasets saved.")

# Feature Selection
print("\nFeature Selection:")
# Assuming 'target' column exists for selection
if 'target' in data.columns:
    y = data['target']
    X = data.drop(['target'], axis=1)

    # ANOVA F-test
    f_test_selector = SelectKBest(score_func=f_classif, k=5)
    f_test_selector.fit(X.select_dtypes(include=['float64', 'int64']), y)
    f_test_selected = f_test_selector.get_support(indices=True)
    print("Top 5 features selected using ANOVA F-test:", X.columns[f_test_selected].tolist())

    # Mutual Information Regression
    mi_selector = SelectKBest(score_func=mutual_info_regression, k=5)
    mi_selector.fit(X.select_dtypes(include=['float64', 'int64']), y)
    mi_selected = mi_selector.get_support(indices=True)
    print("Top 5 features selected using Mutual Information Regression:", X.columns[mi_selected].tolist())

# Save the cleaned dataset
output_file_path = 'cleaned_HDI.csv'
data.to_csv(output_file_path, index=False)
print(f"\nCleaned dataset saved as {output_file_path}")



