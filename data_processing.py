import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
print("Loading the dataset...")
data = pd.read_csv("forestfires.csv")
print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

# Display the first few rows of the dataset
print("\nInitial Dataset Sample:")
print(data.head())

# Step 1: Transform 'area' into binary classification
print("\nTransforming 'area' column into binary classification...")
data['area'] = data['area'].apply(lambda x: 1 if x > 0 else 0)
print(f"Class distribution after transformation:\n{data['area'].value_counts()}")

# Step 2: Handle outliers using the IQR method
print("\nHandling outliers using the IQR method...")
numerical_columns = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
for col in numerical_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[col] = np.clip(data[col], lower_bound, upper_bound)
    print(f"Processed {col} - Outliers capped at bounds [{lower_bound}, {upper_bound}]")

# Step 3: Scale numerical features
print("\nScaling numerical features...")
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
print("Numerical features scaled.")

# Step 4: Encode 'month' and 'day' with label encoding
print("\nEncoding 'month' and 'day' columns...")
label_encoder = LabelEncoder()
data['month'] = label_encoder.fit_transform(data['month'])
data['day'] = label_encoder.fit_transform(data['day'])
print("Encoded 'month' and 'day'.")

# Step 5: Feature engineering (optional)
print("\nFeature engineering...")
data['temp_rain_ratio'] = data['temp'] / (data['rain'] + 1)  # Prevent division by zero
data['wind_rain_interaction'] = data['wind'] * data['rain']
print("Engineered new features: 'temp_rain_ratio', 'wind_rain_interaction'.")

# Step 6: Split into train and test datasets
print("\nSplitting dataset into train and test sets...")
X = data.drop(columns=['area'])
y = data['area']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the datasets
print("\nSaving processed datasets...")
X_train.to_csv("X_train_processed.csv", index=False)
X_test.to_csv("X_test_processed.csv", index=False)
y_train.to_csv("y_train_processed.csv", index=False)
y_test.to_csv("y_test_processed.csv", index=False)
print("Datasets saved as 'X_train_processed.csv', 'X_test_processed.csv', 'y_train_processed.csv', 'y_test_processed.csv'.")

# Display processed dataset sample
print("\nProcessed Dataset Sample:")
print(data.head())
