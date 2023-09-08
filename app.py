# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Data Collection
# Replace 'data.csv' with the path to your dataset file.
data = pd.read_csv('data.csv')

# Step 2: Data Preprocessing
# Handle missing values
data.dropna(inplace=True)

# Encode categorical variables (if applicable)
data = pd.get_dummies(data, columns=['categorical_feature'])

# Split the data into features (X) and target variable (y)
X = data.drop('diabetes_status', axis=1)
y = data['diabetes_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Data Splitting (Optional)
# Further split data into training, validation, and testing sets if needed.