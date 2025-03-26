import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "./data/Loan_Eligibility_Dataset_Extended.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Drop unnecessary columns
df = df.drop(columns=["Customer_id", "Gender", "Location", "Education", "Occupation", "Browsing Behavior"])

# Convert "Loan Eligibility" to numerical labels
label_encoder = LabelEncoder()
df["Loan Eligibility"] = label_encoder.fit_transform(df["Loan Eligibility"])  # Encoding categories as 0,1,2

# Handle missing values
df["Co-Borrower"].fillna("None", inplace=True)  # Fill missing Co-Borrowers
df["Co-Borrower"] = df["Co-Borrower"].astype("category").cat.codes  # Convert to numeric

# Define features and target variable
features = ["Age", "Income per year (in dollars)", "Existing Loans", "Debt-to-Income Ratio", "Co-Borrower"]
target = "Loan Eligibility"

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
xgb_model.fit(X_train, y_train)

# Save trained model
joblib.dump(xgb_model, "xgb_model.pkl")

# Evaluate the model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")
