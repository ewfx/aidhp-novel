import pandas as pd
from google import genai
import pandas as pd
import numpy as np
import joblib

recommendations_df = pd.read_csv("./data/Recommendations.csv", dtype={"Customer_id": str})

client = genai.Client(api_key="AIzaSyDRbVUg2zvMwK5S3QpBtou5h8JFXOcIDfs")

customer_df = pd.read_excel("./data/Updated_Novel_Team_Dataset.xlsx", sheet_name='Customer Profile (Individual)')



xgb_model = joblib.load("xgb_model.pkl")

file_path = "./data/Loan_Eligibility_Dataset_Extended.xlsx"
customer_df = pd.read_excel(file_path, sheet_name="Sheet1")


customer_df["Customer_id"] = customer_df["Customer_id"]
customer_df["Co-Borrower"] = customer_df["Co-Borrower"]

customer_df.dropna(subset=["Customer_id"], inplace=True)


def check_loan_eligibility(customer_id):
    """ Check if a customer is eligible for a loan using the trained XGBoost model. """

    customer = customer_df[customer_df['Customer_id'] == customer_id].copy()

    if customer.empty:
        return 'Customer not found.'

    features = ["Age", "Income per year (in dollars)", "Existing Loans", "Debt-to-Income Ratio", "Co-Borrower"]

    customer[features] = customer[features].apply(pd.to_numeric, errors="coerce")
    customer_df[features] = customer_df[features].apply(pd.to_numeric, errors="coerce")

    customer[features] = customer[features].fillna(customer_df[features].mean())

    if "Co-Borrower" in customer.columns:
        customer["Co-Borrower"] = customer["Co-Borrower"].astype("category").cat.codes

    X = customer[features].values  # Convert to numpy array
    prediction = xgb_model.predict(X)[0]

    if prediction == 1:
        return f'Customer {customer_id} is eligible for a combined auto + home loan.'
    else:
        return f'Customer {customer_id} is not eligible for a combined loan.'

def predict_wealth_transfer(customer_id):
    customer = customer_df[customer_df["Customer_id"] == customer_id]
    if customer.empty:
        return "Customer not found."

    wealth_growth = customer["Income per year (in dollars)"].values[0] * 0.1  # Example projection
    return f"Predicted asset growth for Customer {customer_id}: ${wealth_growth} annually."

def financial_coaching(customer_id):
    customer = customer_df[customer_df["Customer_id"] == customer_id]
    
    if customer.empty:
        return "Customer not found."
    
    if "Browsing Behavior" not in customer.columns:
        return "Error: No spending-related data found."

    spending_habits = customer["Browsing Behavior"].values[0]
    return f"Smart spending reminder for Customer {customer_id}: Consider monitoring {spending_habits}."

def detect_elderly_fraud(customer_id):
    customer = customer_df[customer_df["Customer_id"] == customer_id]
    if customer.empty:
        return "Customer not found."

    if customer["Age"].values[0] > 65 and customer["Suspicious Transactions"].values[0] > 5:
        return f"ALERT! Suspicious activity detected for elderly customer {customer_id}."
    return f"Customer {customer_id} has no unusual transactions."

def get_recommendations(customer_id):
    customer_id = customer_id.strip().upper()  # Normalize case
    user_data = recommendations_df[recommendations_df["Customer_id"].str.upper() == customer_id]
    if not user_data.empty:
        return user_data["Recommendations"].values[0]
    return "No recommendations available."

def chatbot_response(user_message, customer_id, detailed=False):
    recommendations = get_recommendations(customer_id)

    if not recommendations:
        return "No recommendations available."

    if not detailed:
        return recommendations  # One-line response

    prompt = f"Explain why '{recommendations}' is recommended for the user."
    response = client.models.generate_content(model="gemini-1.5-pro", contents = prompt)
    
    return response.text if response and hasattr(response, "text") else "I'm sorry, I couldn't process your request."


def start_chatbot():
    print("💬 Welcome to the Banking Chatbot! Type 'exit' to quit.")
    
    while True:
        customer_id = input("Enter your Customer ID (e.g., CUST2025A): ").strip().upper()
        if customer_id.startswith("CUST") and len(customer_id) > 5:
            break
        print("❌ Invalid Customer ID. Try again (Format: CUST2025A).")

    recommendation = get_recommendations(customer_id)
    print(f"🤖 Chatbot: We reccommend {recommendation}")

    while True:
        user_message = input("You: ").strip().lower()

        if user_message == "exit":
            print("👋 Goodbye!")
            break
        elif "loan eligibility" in user_message.lower():
            response = check_loan_eligibility(customer_id)
        
        elif "wealth transfer" in user_message.lower():
            response = predict_wealth_transfer(customer_id)
        
        elif "fraud detection" in user_message.lower():
            response = detect_elderly_fraud(customer_id)
        
        elif "financial coaching" in user_message.lower():
            response = financial_coaching(customer_id)
        
        elif user_message in ["elaborate", "can you explain", "tell me more", "why?"]:
            response = chatbot_response(recommendation, customer_id, detailed=True)
        else:
            response = chatbot_response(user_message, customer_id)

        print(f"🤖 Chatbot: {response}")

# Start chatbot
if __name__ == "__main__":
    start_chatbot()
