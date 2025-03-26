import streamlit as st
import pandas as pd
import joblib
from google import genai
import speech_recognition as sr

BACKGROUND_COLOR = "#FFFFFF"
PRIMARY_COLOR = "#CD1F26"  
GOLD_COLOR = "#FFD700"

st.set_page_config(page_title="Banking Chatbot", page_icon="💬", layout="wide")

image_path = "./botIcon.png"
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
# st.image(image_path, width=200)
# st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {BACKGROUND_COLOR} !important;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: {PRIMARY_COLOR};
        }}
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: 2px solid {GOLD_COLOR}; /* Gold Outline */
            border-radius: 10px;
            font-size: 30px;
            font-weight: bold;
            padding: 12px;
        }}
        .stButton>button:hover {{
            background-color: white;
            color: {PRIMARY_COLOR};
            border: 2px solid {GOLD_COLOR}; /* Maintain Gold Outline */
        }}
    </style>
    """,
    unsafe_allow_html=True,
)
# image_path = "./botIcon.png" 
st.markdown(f"<h2 style='color:{PRIMARY_COLOR}; text-align: center;'>💬 NOVEL SMARTBOT</h1>", unsafe_allow_html=True)
st.markdown(f"<h5 style='text-align: center;'> Personalized, Secure, Adaptive Banking for Every Generation</h5>", unsafe_allow_html=True)

recommendations_df = pd.read_csv("./data/Recommendations.csv", dtype={"Customer_id": str})
customer_df = pd.read_excel("./data/Updated_Novel_Team_Dataset.xlsx", sheet_name='Customer Profile (Individual)')
xgb_model = joblib.load("xgb_model.pkl")

customer_df["Customer_id"] = customer_df["Customer_id"]
customer_df["Co-Borrower"] = customer_df["Co-Borrower"]
customer_df.dropna(subset=["Customer_id"], inplace=True)

client = genai.Client(api_key="AIzaSyDRbVUg2zvMwK5S3QpBtou5h8JFXOcIDfs")

def display_message(message):
    if "❌" in message:
        st.error(message)
    elif "⚠️" in message:
        st.warning(message)
    elif "✅" in message:
        st.success(message)
    elif "🚨" in message:
        st.error(message)
    else:
        st.info(message)

def get_recommendations(customer_id):
    user_data = recommendations_df[recommendations_df["Customer_id"].str.upper() == customer_id]
    return user_data["Recommendations"].values[0] if not user_data.empty else "Please enter valid customer ID"

def chatbot_response(customer_id):
    recommendation = get_recommendations(customer_id)
    if recommendation == "No recommendations available.":
        return "❌ No recommendations available."
    
    prompt = f"Explain why '{recommendation}' is recommended."
    response = client.models.generate_content(model="gemini-1.5-pro", contents=prompt)
    return response.text if response and hasattr(response, "text") else "I'm sorry, I couldn't process your request."

def casual_chat(user_message):
    if "adhd" in user_message.lower() or "neurodivergent" in user_message.lower():
        stress_level = "Moderate"
        return f"""🧠 **Financial Strategy for Neurodivergence (Stress Level: {stress_level})**:

        🕒 **Smart Reminders**:
        - Bill due alerts **3 days before deadline**.
        - **Daily spending check-ins**.
        - **Savings challenge** to improve habits.

        💡 **Would you like to auto-schedule payments?**"""

    prompt = f"You are a banking assistant. Respond conversationally to: {user_message}"
    response = client.models.generate_content(model="gemini-1.5-pro", contents=prompt)
    return response.text if response and hasattr(response, "text") else "I'm sorry, I couldn't process your request."

def check_loan_eligibility(customer_id):
    customer = customer_df[customer_df['Customer_id'] == customer_id].copy()
    if customer.empty:
        return 'Customer not found.'

    features = ["Age", "Income per year (in dollars)", "Existing Loans", "Debt-to-Income Ratio", "Co-Borrower"]
    customer[features] = customer[features].apply(pd.to_numeric, errors="coerce").fillna(customer_df[features].mean())

    if "Co-Borrower" in customer.columns:
        customer["Co-Borrower"] = customer["Co-Borrower"].astype("category").cat.codes

    X = customer[features].values
    prediction = xgb_model.predict(X)[0]

    return f'Customer {customer_id} is eligible for a combined auto + home loan.' if prediction == 1 else f'Customer {customer_id} is not eligible for a combined loan.'

def detect_elderly_fraud(customer_id):
    customer = customer_df[customer_df["Customer_id"] == customer_id]
    if customer.empty:
        return "Customer not found."

    return f"🚨 ALERT! Suspicious activity detected for elderly customer {customer_id}." if customer["Age"].values[0] > 65 and customer["Suspicious Transactions"].values[0] > 5 else f"✅ No unusual transactions detected for {customer_id}."

def predict_wealth_transfer(customer_id):
    customer = customer_df[customer_df["Customer_id"] == customer_id]
    return f"💰 Predicted asset growth for **{customer_id}**: **${customer['Income per year (in dollars)'].values[0] * 0.1} annually**." if not customer.empty else "Customer not found."

def financial_coaching(customer_id):
    customer = customer_df[customer_df["Customer_id"] == customer_id]
    return f"📊 Smart spending reminder for **{customer_id}**: Avoid unnecessary credit card debt." if not customer.empty else "Customer not found."
st.markdown("")

customer_id = st.text_input(" 🔑 **Enter Customer ID**").strip().upper()

if st.button("🚀 Get Recommendation"):
    if customer_id:
        st.session_state["recommendation"] = get_recommendations(customer_id)

if "recommendation" in st.session_state and st.session_state["recommendation"]:
    display_message(f"🤖 Recommendation: **{st.session_state['recommendation']}**")

    if st.button("📖 Explain More"):
        st.session_state["detailed_response"] = chatbot_response(customer_id)

    if "detailed_response" in st.session_state and st.session_state["detailed_response"]:
        display_message(st.session_state["detailed_response"])

st.subheader("📊 Other Financial Insights:")

col1, col2 = st.columns(2)
with col1:
    if st.button("🏦 Check Loan Eligibility"):
        st.session_state["loan_response"] = check_loan_eligibility(customer_id)
    if "loan_response" in st.session_state:
        display_message(st.session_state["loan_response"])

    if st.button("🛡 Fraud Detection"):
        st.session_state["fraud_response"] = detect_elderly_fraud(customer_id)
    if "fraud_response" in st.session_state:
        display_message(st.session_state["fraud_response"])

with col2:
    if st.button("💰 Wealth Transfer Prediction"):
        st.session_state["wealth_response"] = predict_wealth_transfer(customer_id)
    if "wealth_response" in st.session_state:
        display_message(st.session_state["wealth_response"])

    if st.button("📊 Financial Coaching Advice"):
        st.session_state["coaching_response"] = financial_coaching(customer_id)
    if "coaching_response" in st.session_state:
        display_message(st.session_state["coaching_response"])

st.subheader("💬 Chat with Novel SmartBot")

# Ensure session state is initialized
if "user_message" not in st.session_state:
    st.session_state["user_message"] = ""

# Layout for text input and mic button
col1, col2 = st.columns([5, 1])

with col1:
    user_message = st.text_input("**Type your message here and press Enter:**", value=st.session_state["user_message"], key="user_input")

with col2:
    st.write("")  
    st.write("") 
    if st.button("🎙️", help="Click and speak your message"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak now!")
            try:
                audio = recognizer.listen(source, timeout=5)
                spoken_text = recognizer.recognize_google(audio)
                st.success(f"Recognized: {spoken_text}")
                st.session_state["user_message"] = spoken_text  # Update session state
                st.rerun()  # ✅ Force UI update (NEW)
            except sr.UnknownValueError:
                st.error("Sorry, could not understand the speech.")
            except sr.RequestError:
                st.error("Could not connect to the speech recognition service.")
if user_message:
    response = casual_chat(user_message)
    st.text_area("🤖 Chatbot:", value=response, height=150)

