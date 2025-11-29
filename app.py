import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# Load model + dataset
# ============================================
model = joblib.load("bank_pipeline.pkl")
df = pd.read_csv("bank_full.csv", sep=";")

st.set_page_config(
    page_title="Bank Marketing Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================
# Title
# ============================================
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        📊 Bank Marketing Prediction Dashboard
    </h1>
    <h4 style='text-align: center; color: gray;'>
        Predict whether a customer will subscribe to a term deposit
    </h4>
    """,
    unsafe_allow_html=True
)

st.write(" ")

# ============================================
# Sidebar Inputs
# ============================================
st.sidebar.header("🔧 Input Customer Information")

age = st.sidebar.slider("Age", 18, 95, 30)
job = st.sidebar.selectbox("Job", df["job"].unique())
marital = st.sidebar.selectbox("Marital Status", df["marital"].unique())
education = st.sidebar.selectbox("Education", df["education"].unique())
default = st.sidebar.selectbox("Default?", df["default"].unique())
balance = st.sidebar.number_input("Account Balance", 0, 100000, 1000)
housing = st.sidebar.selectbox("Housing Loan?", df["housing"].unique())
loan = st.sidebar.selectbox("Personal Loan?", df["loan"].unique())
contact = st.sidebar.selectbox("Contact Type", df["contact"].unique())
day = st.sidebar.slider("Last Contact Day", 1, 31, 15)
month = st.sidebar.selectbox("Contact Month", df["month"].unique())
duration = st.sidebar.number_input("Call Duration (sec)", 0, 5000, 120)
campaign = st.sidebar.slider("Campaign Calls", 1, 50, 1)
pdays = st.sidebar.number_input("Days Passed After Last Contact", -1, 1000, 100)
previous = st.sidebar.slider("Previous Contacts", 0, 50, 0)
poutcome = st.sidebar.selectbox("Previous Outcome", df["poutcome"].unique())

# ============================================
# Prepare Input for Model
# ============================================
input_data = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
}])

# ============================================
# Main Dashboard Layout
# ============================================
left, right = st.columns([1.3, 1])

# ============================================
# Charts: Data Insights
# ============================================
with left:
    st.subheader("📈 Data Insights & Distributions")

    # Age distribution chart
    fig1, ax1 = plt.subplots()
    sns.histplot(df['age'], bins=30, kde=True, ax=ax1)
    ax1.set_title("Age Distribution")
    st.pyplot(fig1)

    # Balance distribution
    fig2, ax2 = plt.subplots()
    sns.histplot(df['balance'], bins=40, kde=True, color="orange", ax=ax2)
    ax2.set_title("Balance Distribution")
    st.pyplot(fig2)

    # Correlation heatmap
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    numeric_df = df.select_dtypes(include='number')
    sns.heatmap(numeric_df.corr(), cmap="viridis", annot=False, ax=ax3)
    ax3.set_title("Feature Correlation Heatmap")
    st.pyplot(fig3)

# ============================================
# Prediction Card
# ============================================
with right:
    st.subheader("🤖 Prediction")

    if st.button("Predict Subscription"):
        try:
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            color = "green" if pred == 1 else "red"
            label = "YES – Customer Likely to Subscribe" if pred == 1 else "NO – Customer Not subscribed"
            
            st.markdown(
                f"""
                <div style='padding:20px; border-radius:10px; background-color:#f0f0f0; text-align:center;'>
                    <h2 style='color:{color};'>{label}</h2>
                    <h3 style='color:black;'>Probability: {prob:.2f}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")

