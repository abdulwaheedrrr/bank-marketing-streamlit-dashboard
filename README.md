📊 Bank Marketing Prediction Dashboard

A Streamlit-based interactive dashboard for predicting whether a bank customer will subscribe to a term deposit. This dashboard not only predicts customer behavior using a tuned Decision Tree model, but also provides visual insights on the dataset.

🔹 Features

Interactive Prediction Form: Input customer information and get a real-time subscription prediction.

Probability Score: Displays the likelihood of subscription.

Data Visualizations:

Age distribution

Account balance distribution

Feature correlation heatmap

User-Friendly UI: Modern, clean, and intuitive interface.

Tuned Decision Tree Model: Includes hyperparameter tuning for better accuracy.

Easy Deployment: Works locally with Streamlit.

📂 Project Structure
bank-marketing-dashboard/
│
├─ app.py                   # Main Streamlit app
├─ bank_pipeline.pkl        # Trained Decision Tree model
├─ bank_full.csv            # Bank marketing dataset
├─ README.md                # Project documentation
└─ requirements.txt         # Python dependencies

🛠 Installation

Clone the repository:

git clone https://github.com/your-username/bank-marketing-dashboard.git
cd bank-marketing-dashboard


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows


Install dependencies:

pip install -r requirements.txt


(If you don’t have requirements.txt, install manually:)

pip install streamlit pandas matplotlib seaborn scikit-learn joblib

🚀 Run the App
streamlit run app.py


Open your browser at the URL shown in the terminal (usually http://localhost:8501)

Interact with the dashboard and make predictions.

🧠 How it Works

Load Model & Data: Loads the trained Decision Tree pipeline (bank_pipeline.pkl) and dataset (bank_full.csv).

User Input: Users provide customer details via the sidebar form.

Prediction: The model predicts whether the customer will subscribe, along with the probability score.

Visualization: Provides charts for data insights and feature correlations.

Styled Results: Prediction result and probability are displayed in a colorful card.
