
import streamlit as st
import pandas as pd
import sqlite3
import requests
from datetime import datetime
from pathlib import Path

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "fraudguard.db"
DATASET_PATH = APP_DIR / "transactions.csv"

API_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_URL}/predict"

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. PROFESSIONAL LIGHT THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #F4F8FC;
    color: #26364A;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #183B56 !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown {
    color: #40566D;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EAF3FB 0%, #F4F8FC 100%);
    border-right: 1px solid #D8E5F0;
}

section[data-testid="stSidebar"] * {
    color: #183B56;
}

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background-color: #176B87 !important;
    color: white !important;
    border: 1px solid #176B87 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1rem !important;
}

.stButton > button *,
.stFormSubmitButton > button *,
.stDownloadButton > button * {
    color: white !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #10536D !important;
    border-color: #10536D !important;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: white;
    border-radius: 8px;
    color: #26364A !important;
}

.stTextInput label,
.stNumberInput label,
.stSelectbox label {
    color: #183B56 !important;
    font-weight: 600 !important;
}

.info-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #E0EAF3;
    box-shadow: 0 4px 14px rgba(38, 75, 110, 0.06);
    min-height: 125px;
    margin-bottom: 12px;
}

.info-card h3 {
    font-size: 16px;
    color: #60758A !important;
    margin-bottom: 10px;
}

.info-card h2 {
    font-size: 28px;
    color: #183B56 !important;
    margin: 0;
}

.page-banner {
    background: linear-gradient(110deg, #E1F2F7, #EAF1FC);
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #D5E7F1;
    margin-bottom: 22px;
}

.page-banner h1 {
    margin-bottom: 5px;
    color: #183B56 !important;
}

.page-banner p {
    margin-bottom: 0;
    color: #557087;
}

.section-box {
    background: white;
    border: 1px solid #E0EAF3;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
}

.small-note {
    color: #71869A;
    font-size: 13px;
}

hr {
    border-color: #DCE7F0;
}

footer {
    visibility: hidden;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E0EAF3;
    border-radius: 10px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E0EAF3;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #E0EAF3;
    border-radius: 14px;
    padding: 20px;
}

div[data-baseweb="select"] * {
    color: #26364A !important;
}

.stDownloadButton > button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. DATABASE SETUP
# =====================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():

    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id TEXT,
            receiver_id TEXT,
            amount REAL,
            transaction_type TEXT,
            prediction TEXT,
            fraud_probability REAL,
            date_time TEXT,
            step INTEGER,
            oldbalanceOrg REAL,
            newbalanceOrig REAL,
            oldbalanceDest REAL,
            newbalanceDest REAL
        )
    """)

    existing_columns = {
        row[1] for row in connection.execute(
            "PRAGMA table_info(transactions)"
        ).fetchall()
    }

    required_columns = {
        "sender_id": "TEXT",
        "receiver_id": "TEXT",
        "amount": "REAL",
        "transaction_type": "TEXT",
        "prediction": "TEXT",
        "fraud_probability": "REAL",
        "date_time": "TEXT",
        "step": "INTEGER",
        "oldbalanceOrg": "REAL",
        "newbalanceOrig": "REAL",
        "oldbalanceDest": "REAL",
        "newbalanceDest": "REAL"
    }

    for column, column_type in required_columns.items():

        if column not in existing_columns:
            connection.execute(
                f"ALTER TABLE transactions ADD COLUMN {column} {column_type}"
            )

    connection.commit()
    connection.close()


initialize_database()


# =====================================================
# 4. DATABASE FUNCTIONS
# =====================================================

def save_transaction(
    sender_id,
    receiver_id,
    amount,
    transaction_type,
    prediction,
    fraud_probability
):

    connection = get_connection()

    connection.execute("""
        INSERT INTO transactions (
            sender_id,
            receiver_id,
            amount,
            transaction_type,
            prediction,
            fraud_probability,
            date_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        sender_id,
        receiver_id,
        amount,
        transaction_type,
        prediction,
        fraud_probability,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


def get_history():

    connection = get_connection()

    data = pd.read_sql_query(
        "SELECT * FROM transactions ORDER BY id DESC",
        connection
    )

    connection.close()

    return data


# =====================================================
# 5. SESSION STATE
# =====================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Home"


# =====================================================
# 6. COMMON UI FUNCTIONS
# =====================================================

def show_banner(title, description):

    st.markdown(f"""
        <div class="page-banner">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
    """, unsafe_allow_html=True)


def show_card(title, value):

    st.markdown(f"""
        <div class="info-card">
            <h3>{title}</h3>
            <h2>{value}</h2>
        </div>
    """, unsafe_allow_html=True)


# =====================================================
# 7. LOGIN PAGE
# =====================================================

if not st.session_state.logged_in:

    st.markdown("<br><br>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.3, 1])

    with center:

        st.markdown("""
            <div class="page-banner" style="text-align:center;">
                <h1>🛡️ FraudGuard AI</h1>
                <p>AI-Powered Online Transaction Fraud Detection</p>
            </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):

            st.subheader("Welcome Back")

            st.write("Login to access your FraudGuard dashboard.")

            username = st.text_input(
                "Username",
                placeholder="Enter username"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password"
            )

            if st.button("Login", use_container_width=True):

                if username == "admin" and password == "fraud123":

                    st.session_state.logged_in = True
                    st.session_state.page = "Home"
                    st.rerun()

                else:
                    st.error("Invalid username or password.")

            st.markdown(
                '<p class="small-note">Demo Login: admin | Password: fraud123</p>',
                unsafe_allow_html=True
            )

    st.stop()


# =====================================================
# 8. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("""
        <div style="text-align:center; padding:12px 0;">
            <h1 style="font-size:27px;">🛡️ FraudGuard</h1>
            <p style="font-size:13px;">AI Security System</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### MAIN MENU")

    pages = [
        ("🏠", "Home"),
        ("📊", "Dashboard"),
        ("🔍", "Detect Fraud"),
        ("📋", "Transaction History"),
        ("📈", "Analytics"),
        ("ℹ️", "About Project")
    ]

    for icon, page_name in pages:

        if st.button(
            f"{icon}  {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True
        ):

            st.session_state.page = page_name
            st.rerun()

    st.divider()

    st.caption("FraudGuard AI | Final Year Project")

    if st.button("Logout", use_container_width=True):

        st.session_state.logged_in = False
        st.session_state.page = "Home"
        st.rerun()


# =====================================================
# 9. LOAD CURRENT PAGE DATA
# =====================================================

page = st.session_state.page
history = get_history()

total_transactions = len(history)

if total_transactions > 0:

    fraud_count = int(
        (history["prediction"] == "Fraudulent").sum()
    )

    normal_count = int(
        (history["prediction"] == "Normal").sum()
    )

else:

    fraud_count = 0
    normal_count = 0


# =====================================================
# 10. HOME PAGE
# =====================================================

if page == "Home":

    show_banner(
        "Welcome to FraudGuard AI",
        "Smart and intelligent online transaction fraud detection system."
    )

    st.markdown("### Protecting Digital Transactions with AI")

    st.write("""
    FraudGuard AI is a machine learning based application designed
    to identify suspicious online financial transactions.

    It uses a trained Random Forest model to classify transactions
    as Normal or Fraudulent.
    """)

    st.markdown("### What You Can Do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>🔍 Fraud Detection</h3>
                <p>Analyze transaction details and detect suspicious activity.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="info-card">
                <h3>📊 Smart Dashboard</h3>
                <p>View transaction statistics and detection results.</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="info-card">
                <h3>📈 Analytics</h3>
                <p>Explore transaction trends through charts and reports.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### Project Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        show_card("Total Transactions", total_transactions)

    with c2:
        show_card("Normal Transactions", normal_count)

    with c3:
        show_card("Fraudulent Transactions", fraud_count)

    st.markdown("---")

    if st.button("Start Fraud Detection →", use_container_width=True):

        st.session_state.page = "Detect Fraud"
        st.rerun()


# =====================================================
# 11. DASHBOARD PAGE
# =====================================================

elif page == "Dashboard":

    show_banner(
        "Dashboard",
        "Overview of transaction activity and fraud detection results."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_card("Total Transactions", total_transactions)

    with col2:
        show_card("Normal Transactions", normal_count)

    with col3:
        show_card("Fraudulent Transactions", fraud_count)

    with col4:

        fraud_rate = (
            (fraud_count / total_transactions) * 100
            if total_transactions > 0 else 0
        )

        show_card("Fraud Rate", f"{fraud_rate:.2f}%")

    st.markdown("### Recent Transactions")

    if not history.empty:

        st.dataframe(
            history.head(10),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No transactions available yet.")

    st.markdown("### Dataset Snapshot")

    if DATASET_PATH.exists():

        try:

            dataset = pd.read_csv(DATASET_PATH, nrows=10000)

            st.write(
                f"Showing a sample of {len(dataset)} rows "
                "from the project dataset."
            )

            st.dataframe(
                dataset.head(10),
                use_container_width=True
            )

        except Exception as error:
            st.warning(f"Dataset could not be loaded: {error}")

    else:
        st.info("transactions.csv was not found in the project folder.")


# =====================================================
# 12. DETECT FRAUD PAGE - FOUR INPUT FIELDS
# =====================================================

elif page == "Detect Fraud":

    show_banner(
        "Detect Fraud",
        "Enter four simple transaction details to check for suspicious activity."
    )

    st.markdown("### Transaction Information")

    st.info(
        "Enter the sender, receiver, transaction type and amount. "
        "FraudGuard AI will analyze the transaction."
    )

    with st.form("fraud_detection_form"):

        col1, col2 = st.columns(2)

        with col1:

            sender_id = st.text_input(
                "Sender ID",
                placeholder="Example: C123456"
            )

            transaction_type = st.selectbox(
                "Transaction Type",
                [
                    "PAYMENT",
                    "CASH_IN",
                    "CASH_OUT",
                    "DEBIT",
                    "TRANSFER"
                ]
            )

        with col2:

            receiver_id = st.text_input(
                "Receiver ID",
                placeholder="Example: M987654"
            )

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.01,
                value=1000.0,
                step=100.0,
                format="%.2f"
            )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            use_container_width=True
        )

    if submitted:

        if not sender_id.strip() or not receiver_id.strip():

            st.warning("Please enter both Sender ID and Receiver ID.")

        elif sender_id.strip() == receiver_id.strip():

            st.warning("Sender ID and Receiver ID cannot be the same.")

        elif amount <= 0:

            st.warning("Please enter an amount greater than zero.")

        else:

            # Only four fields are sent to FastAPI.
            # Sender and receiver IDs are used for records.
            # The model should use amount and transaction type.

            payload = {
                "sender_id": sender_id.strip(),
                "receiver_id": receiver_id.strip(),
                "transaction_type": transaction_type,
                "amount": float(amount)
            }

            try:

                with st.spinner("AI is analyzing the transaction..."):

                    response = requests.post(
                        PREDICT_URL,
                        json=payload,
                        timeout=30
                    )

                if response.status_code == 200:

                    result_data = response.json()

                    prediction = result_data.get("prediction")

                    fraud_probability = float(
                        result_data.get("fraud_probability", 0)
                    )

                    if prediction not in ["Normal", "Fraudulent"]:

                        st.error(
                            "The API returned an unexpected prediction."
                        )

                    elif not 0 <= fraud_probability <= 100:

                        st.error(
                            "The API returned an invalid fraud probability."
                        )

                    else:

                        save_transaction(
                            sender_id.strip(),
                            receiver_id.strip(),
                            float(amount),
                            transaction_type,
                            prediction,
                            fraud_probability
                        )

                        st.markdown("---")
                        st.markdown("### Analysis Result")

                        if prediction == "Fraudulent":

                            st.error(
                                "⚠️ Potentially Fraudulent Transaction Detected"
                            )

                            st.warning(
                                "This transaction has been flagged by the AI model. "
                                "Please review it carefully."
                            )

                        else:

                            st.success(
                                "✅ No Fraud Detected - Normal Transaction"
                            )

                            st.info(
                                "The AI model did not identify this transaction "
                                "as fraudulent."
                            )

                        result_col1, result_col2 = st.columns(2)

                        with result_col1:
                            show_card("Prediction", prediction)

                        with result_col2:
                            show_card(
                                "Fraud Probability",
                                f"{fraud_probability:.2f}%"
                            )

                        st.markdown("### Transaction Details")

                        details = pd.DataFrame({
                            "Field": [
                                "Sender ID",
                                "Receiver ID",
                                "Transaction Type",
                                "Transaction Amount"
                            ],
                            "Value": [
                                sender_id.strip(),
                                receiver_id.strip(),
                                transaction_type,
                                f"{amount:,.2f}"
                            ]
                        })

                        st.table(details)

                        st.info(
                            "Transaction result has been saved "
                            "in Transaction History."
                        )

                        st.caption(
                            "This is an AI-based prediction, not a guarantee "
                            "that a transaction is safe or fraudulent."
                        )

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                    try:
                        error_details = response.json()
                        st.code(str(error_details))
                    except ValueError:
                        st.write(response.text)

                    st.warning(
                        "If the API reports missing fields, api.py must also "
                        "be updated to accept the four-field request."
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to FastAPI. "
                    "Make sure the API server is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The API request timed out. Please try again."
                )

            except requests.exceptions.RequestException as error:

                st.error(f"API request failed: {error}")

            except (ValueError, TypeError):

                st.error(
                    "The API returned an invalid prediction or probability."
                )

            except Exception as error:

                st.error(f"An unexpected error occurred: {error}")


# =====================================================
# 13. TRANSACTION HISTORY PAGE
# =====================================================

elif page == "Transaction History":

    show_banner(
        "Transaction History",
        "View, search and download previously analyzed transactions."
    )

    if history.empty:

        st.info("No transaction history available yet.")

    else:

        search = st.text_input(
            "Search by Sender ID or Receiver ID",
            placeholder="Enter an ID to search"
        )

        filtered_history = history.copy()

        if search.strip():

            filtered_history = filtered_history[
                filtered_history["sender_id"].astype(str).str.contains(
                    search,
                    case=False,
                    na=False
                )
                |
                filtered_history["receiver_id"].astype(str).str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        filter_type = st.selectbox(
            "Filter by Prediction",
            ["All", "Normal", "Fraudulent"]
        )

        if filter_type != "All":

            filtered_history = filtered_history[
                filtered_history["prediction"] == filter_type
            ]

        st.write(
            f"Showing {len(filtered_history)} transaction(s)."
        )

        st.dataframe(
            filtered_history,
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered_history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Transaction History (CSV)",
            data=csv_data,
            file_name="fraudguard_transaction_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# =====================================================
# 14. ANALYTICS PAGE
# =====================================================

elif page == "Analytics":

    show_banner(
        "Analytics",
        "Visual analysis of the transactions processed by FraudGuard AI."
    )

    if history.empty:

        st.info(
            "Analytics will appear after you analyze some transactions."
        )

    else:

        total = len(history)

        fraud = int(
            (history["prediction"] == "Fraudulent").sum()
        )

        normal = int(
            (history["prediction"] == "Normal").sum()
        )

        fraud_percentage = (fraud / total) * 100

        c1, c2, c3 = st.columns(3)

        with c1:
            show_card("Total Transactions", total)

        with c2:
            show_card("Fraudulent", fraud)

        with c3:
            show_card("Normal", normal)

        st.markdown("### Transaction Prediction Distribution")

        distribution = pd.DataFrame({
            "Prediction": ["Normal", "Fraudulent"],
            "Count": [normal, fraud]
        })

        st.bar_chart(
            distribution.set_index("Prediction")
        )

        st.markdown("### Transactions by Type")

        type_chart = (
            history.groupby("transaction_type")
            .size()
            .reset_index(name="Count")
        )

        st.bar_chart(
            type_chart.set_index("transaction_type")
        )

        st.markdown("### Fraud Percentage")

        st.progress(
            min(max(fraud_percentage / 100, 0.0), 1.0)
        )

        st.write(
            f"Fraudulent transactions: {fraud_percentage:.2f}%"
        )

        st.caption(
            "These charts are based on transactions analyzed and saved "
            "in this application. They are not model accuracy metrics."
        )


# =====================================================
# 15. ABOUT PROJECT PAGE
# =====================================================

elif page == "About Project":

    show_banner(
        "About FraudGuard AI",
        "Project information, machine learning approach and technologies."
    )

    st.markdown("### Project Information")

    st.markdown("""
    <div class="section-box">
        <h3>🛡️ AI Online Transaction Fraud Detector</h3>
        <p>
            FraudGuard AI is a machine learning based application
            that analyzes online transaction information and predicts
            whether a transaction is Normal or Fraudulent.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Student Details")

        st.write("**Student Name:** Sana Jameel")
        st.write("**Roll Number:** 24-BSIT-87")
        st.write("**Department:** Information Technology")
        st.write("**University:** SBBU-SBA")

    with col2:

        st.markdown("### Project Details")

        st.write("**Project Category:** Data Science")
        st.write("**Machine Learning:** Random Forest Classifier")
        st.write("**Learning Type:** Supervised Learning")
        st.write("**Problem Type:** Classification")

    st.markdown("---")

    st.markdown("### Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("""
            <div class="info-card">
                <h3>🐍 Python</h3>
                <p>Programming language used for the project.</p>
            </div>
        """, unsafe_allow_html=True)

    with tech2:
        st.markdown("""
            <div class="info-card">
                <h3>🤖 Machine Learning</h3>
                <p>Random Forest model for fraud prediction.</p>
            </div>
        """, unsafe_allow_html=True)

    with tech3:
        st.markdown("""
            <div class="info-card">
                <h3>🌐 Streamlit & FastAPI</h3>
                <p>User interface and prediction API.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <h3>How Fraud Detection Works</h3>
        <ol>
            <li>User enters sender ID, receiver ID, transaction type and amount.</li>
            <li>Streamlit sends the transaction details to the FastAPI server.</li>
            <li>The trained Random Forest model analyzes the available transaction features.</li>
            <li>The model returns a Normal or Fraudulent prediction.</li>
            <li>The result is saved in transaction history.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "FraudGuard AI is an academic demonstration project. "
        "Its predictions should not be treated as a production banking decision."
    )


# =====================================================
# 16. FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:10px;">
    <p style="color:#71869A; font-size:13px;">
        🛡️ FraudGuard AI | AI Online Transaction Fraud Detector
    </p>
    <p style="color:#91A2B2; font-size:12px;">
        Smart Security • Intelligent Detection • Digital Protection
    </p>
</div>
""", unsafe_allow_html=True)
