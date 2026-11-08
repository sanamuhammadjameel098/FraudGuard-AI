
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
import pandas as pd
import sqlite3
import joblib

# =====================================================
# 1. PROJECT PATHS AND MODEL
# =====================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "fraud_detection_model.pkl"
DB_PATH = APP_DIR / "fraudguard.db"

model = joblib.load(MODEL_PATH)

EXPECTED_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]

# Provisional threshold for demonstration
FRAUD_THRESHOLD = 0.20

if list(model.feature_names_in_) != EXPECTED_FEATURES:
    raise RuntimeError(
        "Model features do not match API features."
    )

# =====================================================
# 2. CREATE FASTAPI APP
# =====================================================

app = FastAPI(
    title="FraudGuard AI API",
    description="Online Transaction Fraud Detection API",
    version="4.2"
)

# =====================================================
# 3. INITIALIZE DATABASE AND DEMO ACCOUNTS
# =====================================================

def initialize_accounts():

    conn = sqlite3.connect(DB_PATH)

    try:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                balance REAL NOT NULL DEFAULT 0
            )
        """)

        demo_accounts = [
            ("C123456", 1000000.0),
            ("M987654", 1000000.0),
            ("C111222", 1000000.0),
            ("M333444", 1000000.0)
        ]

        conn.executemany("""
            INSERT OR IGNORE INTO accounts (
                account_id,
                balance
            )
            VALUES (?, ?)
        """, demo_accounts)

        conn.commit()

    finally:
        conn.close()


initialize_accounts()

# =====================================================
# 4. TRANSACTION INPUT
# =====================================================

class Transaction(BaseModel):

    sender_id: str = Field(min_length=1)
    receiver_id: str = Field(min_length=1)

    transaction_type: Literal[
        "PAYMENT",
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "TRANSFER"
    ]

    amount: float = Field(gt=0)

# =====================================================
# 5. GET ACCOUNT BALANCE
# =====================================================

def get_account_balance(account_id):

    conn = sqlite3.connect(DB_PATH)

    try:

        cursor = conn.cursor()

        cursor.execute(
            "SELECT balance FROM accounts WHERE account_id = ?",
            (account_id,)
        )

        result = cursor.fetchone()

        if result is None:
            return None

        return float(result[0])

    finally:
        conn.close()

# =====================================================
# 6. HOME ENDPOINT
# =====================================================

@app.get("/")
def home():

    return {
        "message": "FraudGuard AI API is running",
        "model": "Random Forest Classifier",
        "features": 10,
        "fraud_threshold": FRAUD_THRESHOLD * 100,
        "input_fields": [
            "sender_id",
            "receiver_id",
            "transaction_type",
            "amount"
        ]
    }

# =====================================================
# 7. FRAUD PREDICTION
# =====================================================

@app.post("/predict")
def predict_fraud(data: Transaction):

    try:

        # Check sender and receiver IDs

        if data.sender_id == data.receiver_id:

            raise HTTPException(
                status_code=400,
                detail="Sender and receiver IDs must be different."
            )

        # Get account balances

        sender_balance = get_account_balance(
            data.sender_id
        )

        receiver_balance = get_account_balance(
            data.receiver_id
        )

        if sender_balance is None:

            raise HTTPException(
                status_code=404,
                detail="Sender account not found."
            )

        if receiver_balance is None:

            raise HTTPException(
                status_code=404,
                detail="Receiver account not found."
            )

        # Check available balance

        if data.transaction_type in [
            "TRANSFER",
            "CASH_OUT",
            "PAYMENT",
            "DEBIT"
        ]:

            if data.amount > sender_balance:

                raise HTTPException(
                    status_code=400,
                    detail="Insufficient sender balance."
                )

        # Calculate transaction balances

        oldbalanceOrg = sender_balance
        newbalanceOrig = sender_balance - data.amount

        oldbalanceDest = receiver_balance
        newbalanceDest = receiver_balance + data.amount

        # Prepare model features

        transaction = pd.DataFrame([{

            "amount": data.amount,

            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,

            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,

            "type_CASH_IN": int(
                data.transaction_type == "CASH_IN"
            ),

            "type_CASH_OUT": int(
                data.transaction_type == "CASH_OUT"
            ),

            "type_DEBIT": int(
                data.transaction_type == "DEBIT"
            ),

            "type_PAYMENT": int(
                data.transaction_type == "PAYMENT"
            ),

            "type_TRANSFER": int(
                data.transaction_type == "TRANSFER"
            )

        }])

        transaction = transaction.reindex(
            columns=EXPECTED_FEATURES,
            fill_value=0
        )

        # Get fraud probability

        probabilities = model.predict_proba(
            transaction
        )[0]

        fraud_index = list(model.classes_).index(1)

        fraud_probability = float(
            probabilities[fraud_index]
        )

        # Apply fraud threshold

        if fraud_probability >= FRAUD_THRESHOLD:

            result = "Fraudulent"

        else:

            result = "Normal"

        # Print prediction details

        print("--------------------------------")
        print("Sender ID:", data.sender_id)
        print("Receiver ID:", data.receiver_id)
        print("Transaction Type:", data.transaction_type)
        print("Amount:", data.amount)
        print("Sender Balance:", sender_balance)
        print("Receiver Balance:", receiver_balance)

        print(
            "Fraud Probability:",
            round(fraud_probability * 100, 2),
            "%"
        )

        print(
            "Decision Threshold:",
            FRAUD_THRESHOLD * 100,
            "%"
        )

        print("Prediction:", result)
        print("--------------------------------")

        # Return prediction

        return {

            "sender_id": data.sender_id,
            "receiver_id": data.receiver_id,

            "transaction_type": data.transaction_type,
            "amount": data.amount,

            "sender_balance": sender_balance,
            "receiver_balance": receiver_balance,

            "prediction": result,

            "fraud_probability": round(
                fraud_probability * 100,
                2
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        print("Prediction Error:", str(error))

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
