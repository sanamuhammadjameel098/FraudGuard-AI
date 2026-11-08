# FraudGuard AI – Online Transaction Fraud Detection System

## Project Overview
FraudGuard AI is a Machine Learning-based web application designed to identify potentially fraudulent online financial transactions.

The system uses a Random Forest Classifier to analyze transaction details and predict whether a transaction is Normal or Fraudulent.

## Objectives
- Detect suspicious online transactions.
- Apply Machine Learning for fraud classification.
- Display prediction results and fraud probability.
- Maintain transaction history.
- Provide an interactive dashboard.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit
- FastAPI
- SQLite
- Joblib

## Project Features
- Transaction fraud prediction
- Normal and fraudulent transaction classification
- Fraud probability display
- Transaction history
- Dashboard and analytics
- FastAPI backend
- Streamlit user interface

## Project Files
- `app.py` – Streamlit frontend
- `api.py` – FastAPI backend
- `fraud_detection_model.pkl` – Trained Machine Learning model
- `requirements.txt` – Required Python libraries

## Installation and Setup

1. Clone or download this repository.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Start the FastAPI backend:

```bash
uvicorn api:app --reload
```

4. Open another terminal and start the Streamlit application:

```bash
streamlit run app.py
```

5. Open the application in your browser:

`http://localhost:8501`

## Machine Learning Model
Algorithm: Random Forest Classifier

Problem Type: Supervised Learning – Classification

The model analyzes transaction features and predicts the transaction category.

## Dataset
The project uses a transaction dataset for training and evaluation. The original dataset is not included in this repository.

Dataset source: Add the original dataset source URL here.

## Project Category
Data Science / Machine Learning

## Developer
Sana Jameel

## Project Purpose
Developed as an academic Data Science project.
