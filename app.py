import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
st.set_page_config(
    page_title="Loan Approval App",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
df = pd.read_csv("loan.csv")
# Input (X) and Output (y)
# ----------------------------
X = df[['Income', 'CIBIL_Score', 'Loan_Amount', 'Employment_Years']]
y = df['Loan_Status']
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42
)

model.fit(x_train, y_train)

# Model Accuracy
prediction = model.predict(x_test)
accuracy = accuracy_score(y_test, prediction)

st.success(f"Model Accuracy : {accuracy*100:.2f}%")

st.sidebar.header("Enter Loan Details")

income = st.sidebar.number_input(
    "💵 Income",
    min_value=100000,
    max_value=1000000,
    value=400000,
    step=10000
)

cibil = st.sidebar.number_input(
    "📊 CIBIL Score",
    min_value=300,
    max_value=900,
    value=700,
    step=10
)

loan = st.sidebar.number_input(
    "💰 Loan Amount",
    min_value=5000,
    max_value=1000000,
    value=200000,
    step=5000
)

experience = st.sidebar.number_input(
    "👨‍💼 Employment Years",
    min_value=0,
    max_value=40,
    value=5,
    step=1
)
if st.button("Predict Loan"):

    user_data = [[income, cibil, loan, experience]]

    result = model.predict(user_data)

    if result[0] == 1:

        approved = min(
            loan,
            int(income * 0.8 + (cibil - 650) * 400 + experience * 12000)
        )

        if approved < 0:
            approved = 0

        st.success("🎉 Loan Approved")
        st.info(f"Approved Loan Amount : ₹{approved:,}")

    else:
        st.error("❌ Loan Rejected")
st.markdown("---")
st.caption("Mini Project 3 | Loan Approval Prediction | Decision Tree Classifier")