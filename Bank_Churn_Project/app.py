import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Smart Customer Retention Analytics System")

st.subheader(
    "AI-Based Customer Churn Prediction for European Banks"
)

st.info("""
This system uses Machine Learning to identify customers
who are likely to leave the bank and assigns a risk score.

The goal is to help banks improve customer retention,
reduce customer loss and support decision making.
""")

st.markdown("""
### Internship Project

**Developed By:** Lakhendra Mirchanay

""")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")
    return df

df = load_data()


# -----------------------------
# PREPROCESSING
# -----------------------------
data = df.copy()

data.drop(["CustomerId", "Surname"], axis=1, inplace=True)

geo_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

data["Geography"] = geo_encoder.fit_transform(data["Geography"])
data["Gender"] = gender_encoder.fit_transform(data["Gender"])

X = data.drop("Exited", axis=1)
y = data["Exited"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Customer Profile Analysis")

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

country = st.sidebar.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.slider(
    "Age",
    18,
    100,
    35
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Balance",
    value=50000.0
)

products = st.sidebar.selectbox(
    "Number of Products",
    [1, 2, 3, 4]
)

credit_card = st.sidebar.selectbox(
    "Has Credit Card",
    [1, 0]
)

active_member = st.sidebar.selectbox(
    "Is Active Member",
    [1, 0]
)

salary = st.sidebar.number_input(
    "Estimated Salary",
    value=60000.0
)

# -----------------------------
# DASHBOARD METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))

col2.metric(
    "Model Accuracy",
    f"{accuracy*100:.2f}%"
)

col3.metric(
    "Churn Rate",
    f"{df['Exited'].mean()*100:.2f}%"
)

col4, col5 = st.columns(2)

col4.metric(
    "Average Age",
    round(df["Age"].mean(), 1)
)

col5.metric(
    "Average Balance",
    f"${df['Balance'].mean():,.0f}"
)

# -----------------------------
# CHURN VISUALIZATION
# -----------------------------
st.subheader("Customer Churn Distribution")

fig, ax = plt.subplots()

df["Exited"].value_counts().plot(
    kind="pie",
    labels=["Stayed", "Left"],
    autopct="%1.1f%%",
    ax=ax
)

ax.set_ylabel("")

st.pyplot(fig)

# -----------------------------
# CUSTOMER RISK ASSESSMENT
# -----------------------------

if st.button("Generate Risk Assessment Report"):

    country_encoded = geo_encoder.transform([country])[0]
    gender_encoded = gender_encoder.transform([gender])[0]

    customer_data = [[
        2025,
        credit_score,
        country_encoded,
        gender_encoded,
        age,
        tenure,
        balance,
        products,
        credit_card,
        active_member,
        salary
    ]]

    churn_probability = model.predict_proba(customer_data)[0][1]

    retention_prediction = model.predict(customer_data)[0]

    customer_risk = int(churn_probability * 100)

    st.subheader("Customer Risk Assessment Report")

    colA, colB = st.columns(2)

    with colA:
        st.metric(
            "Customer Risk Score",
            f"{customer_risk}/100"
        )

    with colB:
        st.metric(
            "Churn Probability",
            f"{churn_probability*100:.2f}%"
        )

    st.progress(churn_probability)

    if customer_risk < 30:

        st.success(
            "Customer Category: LOW RISK"
        )

    elif customer_risk < 70:

        st.warning(
            "Customer Category: MEDIUM RISK"
        )

    else:

        st.error(
            "Customer Category: HIGH RISK"
        )

    st.markdown("---")

    if retention_prediction == 1:

        st.error(
            "Prediction: Customer is likely to leave the bank."
        )

        st.subheader("Recommended Retention Strategy")

        st.warning("""
        • Offer loyalty rewards

        • Assign relationship manager

        • Provide personalized financial products

        • Offer premium customer support

        • Provide special loan or credit card offers
        """)

    else:

        st.success(
            "Prediction: Customer is likely to remain with the bank."
        )

        st.info("""
        Current customer behaviour indicates
        strong retention potential.

        Continue providing standard banking services.
        """)

    country_encoded = geo_encoder.transform([country])[0]
    gender_encoded = gender_encoder.transform([gender])[0]

    customer_data = [[
        2025,
        credit_score,
        country_encoded,
        gender_encoded,
        age,
        tenure,
        balance,
        products,
        credit_card,
        active_member,
        salary
    ]]

    probability = model.predict_proba(customer_data)[0][1]
    prediction = model.predict(customer_data)[0]

    risk_score = int(probability * 100)

    st.subheader("Prediction Result")

    st.metric(
        "Risk Score",
        f"{risk_score}/100"
    )

    st.progress(probability)

    if risk_score < 30:
        st.success("LOW RISK CUSTOMER")

    elif risk_score < 70:
        st.warning("MEDIUM RISK CUSTOMER")

    else:
        st.error("HIGH RISK CUSTOMER")

    if prediction == 1:
        st.error("Customer is likely to leave the bank.")
    else:
        st.success("Customer is likely to stay.")

# -----------------------------
# DATA PREVIEW
# -----------------------------
with st.expander("Dataset Sample Records"):

    st.dataframe(df.head(20))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("Factors Influencing Customer Churn")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig2, ax2 = plt.subplots()

ax2.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax2.invert_yaxis()

st.pyplot(fig2)

st.markdown("---")

st.subheader("Model Information")

st.write("""
Algorithm Used: Random Forest Classifier

Why Random Forest?

• High prediction accuracy

• Handles large datasets efficiently

• Suitable for customer classification problems

• Less prone to overfitting

Project Objective:

To identify customers who are likely to leave the bank
and provide actionable insights that help improve
customer retention and business decision-making.
""")

st.markdown("---")

st.caption(
    "Smart Customer Retention Analytics System | Internship Project "
)