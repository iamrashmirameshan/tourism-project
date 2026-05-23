
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------
# LOAD MODEL FROM HUGGING FACE
# --------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="rashmipr/tourism-prediction-model",
        filename="tourism_prediction.joblib"
    )

    model = joblib.load(model_path)

    return model

model = load_model()

# --------------------------------------------
# TITLE
# --------------------------------------------

st.title("🌍 Tourism Package Prediction App")

st.markdown("""
This application predicts whether a customer is likely to purchase a tourism package.

Please enter the customer details below.
""")

# --------------------------------------------
# USER INPUTS
# --------------------------------------------

st.sidebar.header("Customer Information")

Age = st.sidebar.slider("Age", 18, 65, 30)

TypeofContact = st.sidebar.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.sidebar.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.sidebar.slider(
    "Duration Of Pitch",
    5,
    120,
    15
)

Occupation = st.sidebar.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.sidebar.slider(
    "Number Of Persons Visiting",
    1,
    5,
    2
)

NumberOfFollowups = st.sidebar.slider(
    "Number Of Followups",
    1,
    6,
    3
)

ProductPitched = st.sidebar.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.sidebar.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

MaritalStatus = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfTrips = st.sidebar.slider(
    "Number Of Trips",
    1,
    25,
    3
)

Passport = st.sidebar.selectbox(
    "Passport",
    [0, 1]
)

PitchSatisfactionScore = st.sidebar.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

OwnCar = st.sidebar.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.sidebar.slider(
    "Number Of Children Visiting",
    0,
    3,
    1
)

Designation = st.sidebar.selectbox(
    "Designation",
    ["Manager", "Senior Manager", "AVP", "VP", "Executive"]
)

MonthlyIncome = st.sidebar.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=100000,
    value=25000
)

# --------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------

input_data = pd.DataFrame({
    'Age': [Age],
    'TypeofContact': [TypeofContact],
    'CityTier': [CityTier],
    'DurationOfPitch': [DurationOfPitch],
    'Occupation': [Occupation],
    'Gender': [Gender],
    'NumberOfPersonVisiting': [NumberOfPersonVisiting],
    'NumberOfFollowups': [NumberOfFollowups],
    'ProductPitched': [ProductPitched],
    'PreferredPropertyStar': [PreferredPropertyStar],
    'MaritalStatus': [MaritalStatus],
    'NumberOfTrips': [NumberOfTrips],
    'Passport': [Passport],
    'PitchSatisfactionScore': [PitchSatisfactionScore],
    'OwnCar': [OwnCar],
    'NumberOfChildrenVisiting': [NumberOfChildrenVisiting],
    'Designation': [Designation],
    'MonthlyIncome': [MonthlyIncome]
})

# --------------------------------------------
# PREDICTION
# --------------------------------------------

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            f"✅ Customer is likely to purchase the tourism package.\n\nProbability: {probability:.2%}"
        )

    else:
        st.error(
            f"❌ Customer is unlikely to purchase the tourism package.\n\nProbability: {probability:.2%}"
        )

    st.subheader("Input Data")

    st.dataframe(input_data)
