import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import io

# Load the model
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')

# Create a function to make predictions and get probabilities
def make_prediction(features):
    df = pd.DataFrame([features], columns=['NumberOfTenderers', 'MainCategory', 'Budget', 'BidAmount', 'TenderDurationDays', 'ContractDurationDays'])
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]  # Probability of class 1
    return prediction[0], probability, df

# Streamlit application
st.title('Tender Participation Recommender')
st.markdown('This application predicts the probability of participation in a tender process.')

# Create user inputs organized in columns
col1, col2 = st.columns(2)

with col1:
    number_of_tenderers = st.number_input('Number of Tenderers', min_value=0, help="Enter the number of tenderers in the tender.")
    main_category = st.selectbox('Main Category', options=['goods', 'services', 'works'], help="Select the main category of the tender.")
    budget = st.number_input('Budget', min_value=0.0, help="Enter the total budget for the tender.")

with col2:
    bid_amount = st.number_input('Bid Amount', min_value=0.0, help="Enter the bid amount.")
    tender_duration_days = st.number_input('Tender Duration (Days)', min_value=0, help="Enter the duration of the tender in days.")
    contract_duration_days = st.number_input('Contract Duration (Days)', min_value=0, help="Enter the duration of the contract in days.")

# Convert inputs to numerical values
category_map = {'goods': 0, 'services': 1, 'works': 2}
category_num = category_map[main_category]

if st.button('Make Prediction'):
    features = [number_of_tenderers, category_num, budget, bid_amount, tender_duration_days, contract_duration_days]
    prediction, probability, df = make_prediction(features)
    
    st.subheader('Results')
    if probability >= 0.75:
        st.success(f'**Recommendation:** High probability of participation ({probability:.2f}). It is recommended to participate in the tender.')
    elif 0.5 <= probability < 0.75:
        st.warning(f'**Recommendation:** Moderate probability of participation ({probability:.2f}). Consider participating if you can take certain risks.')
    else:
        st.error(f'**Recommendation:** Low probability of participation ({probability:.2f}). It is recommended not to participate in the tender.')
        
    # Visualization
    st.progress(int(probability * 100))
