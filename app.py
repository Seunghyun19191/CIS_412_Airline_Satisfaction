
import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load('decision_tree_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title('Airline Passenger Satisfaction Prediction')
st.write('Predict passenger satisfaction using a Decision Tree model.')

# Passenger Information
st.header('Passenger Information')

gender = st.selectbox('Gender', ['Male', 'Female'])
customer_type = st.selectbox('Customer Type', ['Loyal Customer', 'disloyal Customer'])
age = st.number_input('Age', min_value=7, max_value=85, value=30)
travel_type = st.selectbox('Type of Travel', ['Business travel', 'Personal Travel'])
travel_class = st.selectbox('Class', ['Business', 'Eco', 'Eco Plus'])
flight_distance = st.number_input('Flight Distance', min_value=0, value=1000)

# Service Ratings
st.header('Service Ratings')

rating_features = [
    'Inflight wifi service',
    'Departure/Arrival time convenient',
    'Ease of Online booking',
    'Gate location',
    'Food and drink',
    'Online boarding',
    'Seat comfort',
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Inflight service',
    'Cleanliness'
]

ratings = {}

for feature in rating_features:
    ratings[feature] = st.slider(feature, 0, 5, 3)

# Flight Delays
st.header('Flight Delays')

departure_delay = st.number_input(
    'Departure Delay in Minutes', min_value=0, value=0
)

arrival_delay = st.number_input(
    'Arrival Delay in Minutes', min_value=0, value=0
)

# Create input data
input_data = {
    'Gender': gender,
    'Customer Type': customer_type,
    'Age': age,
    'Type of Travel': travel_type,
    'Class': travel_class,
    'Flight Distance': flight_distance,
    'Departure Delay in Minutes': departure_delay,
    'Arrival Delay in Minutes': arrival_delay
}

input_data.update(ratings)

input_df = pd.DataFrame([input_data])

# One-Hot Encoding
input_encoded = pd.DataFrame(0, index=[0], columns=feature_columns)

for col in input_df.columns:
    if col in input_encoded.columns:
        input_encoded[col] = input_df[col].values
    else:
        encoded_col = col + '_' + str(input_df[col].iloc[0])
        if encoded_col in input_encoded.columns:
            input_encoded[encoded_col] = 1

# Prediction
if st.button('Predict Satisfaction'):
    prediction = model.predict(input_encoded)

    st.subheader('Prediction Result')
    st.write(prediction[0])
