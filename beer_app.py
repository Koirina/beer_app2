import streamlit as st
import joblib
import pandas as pd

# Loading the KNN model
model = joblib.load('models/knn-k3.joblib')

# Loading breweries
with open('models/brewery_names.pkl', 'rb') as file:
    brewery_names = pd.read_pickle(file)

# Setting page title and add some style
st.set_page_config(page_title="Beer Style Prediction", page_icon="🍺", layout="centered")

# Create a centered header
st.title("Beer Style Predictor")
st.markdown("Enter the details of your beer to predict its style.")

# Create input fields for new data points
brewery_name = st.selectbox('Brewery Name', brewery_names)
review_aroma = st.slider('Review Aroma (1-5)', 1.0, 5.0, step=0.1)
review_appearance = st.slider('Review Appearance (1-5)', 1.0, 5.0, step=0.1)
review_palate = st.slider('Review Palate (1-5)', 1.0, 5.0, step=0.1)
review_taste = st.slider('Review Taste (1-5)', 1.0, 5.0, step=0.1)
beer_abv = st.slider('Beer ABV', 0.0, 95.0, step=0.1)

# Create a button to predict beer style
if st.button('Predict Beer Style'):
    # Prepare the data for prediction
    new_data = {
        'brewery_name': brewery_name,
        'review_aroma': review_aroma,
        'review_appearance': review_appearance,
        'review_palate': review_palate,
        'review_taste': review_taste,
        'beer_abv': beer_abv
    }

    # Create a DataFrame from the new data
    new_data_df = pd.DataFrame([new_data])

    # Make a prediction using the pre-trained KNN model
    prediction = model.predict(new_data_df)

    # Display the predicted beer style in a highlighted box
    st.markdown(f"**Predicted Beer Style:** {prediction[0]}", unsafe_allow_html=True)





