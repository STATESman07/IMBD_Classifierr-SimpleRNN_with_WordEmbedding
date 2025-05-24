# imports libraries
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# load model
model = load_model("imbd.keras")

# load the imbd dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


# Step 2: Helper functions
# decode the review
def decode_review(encode_text):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encode_text])


# function to preprocess the User review
def pre_process_text(text):
    words = text.lower().split()
    # enocde text to vector
    encoded_review = [word_index.get(word, 2) + 3 for word in words]

    # padding means adding zero so that all reviews have same length
    padding_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padding_review


# i did not use this fuction
### Step 3: predict the sentiment of the review
# def predict_review(review):
#     preprocessed_input=pre_process_text(review)

#     prediction=model.predict(preprocessed_input)

#     if prediction[0][0]>0.5:
#         sentiment="Positive Review"
#     else:
#         sentiment="Negative Review"

#     return sentiment,prediction,prediction[0][0]


## streamlit app
st.title("IMBDSentiment Analysis")
st.write("This app predicts the sentiment of a given review")

user_input = st.text_input("Enter your review")

if st.button("Classify"):

    preprocess_value = pre_process_text(user_input)

    ## make prediction
    prediction = model.predict(preprocess_value)
    sentiment = "Positive Review" if prediction[0][0] > 0.5 else "Negative Review"

    # Display output
    st.write(f"Sentiment: {sentiment}")
    st.write(f"prediction: {prediction}")
    st.write(f"score: {prediction[0][0]}")
else:
    st.write("Please enter a review")
