import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Emotion Recognition App",
    page_icon="😊",
    layout="centered"
)

# Initialize the stemmer
stemmer = PorterStemmer()

# Create model architecture
def create_model():
    model = Sequential()
    model.add(Embedding(16000, 150, input_length=66))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(16000, activation='softmax'))
    return model

# Load the model
@st.cache_resource
def load_emotion_model():
    try:
        model_path = 'Emotions_Project.h5'
        
        # Create and compile the model
        model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        
        # Load weights
        model.load_weights(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess text
def preprocess_text(text):
    # Tokenize and stem the text
    stemmed_words = [stemmer.stem(word) for word in text.split()]
    
    # Create a tokenizer and fit it on the input text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    
    # Convert text to sequence
    tokens_list = tokenizer.texts_to_sequences([stemmed_words])[0]
    
    # Pad sequence to match model's expected input length
    padded_sequence = np.zeros(66)  # Using the same length as in the original model
    padded_sequence[:len(tokens_list)] = tokens_list
    
    return padded_sequence

# Main app
def main():
    st.title("😊 Emotion Recognition App")
    st.write("Enter a text to predict the emotion it expresses.")
    
    # Load the model
    model = load_emotion_model()
    if model is None:
        return
    
    # Create text input
    text_input = st.text_area("Enter your text here:", height=100)
    
    # Emotion labels
    labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
    
    if st.button("Predict Emotion"):
        if text_input:
            # Preprocess the input text
            processed_text = preprocess_text(text_input)
            
            # Make prediction
            prediction = model.predict(np.array([processed_text]))
            predicted_class = np.argmax(prediction)
            predicted_emotion = labels_dict.get(predicted_class)
            
            # Display result
            st.markdown("---")
            st.markdown(f"### Predicted Emotion: {predicted_emotion.upper()}")
            
            # Display confidence scores
            st.markdown("### Confidence Scores:")
            for i, (emotion, score) in enumerate(zip(labels_dict.values(), prediction[0])):
                st.progress(float(score))
                st.write(f"{emotion.capitalize()}: {score:.2%}")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 