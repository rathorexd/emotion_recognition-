# Emotion Recognition App

A Streamlit-based web application that uses a Bidirectional LSTM model to predict emotions from text input. The model can classify text into six different emotions: sadness, joy, love, anger, fear, and surprise.

## Features

- Real-time emotion prediction from text input
- Visual display of confidence scores for all emotions
- Clean and intuitive user interface
- Support for six different emotions
- Text preprocessing with stemming

## Prerequisites

Before running the application, make sure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the source code.

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the required NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Model Architecture

The application uses a Bidirectional LSTM model with the following architecture:
- Embedding Layer (16000 vocabulary size, 150 dimensions)
- Bidirectional LSTM Layer (150 units)
- Dense Output Layer with Softmax activation

## Usage

1. Make sure the model file `Emotions_Project.h5` is in the same directory as `app.py`

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

4. Enter your text in the text area and click "Predict Emotion" to see the results

## Input Format

- Enter any text in the text area
- The model will process the text and predict the most likely emotion
- Results will show the predicted emotion and confidence scores for all emotions

## Output

The application provides:
- The predicted emotion (sadness, joy, love, anger, fear, or surprise)
- Confidence scores for each emotion displayed as progress bars
- Percentage values for each emotion's confidence

## Dependencies

- streamlit==1.32.0
- tensorflow==2.12.0
- numpy==1.23.5
- nltk==3.8.1

## Note

Make sure you have the trained model file (`Emotions_Project.h5`) in the same directory as the application. The model file should contain the weights of a trained Bidirectional LSTM model.

## License

This project is open source and available under the MIT License. 