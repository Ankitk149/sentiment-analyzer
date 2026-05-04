import tensorflow as tf
import pickle
from tensorflow.keras.layers import TextVectorization

# Load model
model = tf.keras.models.load_model("/home/ankit/mvl/Sentiment-app/sentiment_model.keras")

# Load vocabulary
with open("vectorizer.pkl", "rb") as f:
    vocab = pickle.load(f)

# Recreate vectorizer
vectorize_layer = TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=250
)
vectorize_layer.set_vocabulary(vocab)

def predict(text):
    text = tf.constant([text])
    x = vectorize_layer(text)
    pred = model.predict(x)[0][0]

    sentiment = "Positive" if pred > 0.5 else "Negative"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    return sentiment, confidence