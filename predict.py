import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

model = load_model("model.h5")

with open("vectorizer.pkl", "rb") as f:
    vocab = pickle.load(f)

vectorize_layer = TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=250
)
vectorize_layer.set_vocabulary(vocab)

def predict(text):
    text = np.array([text])
    x = vectorize_layer(text)
    pred = model.predict(x)[0][0]

    sentiment = "Positive" if pred > 0.5 else "Negative"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    return sentiment, confidence