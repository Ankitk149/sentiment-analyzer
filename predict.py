import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

# ✅ correct model file name
model = load_model("sentiment_model.keras")

# ✅ load vocabulary
with open("vectorizer.pkl", "rb") as f:
    vocab = pickle.load(f)

# ✅ rebuild vectorizer
vectorize_layer = TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=250
)
vectorize_layer.set_vocabulary(vocab)

def predict(text):
    text = np.array([text])
    x = vectorize_layer(text)

    # ✅ silent prediction
    pred = model.predict(x, verbose=0)[0][0]

    sentiment = "Positive" if pred > 0.5 else "Negative"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    return sentiment, confidence