import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

vocab_size = 10000
max_len = 100

def build_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=16, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_model()
    model.save("sentiment_model.h5")
    print("Model saved!")
