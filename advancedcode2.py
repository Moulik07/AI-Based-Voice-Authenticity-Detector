# Importing necessary libraries
import os
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
import glob
from twilio.rest import Client

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
device = torch.device("cpu")

# Twilio Configuration for sending SMS alerts
TWILIO_ACCOUNT_SID = '****75****dd72e****185dcf****f746*'
TWILIO_AUTH_TOKEN = '********************************'
TWILIO_PHONE_NUMBER = '+1*05****013'

# Initialize Wav2Vec2 model and tokenizer
tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=2).to(device)

def preprocess_audio(path):
    """
    Load and preprocess audio from the given path.
    """
    try:
        audio, source_sr = librosa.load(path, sr=None, res_type='kaiser_fast')
        if source_sr != 16000:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=16000)
        return audio, 16000
    except Exception as e:
        print(f"Error with {path}: {e}")
        return None, None

def extract_features(audio, sr):
    """
    Extract features from audio using Wav2Vec2.
    """
    input_values = tokenizer([audio], return_tensors="pt", padding="longest", sampling_rate=sr).input_values.to(device)
    logits = model(input_values).logits.detach().cpu().numpy().squeeze()
    return logits.reshape(-1)

def load_data_from_folder(folder_path):
    """
    Load audio data and labels from the specified folder.
    """
    X, y = [], []
    for label, value in [("real", 1), ("fake", 0)]:
        for file in glob.glob(f"{folder_path}/{label}/*.wav"):
            audio, sr = preprocess_audio(file)
            if audio is not None:
                X.append(extract_features(audio, sr))
                y.append(value)
    return np.array(X), np.array(y)

def train_model(X_train, y_train, X_val, y_val):
    """
    Train a classifier to distinguish between real and fake audio.
    """
    # Ensure the input data is 3D
    if len(X_train.shape) == 2:
        X_train = np.expand_dims(X_train, axis=-1)
    if len(X_val.shape) == 2:
        X_val = np.expand_dims(X_val, axis=-1)

    # Define the classifier model
    classifier = Sequential([
        Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    save_model(classifier, 'voice_classifier_model')
    return classifier

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    """
    y_pred = (model.predict(X_test) > 0.5).astype(int).squeeze()
    metrics = {
        "Accuracy": np.mean(y_pred == y_test),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

def predict_audio(model, path):
    """
    Predict if the given audio is real or fake.
    """
    audio, sr = preprocess_audio(path)
    if audio is not None:
        features = extract_features(audio, sr)
        features = features.reshape(1, features.shape[0], 1)
        prediction = model.predict(features)
        return "real" if prediction[0][0] > 0.5 else "fake"
    return "Error with audio"

# Main execution
if __name__ == "__main__":
    # Load training, validation, and testing data
    X_train, y_train = load_data_from_folder("for-rerecorded/training")
    X_val, y_val = load_data_from_folder("for-rerecorded/validation")
    X_test, y_test = load_data_from_folder("for-rerecorded/testing")

    # Train the model
    trained_model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate the model
    evaluate_model(trained_model, X_test, y_test)

    # Predict using a sample audio
    sample_audio_path = "suna hai yaadein audio sample_Moulik.mp3"
    label = predict_audio(trained_model, sample_audio_path)
    print(f"Predicted Label for {sample_audio_path}: {label}")
