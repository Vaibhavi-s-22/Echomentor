import pickle
import librosa
import numpy as np

def load_model(model_file):
    """Load the pre-trained model from a pickle file."""
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

def extract_mfcc(audio_file, num_mfcc=13):
    """Extract MFCC features from the audio file."""
    waveform, sample_rate = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(waveform, sr=sample_rate, n_mfcc=num_mfcc)
    return mfccs

def classify_audio(audio_file, model):
    """Classify the audio using the pre-trained model."""
    mfccs = extract_mfcc(audio_file)
    mfccs_flattened = mfccs.reshape(1, -1)  # Flatten MFCC features
    prediction = model.predict(mfccs_flattened)
    return prediction

if __name__ == "__main__":
    # Paths to the pickle model and audio file
    model_file = 'model.pickle'
    audio_file = 'sample_audio.wav'

    # Load the pre-trained model
    model = load_model(model_file)

    # Classify the audio file
    prediction = classify_audio(audio_file, model)

    print("Predicted label:", prediction)
