
import pyaudio
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import keras
import logging
import tensorflow as tf


# Suppress TensorFlow logging
tf.get_logger().setLevel(logging.ERROR)  

# Suppress Keras logging
keras.utils.get_custom_objects()
keras.backend.set_learning_phase(0)  


RATE = 44100
CHUNK = 1024
THRESHOLD = 0.6  # threshold for loud sounds
CLASS_NAMES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

def extract_feature(audio_data):
    sample_rate = RATE
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    feature_scaled = np.mean(feature.T, axis=0)
    # Reshape the feature to match the expected input shape of the model
    return feature_scaled.reshape(1, -1)  # Reshape to a 2D array with shape (1, n_features)

def classify_audio(audio_data, models):
    feature = extract_feature(audio_data)
    feature = feature.reshape(-1, 128)  # Reshape here

    # feature = np.expand_dims(feature, axis=0)  
    class_names = []
    prediction = models.predict(feature)
    print(prediction)
    class_idx = np.argmax(prediction)
    # print("Predicted class index:", class_idx)  # Debug: Print predicted class index
    
    # Check if the predicted index is within the range of CLASS_NAMES
    if class_idx < len(CLASS_NAMES):
        class_name = CLASS_NAMES[class_idx]
        class_names.append(class_name)
    else:
        class_names.append("Unknown")  # Use a placeholder if the index is out of range
    return class_names

def load_model_():
    model1 = load_model('Model1.h5')
    return model1

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

    model = load_model_()

    for i in range(int(20 * RATE / CHUNK)):  # do this for 20 seconds
        audio = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        audio_float = audio.astype(np.float32) / np.iinfo(np.int16).max  # Convert to floating-point and normalize to [-1, 1]

        # Classify the audio
        class_names = classify_audio(audio_float, model)
        
        # Check if important sound class is detected
        important_sound_detected = 'important_class' in class_names  # Replace 'important_class' with the actual important class name

        if not important_sound_detected:
            inverse_audio = -audio_float  # Invert the audio signal for non-important sounds
            audio_float = audio_float + inverse_audio  # Mix the original and inversed audio

        player.write((audio_float * np.iinfo(np.int16).max).astype(np.int16).tobytes(), CHUNK)  # Convert back to int16 for PyAudio

    stream.stop_stream()
    stream.close()
    player.stop_stream()
    player.close()
    p.terminate()

record_audio()
