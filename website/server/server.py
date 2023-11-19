from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import pyaudio
import numpy as np
import torch
import threading
import time

from speechbrain.pretrained import EncoderClassifier

app = Flask(__name__)
CORS(app)
# Load the pre-trained model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/urbansound8k_ecapa",
    savedir="pretrained_models/gurbansound8k_ecapa"
)

# Global variables for audio streaming
audio_frames = []
confidence = 0.0
danger = False
stop_i = 0
stream_thread = None

# Function to record audio and perform classification
def record_and_classify():
    global audio_frames, confidence, danger, stop_i
    audio = pyaudio.PyAudio()

    # Get default input device info
    default_device_index = audio.get_default_input_device_info()["index"]

    # Define audio parameters
    FORMAT = pyaudio.paInt16  # Sample format
    CHANNELS = 1              # Number of audio channels (mono)
    RATE = 16000              # Sample rate (samples per second)
    CHUNK = 1024              # Number of frames per buffer
    DURATION = 0.5            # Duration of audio to record

    # Initialize stream outside the try block
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=default_device_index,
                        frames_per_buffer=CHUNK)
    player = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

    try:
        while True:
            frames = []
            for i in range(0, int(RATE / CHUNK * DURATION)):
                audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
                inverse_audio = -audio_data
                cancelled_audio = audio_data + inverse_audio
                frames.append(audio_data)

                if danger:
                    stop_i = i + 5
                if i < stop_i:
                    player.write(audio_data.tobytes(), CHUNK)
                else:
                    player.write(cancelled_audio.tobytes(), CHUNK)

            # Stop recording
            stream.stop_stream()
            stream.close()

            # Convert the recorded audio frames to a single numpy array
            recorded_audio = np.concatenate(frames)

            # Convert numpy array to torch tensor
            audio_tensor = torch.tensor(recorded_audio, dtype=torch.float32)

            # Perform classification on the recorded audio
            out_prob, score, index, text_lab = classifier.forward(audio_tensor.unsqueeze(0))

            # Update global variables
            confidence = float(score)
            label = text_lab[0]
            print(label) # printed in server.py
            if label in ["siren", "gun_shot", "car_horn", "engine_idling"]:
                danger = True
                stop_i = i + 5
            else:
                danger = False

            # Wait for a short duration before starting the next recording
            # Reopen the audio stream for the next recording
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                input_device_index=default_device_index, frames_per_buffer=CHUNK)

    except KeyboardInterrupt:
        print("Recording and classification interrupted.")

    finally:
        # Close the audio stream
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        audio.terminate()


# Flask route for streaming audio and confidence
@app.route('/audio_stream')
def audio_stream():
    def generate():
        while True:
            yield f"data:{confidence},{danger}\n\n"
    return Response(generate(), mimetype="text/event-stream")

# Flask route for starting the recording and classification
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global stream_thread
    if not stream_thread or not stream_thread.is_alive():
        # Start a new thread for audio streaming
        stream_thread = threading.Thread(target=record_and_classify)
        stream_thread.start()
        return jsonify({'status': 'Recording started'})
    else:
        return jsonify({'status': 'Recording already in progress'})

if __name__ == '__main__':
    audio = pyaudio.PyAudio()

    # Get default input device info
    default_device_index = audio.get_default_input_device_info()["index"]

    # Define audio parameters
    FORMAT = pyaudio.paInt16  # Sample format
    CHANNELS = 1              # Number of audio channels (mono)
    RATE = 16000              # Sample rate (samples per second)
    CHUNK = 1024              # Number of frames per buffer
    DURATION = 2.5            # Duration of audio to record

    # Open audio stream for recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=default_device_index,
                        frames_per_buffer=CHUNK)
    player = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

    try:
        app.run(port=8080, debug=True)
    except KeyboardInterrupt:
        print("Flask app terminated.")
    finally:
        audio.terminate()
