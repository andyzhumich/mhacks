import pyaudio
import numpy as np
import torch
import time

from speechbrain.pretrained import EncoderClassifier

# Load the pre-trained model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/urbansound8k_ecapa",
    savedir="pretrained_models/gurbansound8k_ecapa"
)

# Function to record 5 seconds of audio and perform classification
def record_5s_and_classify(device='cpu'):
    # Set the device (CPU/GPU)
    device = torch.device(device)

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Get default input device info
    default_device_index = audio.get_default_input_device_info()["index"]

    # Define audio parameters
    FORMAT = pyaudio.paInt16  # Sample format
    CHANNELS = 1              # Number of audio channels (mono)
    RATE = 16000              # Sample rate (samples per second)
    CHUNK = 1024              # Number of frames per buffer
    DURATION = 2.5              # Duration of audio to record

    # Open audio stream for recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=default_device_index,
                        frames_per_buffer=CHUNK)

    try:
        while True:
            print(f"Recording audio for {DURATION} seconds...")
            frames = []

            # Record audio for the specified duration
            for i in range(0, int(RATE / CHUNK * DURATION)):
                audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
                frames.append(audio_data)

            # Stop recording
            print("Finished recording.")
            stream.stop_stream()
            stream.close()

            # Convert the recorded audio frames to a single numpy array
            recorded_audio = np.concatenate(frames)

            # Convert numpy array to torch tensor
            audio_tensor = torch.tensor(recorded_audio, dtype=torch.float32).to(device)

            # Perform classification on the recorded audio
            out_prob, score, index, text_lab = classifier.forward(audio_tensor.unsqueeze(0))

            # Print the predicted label and the associated confidence score
            print("Predicted class:", text_lab, "with confidence:", score)

            # Wait for a short duration before starting the next recording
            time.sleep(1)  # Adjust this value if needed

            # Reopen the audio stream for the next recording
            stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                input_device_index=default_device_index,
                                frames_per_buffer=CHUNK)

    except KeyboardInterrupt:
        print("Recording and classification interrupted.")

    finally:
        # Close the audio stream
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        audio.terminate()

# Call the function to record for 5 seconds and classify
record_5s_and_classify(device='cpu')  # Use 'cuda' for GPU if available
