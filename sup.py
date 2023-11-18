import pyaudio
import numpy as np
from tensorflow import load_model
RATE = 44100
CHUNK = 1024
THRESHOLD = 0.6  # threshold for loud sounds

def record_audio():
    p = pyaudio.PyAudio()

    player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    model1 = load_model('Audio-Classification-Deep-Learning/assets/Model1.h5') 
    model2 = load_model('Audio-Classification-Deep-Learning/assets/Model2.h5')
    model3 = load_model('Audio-Classification-Deep-Learning/assets/Model3.h5')

    for i in range(int(20 * RATE / CHUNK)):  # do this for 10 seconds
        audio = np.frombuffer(stream.read(CHUNK), dtype=np.int16)  # Use np.frombuffer instead of np.fromstring
        if np.max(audio) < THRESHOLD * np.iinfo(np.int16).max:
            inverse_audio = -audio  # inverse the audio signal
            mixed_audio = audio + inverse_audio  # mix the original and inversed audio
            player.write(mixed_audio.tobytes(), CHUNK)
        else:
            print("threshold not reached")
            player.write(audio.tobytes(), CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()

record_audio()
