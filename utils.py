import io
import tempfile
import keyboard

import librosa
import numpy as np
import pyaudio
from pydub import AudioSegment
import wave


DEFAULT_SAMPLE_RATE = 16000
SAMPLE_RATE = 22050
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels (mono)
RATE = 44100  # Sample rate (samples per second)


def preprocess_input(wav_data, sample_rate=DEFAULT_SAMPLE_RATE, normalize_db=0.1):
    """Convert audio file data (in bytes) into a numpy array.
    Saves to a tempfile and loads with librosa.
    Args:
      wav_data: A byte stream of audio data.
      sample_rate: Resample recorded audio to this sample rate.
      normalize_db: Normalize the audio to this many decibels. Set to None to skip
        normalization step.
    Returns:
      An array of the recorded audio at sample_rate.
    """
    # Parse and normalize the audio.
    audio = AudioSegment.from_file(io.BytesIO(wav_data))
    audio.remove_dc_offset()
    if normalize_db is not None:
        audio.normalize(headroom=normalize_db)
    # Save to tempfile and load with librosa.
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_wav_file:
        fname = temp_wav_file.name
        audio.export(fname, format='wav')
        audio_np, unused_sr = librosa.load(fname, sr=sample_rate)
    return audio_np


def resample(audio, src_sample_rate, dst_sample_rate):
    """
    Resample audio to specific sample rate

    Parameters:
      audio: input audio signal
      src_sample_rate: source audio sample rate
      dst_sample_rate: destination audio sample rate
    Returns:
      resampled_audio: input audio signal resampled with dst_sample_rate
    """
    if src_sample_rate == dst_sample_rate:
        return audio
    duration = audio.shape[0] / src_sample_rate
    resampled_data = np.zeros(shape=(int(duration * dst_sample_rate)), dtype=np.float32)
    x_old = np.linspace(0, duration, audio.shape[0], dtype=np.float32)
    x_new = np.linspace(0, duration, resampled_data.shape[0], dtype=np.float32)
    resampled_audio = np.interp(x_new, x_old, audio)
    return resampled_audio.astype(np.float32)


def audio_to_float(audio):
    """
    convert audio signal to floating point format
    """
    return audio.astype(np.float32) / np.iinfo(audio.dtype).max


def process_audio(path_: str):
    audio, sample_rate = librosa.load(path_)
    if audio.dtype != np.float32:
        audio = audio_to_float(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    resampled_audio = resample(audio, sample_rate, 16000)
    return resampled_audio


def record_from_mic():
    temp_wav = "temp.wav"
    frames = []
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=1024)

    print("Recording...Press Enter again when you want to finish it")
    while True:
        data = stream.read(1024)
        frames.append(data)
        if keyboard.is_pressed("Enter"):
            print("Finished recording.")
            break
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    return process_audio(temp_wav)
