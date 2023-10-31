import pandas as pd
import os
import torchaudio
from torchaudio.transforms import Resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load the dataset from the CSV file
data = pd.read_csv('./female/dup.csv')

# Specify the directory where audio files are located
audio_files_dir = './female/'

# Initialize the Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="telugu", task="transcribe")

# Define the sampling rate
sampling_rate = 16000  # The expected sampling rate of the model

for index, row in data.iterrows():
    audio_filename = row['audio']
    audio_path = os.path.join(audio_files_dir, audio_filename)

    if os.path.exists(audio_path):
        # Read the audio file and process it as an array
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample the audio
        resampler = Resample(orig_freq=sample_rate, new_freq=sampling_rate)
        audio_data_resampled = resampler(waveform)

        if audio_data_resampled.shape[0] > 1:
            audio_data_resampled = torch.mean(audio_data_resampled, dim=0)

        # Convert the audio data to a numpy array
        audio_data_resampled = audio_data_resampled.numpy()

        input_features = processor(audio_data_resampled, sampling_rate=sampling_rate, return_tensors="pt").input_features

        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        print(f"Audio: {audio_filename}")
        print(f"Transcription: {transcription}\n")
    else:
        print(f"Audio file not found: {audio_filename}")
