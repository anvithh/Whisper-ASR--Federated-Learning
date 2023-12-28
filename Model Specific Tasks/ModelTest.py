import jiwer
import pandas as pd
import os
import torchaudio
from torchaudio.transforms import Resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Dataset from the CSV file
data = pd.read_csv('duptest.csv')

audio_files_dir = '.'

# Initialize the Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="tamil", task="transcribe")

sampling_rate = 16000  #Actual sampling rate
wer_list = []

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

        transcription = processor.batch_decode(predicted_ids)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        actual = row['sentence']

        wer = jiwer.wer(actual, transcription[0])

        wer_list.append(wer)

        print(f"Audio: {audio_filename}")
        print(f"Transcription: {transcription}\n")
        print(f"WER: {wer}\n")
    else:
        print(f"Audio file not found: {audio_filename}")