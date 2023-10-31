import os
import torch
import pandas as pd
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, TrainingArguments, Trainer,WhisperProcessor
import torchaudio
import datasets
from transformers import Wav2Vec2Processor 
from transformers import DataCollatorWithPadding

# Load your custom dataset from a CSV file
dataset_path = './female/dup.csv'
custom_dataset = pd.read_csv(dataset_path)

# Directory where audio files are located
audio_files_dir = './female/'

# Load the pre-trained Whisper model and tokenizer
model_name = "openai/whisper-tiny"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


# Define a function to load and preprocess audio data
def preprocess_csv_data(example):
    audio_files = example["audio"]

    if isinstance(audio_files, list):
        audio_files = [str(audio) for audio in audio_files]

    processed_examples = []

    for audio_file in audio_files:
        try:
            audio_path = os.path.join(audio_files_dir, audio_file)
            # Load the audio file
            waveform, sample_rate = torchaudio.load(audio_path)

            # Tokenize transcription
            labels = tokenizer(example["transcription"], return_tensors="pt", padding="longest", truncation=True)

            processed_examples.append({"labels": labels.input_ids, "audio": waveform, "sampling_rate": sample_rate})
        except Exception as e:
            print(f"Error processing audio file: {audio_file}")
            print(f"Exception: {str(e)}")

    return {"data": processed_examples}  # Return a dictionary with the list of processed examples

# Create a `datasets.Dataset` object
custom_dataset = datasets.Dataset.from_pandas(custom_dataset)

# # Tokenize and preprocess your custom dataset
custom_dataset = custom_dataset.map(preprocess_csv_data, batched=True)

# # Flatten the list of processed examples
custom_dataset = custom_dataset.flatten()

# train_dataset, eval_dataset = custom_dataset.train_test_split(test_size=0.1, seed=42)

# # Tokenize and preprocess your custom dataset
# train_dataset = train_dataset.map(preprocess_csv_data, batched=True)
# eval_dataset = eval_dataset.map(preprocess_csv_data, batched=True)



# print("Number of examples in the processed dataset:", len(custom_dataset))

# Define a custom data collator
class CustomDataCollator(DataCollatorWithPadding):
    def collate_batch(self, batch):
        audio_inputs = self.prepare_audio_inputs([example["audio"] for example in batch])
        label_inputs = self.prepare_labels([example["labels"] for example in batch])

        return {
            "input_values": audio_inputs["input_values"],
            "attention_mask": audio_inputs["attention_mask"],
            "labels": label_inputs["input_ids"],
        }

    def prepare_audio_inputs(self, audio_list):
        audio_inputs = processor(audio_list, return_tensors="pt", padding=True, truncation=True)
        return {
            "input_values": audio_inputs["input_values"],
            "attention_mask": audio_inputs["attention_mask"],
        }

    def prepare_labels(self, label_list):
        labels = tokenizer(label_list, return_tensors="pt", padding="longest", truncation=True)
        return {
            "input_ids": labels.input_ids,
            "attention_mask": labels.attention_mask,
        }

# print(custom_dataset["data.audio"])
# processor = WhisperProcessor.from_pretrained(model_name)

# train_dataset = custom_dataset.train_test_split(test_size=0.2) 
# Usage
custom_data_collator = CustomDataCollator(tokenizer)

# train_dataset = train_dataset['train']
# eval_dataset = train_dataset['test']


# Define training arguments
training_args = TrainingArguments(
    output_dir="./whisper_finetuned_model",
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
    num_train_epochs=10,
    logging_dir="./logs",
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=custom_dataset,
    data_collator=custom_data_collator,  # Use the custom data collator
)

# Fine-tune the model on the custom dataset
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_whisper_model")
tokenizer.save_pretrained("./fine_tuned_whisper_model")
