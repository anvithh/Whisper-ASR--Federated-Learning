import multiprocessing  # Import the multiprocessing module

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Add this line if necessary
    # Rest of your code that starts processes

from dataclasses import dataclass
from typing import Any,Dict,List,Union
import torch
from datasets import load_dataset, DatasetDict
import flwr as fl
import torch
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
from transformers import WhisperProcessor
from dataclasses import dataclass
from huggingface_hub import HfApi, HfFolder

import transformers
from transformers import WhisperForConditionalGeneration, WhisperTokenizer

# -- LOADING THE MODEL --
model = WhisperForConditionalGeneration.from_pretrained("./tamilnew2")
tokenizer = WhisperTokenizer.from_pretrained("./tamilnew2")

# -- LOADING THE DATASET --

common_voice = load_dataset(
    'csv', data_files={
        'train': './dup.csv', 
        'test':'./duptest.csv'
    }
)

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    """
    Prepare audio data to be suitable for Whisper AI model.
    """
    # (1) load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # (2) compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # (3) encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=2 # num_proc > 1 will enable multiprocessing
    )
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Use Data Collator to perform Speech Seq2Seq with padding
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        training_args = transformers.Seq2SeqTrainingArguments(
    output_dir="./federated_training",
    # overwrite_output_dir=True,  # If you want to retrain from scratch
    per_device_train_batch_size=2,  # Adjust based on your hardware
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    learning_rate=1e-4,
    warmup_steps=500,
    evaluation_strategy="epoch",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    early_stopping_patience=3,
    gradient_accumulation_steps=4,  # Adjust based on batch size
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

        self.local_dataset = load_dataset(
            "csv", data_files={"train": "./dup.csv", "test": "./duptest.csv"}
        )
        self.local_dataset = self.local_dataset.map(
            prepare_dataset,
            remove_columns=self.local_dataset.column_names["train"],
            num_proc=2,
        )
        self.model = model  # Your pre-loaded Whisper model
        self.tokenizer = tokenizer  # Your pre-loaded Whisper tokenizer

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, client):
        train_data = self.local_dataset["train"].select(range(100))
        train_data = train_data.map(prepare_dataset, remove_columns=train_data.column_names["train"])

        self.model.set_weights(parameters)

        trainer = transformers.Seq2SeqTrainer(
        model=self.model,
        args=self.training_args,  # Define your training arguments
        train_dataset=train_data,
        data_collator=data_collator
        # Use the same collator from training
        )
        trainer.train()
        return self.model.get_weights(), len(train_data), {}
    

    def evaluate(self, parameters, config):
        eval_data = self.local_dataset["test"]
        eval_data = eval_data.map(prepare_dataset, remove_columns=eval_data.column_names["test"])

        self.model.set_weights(parameters)
        metrics = self.trainer.evaluate(eval_data)
        return metrics["eval_loss"], len(eval_data), metrics
        
fl.client.start_numpy_client(
    server_address="127.0.0.1:5011",
    client = FlowerClient(),
)

