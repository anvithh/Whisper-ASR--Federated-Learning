from dataclasses import dataclass
from typing import Any,Dict,List,Union
import torch
from datasets import load_dataset, DatasetDict

common_voice = load_dataset(
    'csv', data_files={
        'train': './dup.csv', 
        'test':'./duptest.csv'
    }
)
print(common_voice)
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer


# - Load Feature extractor: WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# - Load Tokenizer: WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Tamil", task="transcribe")
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Tamil", task="transcribe")
print('| Check the random audio example from Common Voice dataset to see what form the data is in:')
print(f'{common_voice["train"][0]}\n')

# -> (1): Downsample from 48kHZ to 16kHZ

from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print('| Check the effect of downsampling:')
print(f'{common_voice["train"][0]}\n')
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
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
import evaluate
metric = evaluate.load("wer")
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    # output_dir="./tamilnew",
    # evaluation_strategy="steps",
    # eval_steps=100,
    # per_device_train_batch_size=2,  # Adjusted batch size for your specific model
    # per_device_eval_batch_size=2,  # Adjusted eval batch size
    # save_steps=1000,
    # save_total_limit=2,
    # num_train_epochs=10,
    # logging_dir="./logs",
    # report_to=["tensorboard"],
    # load_best_model_at_end=True,
    # metric_for_best_model="wer",
    # greater_is_better=False,
    output_dir="./tamilnew2",
evaluation_strategy="steps",
eval_steps=100,
per_device_train_batch_size=2,
per_device_eval_batch_size=2,
save_steps=1000,
save_total_limit=2,
num_train_epochs=10,
logging_dir="./logs",
logging_steps=50,  # Adjusted to log more frequently
report_to=["tensorboard"],
load_best_model_at_end=True,
metric_for_best_model="wer",
greater_is_better=False,
logging_first_step=True,
)
from transformers import Seq2SeqTrainer
def compute_metrics(pred):
    """
    Define evaluation metrics. We will use the Word Error Rate (WER) metric.
    For more information, check:
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
trainer = Seq2SeqTrainer(
    model=model,                 # The instantiated Transformers model to be trained
    args=training_args,          # Training arguments, defined above
    train_dataset=common_voice["train"],  # Your training dataset
    eval_dataset=common_voice["test"],    # Your evaluation dataset
    data_collator=data_collator,  # Your data collator (if needed)
    compute_metrics=compute_metrics,  # Your custom metrics function (if needed)
    tokenizer=processor.feature_extractor,  # Your tokenizer
)
print('Training is started.')
trainer.train()  # <-- !!! Here the training starting !!!
print('Training is finished.')
trainer.save_model("./tamilnew2")
tokenizer.save_pretrained("./tamilnew2")
