import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer)
import evaluate
import argparse
import logging
from utils import get_label_names


torch.cuda.empty_cache()


# loading flags
#_______________________________
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--warmup", type=int)
parser.add_argument("--label_strat", type=str)


args = parser.parse_args()

#_______________________________

# load data from previous step

df = pd.read_csv('input_data.csv', index_col = 0)
num_labels = len(set(df.label.tolist()))
logging.info(f"num_labels: {num_labels}")
label_names = get_label_names(args.label_strat)

# list of available models
available_models = {"bert": "bert-base-multilingual-cased",
                    "gbert" : "bert-base-german-cased",
                    "distilbert" : "distilbert-base-german-cased",
                    "jobbert" : "agne/jobBERT-de",
}


# load model and tokenizer depending on model chosen for the run
MODEL_NAME = available_models[args.model]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#use GPU
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = num_labels).to(device)



def tokenize_function(data):
    """tokenizes the data using the tokenizer passed to the function"""
    return tokenizer(data["text"], padding = "max_length", truncation = True)

# create huggingface dataset and apply tokenization
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# test / eval split 70 / 30
input_dict = tokenized_dataset.train_test_split(test_size = 0.3)

## load metrics
metric1 = evaluate.load("precision")
metric2 = evaluate.load("recall")
metric3 = evaluate.load("accuracy")
metric4 = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Define function to compute metrics for training evaluation. 
    If there are more than 2 labels, precision, recall and f1 need to have the average argument passed onto them, which is 'macro' in this case"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric3.compute(predictions=predictions, references = labels)["accuracy"]
    if num_labels == 2:
        precision = metric1.compute(predictions=predictions, references = labels, zero_division= 0)["precision"]
        recall = metric2.compute(predictions=predictions, references = labels, zero_division = 0)["recall"]
        f1 = metric4.compute(predictions=predictions, references = labels)["f1"]
    else: 
        precision = metric1.compute(predictions=predictions, references = labels, zero_division= 0, average = "macro")["precision"]
        recall = metric2.compute(predictions=predictions, references = labels, zero_division = 0, average = "macro")["recall"]
        f1 = metric4.compute(predictions=predictions, references = labels, average = "macro")["f1"]
    
    return {"precision" : precision, "recall" : recall, "accuracy" : accuracy, "f1" : f1}




# initializing training arguments
training_args = TrainingArguments(
    output_dir = './results',
    num_train_epochs = args.epochs,
    learning_rate = args.lr,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    warmup_steps = args.warmup,
    logging_dir =  './logs',
    logging_strategy = 'epoch',
    evaluation_strategy = 'epoch',
    save_strategy = 'no'
    )


# initializing the trainer instance
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=input_dict["train"],       
    eval_dataset=input_dict["test"],
    tokenizer = tokenizer,
    compute_metrics=compute_metrics              
)

trainer.train()

trainer.save_model("C:/Users/Admin/Desktop/dev/kk/azb_klassifizierer/experiment_saves/current_model")