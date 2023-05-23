import pandas as pd
import numpy as np
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          pipeline)
from datasets import Dataset
import argparse
import torch
import logging
import evaluate
from utils import (get_label_names)

# loading flags
#_______________________________
parser = argparse.ArgumentParser()
parser.add_argument("--label_strat", type=str)
args = parser.parse_args()
#_______________________________
#load labels
label_names = get_label_names(args.label_strat)
num_labels = len(label_names)

#load in data
test_df = pd.read_csv("test_data.csv")

# handle cuda
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# load in pretrained model
model_path = "C:/Users/Admin/Desktop/dev/kk/azb_klassifizierer/experiment_saves/current_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = num_labels).to(device)

# construct classifier
classifier = pipeline(model=model, tokenizer=tokenizer, task="text-classification", device=0)
# device=0 zeigt an, dass es auf die erste GPU geladen werden soll. -1 würde CPU bedeuten. 
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}


#------------------------------------------------------------------
# Define helper functions. Potentially outsource to utils.py, but maybe less efficient
def get_preds(classifier, df, num_labels, tokenizer_kwargs):
    dataset = Dataset.from_pandas(test_df)
    y_preds = []
    scores = []
    for pred in classifier(dataset["text"], **tokenizer_kwargs):
        y_preds.append(pred["label"][-1])
        scores.append(pred["score"])
    
    
    df["scores"] = scores
    if num_labels == 2:
        df["y_preds"] = y_preds
    else:
        df["y_preds"] = [pred if pred == '0' else 1 for pred in y_preds]
    
    df["y_preds"] = df["y_preds"].apply(pd.to_numeric)

    return(df)

metric1 = evaluate.load("precision")
metric2 = evaluate.load("recall")
metric3 = evaluate.load("accuracy")
metric4 = evaluate.load("f1")

def compute_metrics(predictions, labels):
    accuracy = metric3.compute(predictions=predictions, references = labels)["accuracy"]
    precision = metric1.compute(predictions=predictions, references = labels, zero_division= 0, pos_label= 0)["precision"]
    recall = metric2.compute(predictions=predictions, references = labels, zero_division = 0, pos_label= 0)["recall"]
    f1 = metric4.compute(predictions=predictions, references = labels, pos_label= 0)["f1"]

    return {"precision" : precision, 
            "recall" : recall, 
            "accuracy" : accuracy, 
            "f1" : f1}
#---------------------------------------------------------------------

# get predictions and scores
# Printing to match Output Scalars in Guild File
# For the entire Data
test_df = get_preds(classifier=classifier, df = test_df, num_labels=num_labels, tokenizer_kwargs=tokenizer_kwargs)
try:
    metrics_all = compute_metrics(predictions=test_df["y_preds"].tolist(), 
                                labels = test_df["y_true"].tolist())
    print(f"all_f1: {metrics_all['f1']}")
    print(f"all_recall: {metrics_all['recall']}")
    print(f"all_precision: {metrics_all['precision']}")
    print(f"all_accuracy: {metrics_all['accuracy']}")
except Exception as e:
    logging.error(f"""Error: Could not log metrics for the entire dataset.\n
                  The following error occured: \n
                  {e}""")
    
# For tk data only
try:   
    metrics_tk = compute_metrics(predictions=test_df[test_df["source"] == "tk"]["y_preds"].tolist(),
                                labels = test_df[test_df["source"] == "tk"]["y_true"].tolist())
    print(f"tk_f1: {metrics_all['f1']}")
    print(f"tk_recall: {metrics_all['recall']}")
    print(f"tk_precision: {metrics_all['precision']}")
    print(f"tk_accuracy: {metrics_all['accuracy']}")
except Exception as e:
    logging.error(f"""Error: Could not log metrics for the tk dataset.\n
                  The following error occured: \n
                  {e}""")
    
try:
    metrics_ba = compute_metrics(predictions=test_df[test_df["source"] == "ba"]["y_preds"].tolist(),
                                labels = test_df[test_df["source"] == "ba"]["y_true"].tolist())
    print(f"ba_f1: {metrics_ba['f1']}")
    print(f"ba_recall: {metrics_ba['recall']}")
    print(f"ba_precision: {metrics_ba['precision']}")
    print(f"ba_accuracy: {metrics_ba['accuracy']}")
except Exception as e:
    logging.error(f"""Error: Could not log metrics for the ba dataset.\n
                  The following error occured: \n
                  {e}""")


# For long texts
try:
    metrics_len = compute_metrics(predictions=test_df[test_df["num_tokens"] >  512]["y_preds"].tolist(),
                                labels = test_df[test_df["num_tokens"] >  512]["y_true"].tolist())
    print(f"len_f1: {metrics_len['f1']}")
    print(f"len_recall: {metrics_len['recall']}")
    print(f"len_precision: {metrics_len['precision']}")
    print(f"len_accuracy: {metrics_len['accuracy']}")
except Exception as e:
    logging.error(f"""Error: Could not log metrics for the long texts.\n
                  The following error occured: \n
                  {e}""")




