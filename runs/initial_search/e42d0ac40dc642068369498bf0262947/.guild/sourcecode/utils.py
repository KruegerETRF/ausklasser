import pandas as pd
import logging
from collections import Counter
from torch import nn
from transformers import Trainer



def remove_test_data(df):
    """Takes a Dataframe and checks if any of the texts is already in the test set. If it is, the function removes that text.
    Returns the cleaned Dataframe. 
    It finally logs the amounts of text removed that way."""
    test_df = pd.read_csv("test_data.csv")
    orig_len = len(df)
    test_texts = test_df["text"].tolist()
    df = df[~df['text'].isin(test_texts)]
    if orig_len != len(df):
        logging.warning(f"Warning: {orig_len - len(df)} ad(s) have been removed, because they are already in the test set.")


    return df



def compose_data(df, label_strat, balance_strat):
    if label_strat == "binary": 
        df["label"] = [0 if lc in ('Auszubildende','Azubi') else 1 for lc in df.label_class.tolist()]
        if balance_strat == "no_balance":
            log_counts(df)
            return (df)
        elif balance_strat == "downsample":
            df = downsample(df)
            log_counts(df)
            return (df)
        elif balance_strat == "oversample":
            df = oversample(df)
            log_counts(df)
            return (df)
        else:
            logging.error ("No valid balance strat has been given.")
            pass
    elif label_strat == "multiclass": 
        df = compose_multiclasses(df)
        if balance_strat == "no_balance":
            log_counts(df)
            return (df)
        elif balance_strat == "downsample":
            df = downsample(df)
            log_counts(df)
            return (df)
        elif balance_strat == "oversample":
            df = oversample(df)
            log_counts(df)
            return (df)
        else:
            logging.error ("No valid balance strat has been given.")
            pass
    else:
        logging.error ("No valid label_start has been given")
        pass 



def log_counts(df):
    counts = Counter(df["label"])
    logging.info(f"Final Distribution of Labels: {counts} ")

def downsample(df): 
    """This funtion returns a Dataframe with a balanced amount of class values equal to the amount of values for the smallest class. 
    Sample is drawn random from random_state = 100 for reproducability."""
    group = df.groupby('label')
    df = group.apply(lambda x: x.sample(group.size().min(), random_state = 100)).reset_index(drop=True)
    return (df)


def oversample(df): 
    """This funtion returns a Dataframe with a balanced amount of class values equal to the amount of values for the largest class.
    Oversampling is down by simply copying values randomly. 
    Random state is 100 for reproducability."""
    group = df.groupby('label')
    df = group.apply(lambda x: x.sample(group.size().max(), replace = True, random_state = 100)).reset_index(drop=True)
    return (df)


def compose_multiclasses(df):
    label_mapping = {
        'Auszubildende' : 0,
        'Azubi' : 0,
        '- nur Helfer' : 1,
        'nur Helfer' : 1, 
        'Praktikum' : 1,
        'SHK' : 1,
        'Trainee' : 1,
        'Studium' : 1,
        'FSJ' : 1,
        'Dual' : 1,
        '- nur Führungskräfte' : 2,
        'nur Führungskräfte' : 2,
        'NoAz' : 3,
        'Arbeitskräfte' : 3,
        '- nur Fachkräfte' : 3,
        'nur Fachkräfte' : 3
    }
    df["label"] = [label_mapping[lc] for lc in df.label_class.tolist()]
    return(df)



def get_label_names(label_strat):
    if label_strat == "binary":
        names = ["Auszubildende", "Sonstige Arbeitnehmer"]
    elif label_strat == "multiclass":
        names = ["Auszubildende", "Verschiedenes", "Führungskräfte", "Fach- und Arbeitskräfte"]
    
    logging.info(f"Detected label names: {names}")
    return (names)



#to do: longtext? 
#to do: stresstest Dataset? (z.B. Katharinas, ...Suchen Keine Azubi...., sehr lange, )