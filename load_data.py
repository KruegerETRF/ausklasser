# load imports
import argparse
import pandas as pd
import pyodbc
from utils import remove_test_data, compose_data

# loading flags
#_______________________________
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=1000)
parser.add_argument("--ratio", type=float, default=0.5)
parser.add_argument("--label_strat", type=str, default="binary")
parser.add_argument("--balance_strat", type=str, default="no_balance")

args = parser.parse_args()

#_______________________________

# setup sql connection
sql_conn = pyodbc.connect("DRIVER={SQL Server};\
                            SERVER=DESKTOP-IQD0VQR; \
                            DATABASE=stea_ba; \
                            Trusted_Connection=yes;") 

# determin, how much data each dataset should contain based on the size and ratio flags
# the current implementation is not clean, because it will check and remove ads that are in the test set AFTER this,
# so the final numbers may differ (will however be logged).
# Since the test data, however, is relatively small, the amount of data removed this way should be neglectable.
num_d1 = int(args.size * args.ratio)
num_d2 = int(args.size * (1 - args.ratio))


# pseudo code
# access data from d1 based on num_d1 as df
# access data from d2 based on num_d2 as df
# harmonize column names
# run remove_test_data
# concatinate both dfs as df


# compose the dataset with regard to balance_strat and label_strat. see utily.py for details.
df = compose_data(df = df, label_strat = args.label_strat, balance_strat = args.balance_strat)

df.to_csv("input_data.csv")
