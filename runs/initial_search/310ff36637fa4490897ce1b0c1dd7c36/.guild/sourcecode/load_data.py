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
# Since the test data, however, is relatively small, the amount of data removed this way should be neglectable for now
num_tk = int(args.size * args.ratio)
num_ba = int(args.size * (1 - args.ratio))


# query to select a random data by even years 
query_ba = f"""select first_query.id, first_query.ANGEBOTSART,[STELLENBESCHREIBUNG]
FROM 
(SELECT TOP {num_ba} WITH TIES 
a.[id]
,a.[JAHR]
,[ANGEBOTSART]
FROM [stea_ba].[dwh_stg].[ba_sel] a
WHERE [ANGEBOTSART] IN
(
SELECT  DISTINCT [ANGEBOTSART]
  FROM [stea_ba].[dwh_stg].[ba_sel] 
  ) AND [ANGEBOTSART] != 'Auszubildende / Duales Studium' 
ORDER BY 
ROW_NUMBER() OVER (PARTITION By [ANGEBOTSART] ORDER BY [ANGEBOTSART] ),
[ANGEBOTSART])
first_query
join [stea_ba].[dwh_stg].[ba_fulltext] b
on first_query.id = b.id
"""


# get the datasets as individual dataframes 
df_ba = pd.read_sql(query_ba,sql_conn)
df_tk = pd.read_csv("az_tk_data.csv", index_col = 0).sample(n = num_tk).reset_index(drop = True)


# bring the two dfs together
df_ba = df_ba.rename(columns = {"STELLENBESCHREIBUNG" : "text",
                                "ANGEBOTSART" : "label_class"})

df_tk = df_tk.rename(columns = {"full_text" : "text",
                                "true" : "label_class"})


# remove possible duplicates from BA set that are already in the testset 
df_ba = remove_test_data(df_ba)

df = pd.concat([df_ba, df_tk], ignore_index = True)

# compose the dataset with regard to balance_strat and label_strat. see utily.py for details.
df = compose_data(df = df, label_strat = args.label_strat, balance_strat = args.balance_strat)

df.to_csv("input_data.csv")
