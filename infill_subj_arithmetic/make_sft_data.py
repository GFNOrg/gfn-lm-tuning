import pyarrow as pa
import pandas as pd
import numpy as np

df = pd.read_csv("data/stories.csv")
data = []
data_strs = []

for index in range(100, 1000):
    row = df.iloc[index]
    data_strs.append(row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"] + " " + row["sentence4"] + " " + row["sentence5"])

table = pa.Table.from_arrays([data_strs], names=['text'])

import pyarrow.csv
pa.csv.write_csv(table, 'data/stories_sft.csv', write_options=pa.csv.WriteOptions(include_header=True))
