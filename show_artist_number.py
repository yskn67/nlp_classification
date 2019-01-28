#! /usr/bin/env python

import pandas as pd

df = pd.read_csv('songdata.csv')
adf = df['artist'].value_counts().reset_index()

print(adf.head(50))
