#! /usr/bin/env python

import click
import pandas as pd

@click.command()
@click.option('--num', default=50, help='Number of artist')
def main(num):
    df = pd.read_csv('songdata.csv')
    adf = df['artist'].value_counts().sort_values(ascending=False)
    print(adf.head(num))

if __name__ == '__main__':
    main()
