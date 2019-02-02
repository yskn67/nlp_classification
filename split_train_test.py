#! /usr/bin/env python

import click
import numpy as np
import pandas as pd


@click.command()
@click.option('--nclass', default=5, help='Number of class')
@click.option('--test_ratio', default=0.2, help='Test data ratio')
def main(nclass, test_ratio):
    df = pd.read_csv('songdata.csv')
    target_artists = df['artist'].value_counts().sort_values(ascending=False).index.values[:nclass]
    df = df[df['artist'].isin(target_artists)][['artist', 'text']]

    train = []
    test = []
    for artist in target_artists:
        data = np.random.permutation(df[df['artist'] == artist].values)
        threshold = int(len(data) * test_ratio)
        train.extend(data.tolist()[threshold:])
        test.extend(data.tolist()[:threshold])
    np.random.shuffle(train)
    np.random.shuffle(test)
    pd.DataFrame(train, columns=['artist', 'text']).to_csv('train.csv', index=False)
    pd.DataFrame(test, columns=['artist', 'text']).to_csv('test.csv', index=False)


if __name__ == '__main__':
    main()
