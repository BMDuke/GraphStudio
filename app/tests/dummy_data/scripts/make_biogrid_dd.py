from dataclasses import replace
import os

import pandas as pd

from source.utils.config import Config

raw_dir = 'data/raw'
organism = 'Homo sapiens'
interactor_a = 'Entrez Gene Interactor A'
interactor_b = 'Entrez Gene Interactor B'  
organism_a = 'Organism Name Interactor A'
organism_b = 'Organism Name Interactor B'
chunksize = 10000

out = 'tests/dummy_data/biogrid.csv'

config = Config()

# Make sure current raw biogrid data exists
version  = config.show()['data']['version']
raw_filepath = os.path.join(raw_dir, f"{version}.txt")

assert os.path.exists(raw_filepath), "ERROR: Raw BioGrid data version {version} not found - Have you downloaded the data?"

dataframe = pd.DataFrame()

with pd.read_table(raw_filepath, chunksize=chunksize) as data_in:
    for chunk in data_in:
        print('Reading...')
        frames = [dataframe, chunk]
        dataframe = pd.concat(frames, ignore_index=True)

dataframe = dataframe.groupby(organism_a).sample(n=100, replace=True)
dataframe.to_csv(out, index=False)
print(dataframe)