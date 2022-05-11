#!/usr/bin/python

'''

Documentation:  

Initialisation script to create project directories in container. 
Directories that need to be created are:
> data/raw/...
> data/processed/...

'''

import os

# Create a list of directories
directories = [
    'data/raw/',
    'data/processed/biogrid',
    'data/processed/edges',
    'data/processed/gene_ids',
    'data/processed/skipgrams',
    'data/processed/transition_probs',
    'data/processed/vertices',
    'data/processed/walks',
]

# Make dirs
for directory in directories:
    os.makedirs(directory, exist_ok=True)

