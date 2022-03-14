# node2vec for human PPI data
An implementation of the node2vec algorithm for learning node embeddings in python.

---

## Quickstart


### Download the data
To download the data for the project, you will need to have registered an account with [GSEA/MSigDB](http://www.gsea-msigdb.org/gsea/register.jsp). The account is free and you do not need a password - it's so that they can understand who is using the database. 
Make sure you are in the root project directory. Then run the following command

```
.../node2vec$ python run/data_download.py -e=<your@email.com>
```
**Update required**

This will download a protein-protein interaction database from BioGrid, and a gene hallmark dataset which labels proteins with functional annotations. You will need to supply the email address, using the `-e / --email` flag, you used to create your GSEA account with.
If you would like to download a specific version of the BioGrid dataset, you can specify this with the `-v / --version` flag. For example,

```
.../node2vec$ python run/data_download.py -e=<your@email.com> -v=2.3.103
```
**Update required**

BioGrid releases can be found [here](https://downloads.thebiogrid.org/BioGRID/Release-Archive/). 

## Project structure
All commands should be run from the root directory `/node2vec-v2/`. There are the following subdirectories:

### `/models`
This contains the definition of all the models that are used in the project. They are broadly categorised into embedding models and prediction models. 
- Embedding models: These are models which are involved in creating the node embeddings. This includes the node2vec implementation, the word2vec implementation and the encoder. The encoder is a block which uses the trained weights from the node2vec implementation, such that it can be included in other models. 
- Prediction models: Thsi includes the multi-label classifier which is used for node classification, and a binary classifier which is used for edge prediction. 

### `/data`
This contains human PPI data that is downloaded from BioGrid. The data can be found [here](https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.207/). The information about the raw data can be found [here](https://wiki.thebiogrid.org/doku.php/biogrid_tab_version_3.0). 

### `/utils`
This contains any utility functions for the project.

### `/cache`
This contains all cached models, weights, graphs and datasets which are produced while the project is run. These are designed to minimise computational costs, and streamline exploration of parameters during development. This contain an archive of trained models. 

### `/run`
This folder contains the scripts which orchestrate the project. 
- `download.py`
- `etl.py`
- `config.py`
- `train.py`

### `/test`
This folder contains all the tests used to verify the accuracy of the code that is being used in this project. 

### `/.dev`
This folder contains any scraps of code that have been used in development. None of this is intended for external use.

---

# Documentation

