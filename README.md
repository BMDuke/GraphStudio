# Graph Studio <br>
### An experimentation framework for node embeddings. Built in python, using networkx and tensorflow. <br>


### Current bugs and issues

1. `fetch` CLI utility fails to download [molecular sigantures database](https://www.gsea-msigdb.org/gsea/msigdb/)
    - Work around: Manually download [hallmark geneset](https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp#H) from msigDB website, and move it to `app/data/raw/` with the filename `gene_set.entrez.gmt`
2. RAM likely too small on client machine for some processing steps. This mainly affects ETL step where biased walks are generated. The networkx graph is huge! It takes like ~40GB of RAM. This is a point for future work.
    - Work around: Ping me an [email](#Contact) and I will make and send you whatever data assets you need. 
3. `etl process all` has a bug which I haven't identified yet
    - Work around: Process each data asset individually

---

## Contents Table
1. [Project description](#Project-Description)
2. [Features & How it works](#Features-&-How-it-Works)
3. [Requirements](#Requirements)
4. [Installation](#Installation)
5. [Usage](#Usage)
6. [Project Structure](#Project-Structure)
7. [Documentation](#Documentation)
8. [Theory](#Theory)
9. [How to approach model development](#How-to-Approach-Model-Development)
10. [Examples](#Examples)
11. [Contact](#Contact)
12. [Credits](#Credits)

---

## Project Description

### Description

Graph studio is an experiment/reproducibility framework for use in creating node embeddings. It's main contribution is the automatic management of data assets used to train node embedding algorithms using SGD. Particular values used to generate datasets are defined in an `experiment` object, which is recorded in a `config` file and managed using a CLI utility. See the [examples](#Examples) to find out more!<br><br>
This project still has a massive ammount of headroom. There are a heap more features that I would like to include. 

### Inspiration
This project was initiated with my initial curiosity with AI x Biology and learning about bioinformatics. I started out with an open course from [Stanford](http://snap.stanford.edu/deepnetbio-ismb/) which introduced me to some of the current methods used in bioinformatics. From here I began implementing the foundational algorithms and testing them on biological datasets. This was initially in a style not disimillar to a notebook, as you so often see in ML. <br><br> 
The problem with this is that often you find yourself relying on the sequential execution of code. It's also not as regular to persist intermediate data assets during processing, something which quickly became a problem during development. Additionally, I wasnt making full use of the computational resources on my machine. So I had a refactor, and in doing so I have created Graph Studio, a nascent reproducibility/experimentation framework, which aims to catalogue and cache data assets for later training. The fundamental insight here is that

> You can do experiments, both in the model **and** the data.

So far, this project helps the user manage the parameterisation of intermediate data assets by abstracting away their creation, and providing archive utilities. 

[Content table](#Contents-Table)

---

## Features & How it Works

### Data Retrieval
This consists of a [CLI utility](https://github.com/BMDuke/GraphStudio/blob/main/app/source/cli/download.py) which fetches the Biorgrid dataset of protein-protein interations (PPI's) and the MSIGDB Gene Annotation dataset and stores these as raw inputs to the ETL pipeline.<br><br>
**Note**: This is currently semi-functional, see [bugs and issues](#Current-bugs-and-issues) for more details.

### Config & Archive Management
The parameter values which are used to describe both models and data are encapsulated in objects called `experiments`. These are given whatever names you like and consist of key:value pairs which are used in creating derived data products and in training models. This is performed by a [config management tool](https://github.com/BMDuke/GraphStudio/blob/main/app/source/utils/config.py) and users interact with this via a [CLI utility](https://github.com/BMDuke/GraphStudio/blob/main/app/source/cli/config.py).<br><br>
As eluded to a moment ago, some data products are derived from other raw or intermediate data products. This naturally results in dependency hierarchies, which have non-trivial consequences for the final node embeddings. To manage dependencies and prevent duplication of data assets, an [archive manager](https://github.com/BMDuke/GraphStudio/blob/main/app/source/utils/archive.py) keeps track of what data assets exist and what parameters were used to create them. This is achieved by assigning each data asset a UUID which is created from its parameter values. This is a cross cutting utility and is not directly accessible by the user. 

### Extract, Transform, Load (ETL)
There are various steps along the journey from raw data to training dataset. Each step along the way is handled by a [class](https://github.com/BMDuke/GraphStudio/tree/main/app/source/etl) designed expecially for that step. Each class shares a common set of methods:
 - process
 - describe
 - head
 - validate
<!-- end of the list -->
which are presented to the user via a [CLI utility](https://github.com/BMDuke/GraphStudio/blob/main/app/source/cli/etl.py). These methods are designed to perform the processing step, provide reassurance that the result is valid and give the user a peek at the result and any statistics about the data. 

### Train
We combine the data definitions and model definitions by training the model on the data. This is performed by another [CLI utility](https://github.com/BMDuke/GraphStudio/blob/main/app/source/cli/train.py) and is described by `experiments` for a given model. This is organised as three layers. First, there is the CLI layer. Second, there is the [training layer](https://github.com/BMDuke/GraphStudio/tree/main/app/source/train) which aligns the model with the appropriate data asset. Finally, the third layer consists of the [model definitions](https://github.com/BMDuke/GraphStudio/tree/main/models) which are defined by the user. See the [docs](#Documentation) and [examples](#Examples) to see how this works in practice.

[Content table](#Contents-Table)

---

## Requirements
- ~20GB Disk
- ~40GB RAM
- Docker v20.10.0 and up 
- Docker-compose v2.5.0 and up (probably can get away with [v1.28.0+](https://docs.docker.com/compose/gpu-support/))
- An [MsigDB](https://www.gsea-msigdb.org/gsea/msigdb/) account email address 

[Content table](#Contents-Table)

---

## Installation

Although this project is written in python, you don't actually need to worry about installing it or managing any packages because this is all taken care of in the [Docker image](https://github.com/BMDuke/GraphStudio/blob/main/Dockerfile). 

### Docker 
You will need to ensure that you have docker installed. Some useful guides for this can be found here:
 - Official docker [website](https://docs.docker.com/get-docker/)

### Docker compose
You will also need to ensure that you have docker compose installed:
 - Official docker compose [website](https://docs.docker.com/compose/install/)
 - [How to upgrade docker-compose to latest version](https://stackoverflow.com/a/49839172/18272153)

### Nvidia container toolkit
If you want to use GPU accelerated training (and I haven't checked if not doing so works yet), then you will need to install the nvidia container toolkit. This let's the container access any GPU's available on the host machine. :
 - [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

### Clone project
In a working directory of your choice, clone the repo and enter that directory. For example
```
cd ~/Desktop

git clone https://github.com/BMDuke/GraphStudio.git

cd GraphStudio
```

### Building the Dockerfile
Once you've cloned the project, you need to build the Docker image. This requires a fair few heavy libraries and may take some time. 
```
sudo docker build . -t graphstudio
```

### Spin up the container
First you will want to test whether the nvidia-docker installation has been successful. To do this, run
```
sudo docker-compose run test-nvidia
```
and you should get the nvidia-smi print out.  After that, take the container down
```
sudo docker-compose down test-nvidia
``` 
After that you are ready to spin up the container with
```
sudo docker-compose run graphstudio bash
```
and you should now be inside the container with a command prompt something like 
```
root@h93d9dkw:/graphstudio#  
```
Now you're ready to get started embedding some nodes :metal:.

[Content table](#Contents-Table)

---

## Usage

### Fetching the raw data
To get started you are going to need to fetch the raw data. This uses data from two public data repositories for bioinformatics data. These are BioGrid and MsigDB. In order to do this, you need to have created an account with MsigDB, becuase you will need to use the email address registered with the account to download the data. To download the data, call the `fetch` command, passing your email as an argument to the `-e` flag. 
```
fetch -e=your@email.goes.here
```
**Note**: This is currently semi-functional, see [bugs and issues](#Current-bugs-and-issues) for more details.

### Experiments in the data
Before you run the ETL toolkit over the raw data, you will want to have a think about how you want to parameterise your data. To understand how the choice of parameters affects the data, have a look at the explanation in the [theory](#Theory) section. But as an example, suppose we want to generate biased walks with `p equal to .3` and `q equal to .9`. <br><br>
The first thing to do is to create a new config file. This will hold all the experiments you are going to perform in your research. Typically a research project wont need more than one config file.
```
conf new humanPPI
```
Next, you want to create a new experiment for the data. In a real life scenario, you will want to design a schema of parameter combinations so that you can produce a adequate selection of combinations on which you will base your experiments. However, for the sake of this example, suppose we want to run an experiment called `more_out_than_in` with `p=.3` and `q=.9` like we previously said. So, let's do it
```
conf new experiment data more_out_than_in p=.3 q=.9
```
Here, we are saying we want the config utility to create a new **experiment** in the **data** called **more_out_than_in** with parameter values **p=.3** and **q=.9**. <br><br>
**Note**: Dont use hyphens in experiment names, only use underscores. This will cause the type checking functionality to fail downstream.<br><br>
To check that worked lets have a peek at the config to make sure. 
```
conf show
```
If we look at the `data` key and then have a look under the `experiments` header, we can see that, indeed, we have created a new experiment called `more_out_than_in` with the correct values. You will notice that there are heaps of other experiment configurations in the file. They are all **default** configurations. These are the default values that would be used to process the data and the default training combinations. Also, note that there is the `current` field under the `data` key. This is the experiment that will be used by the ETL toolkit, unless specified otherwise. <br><br>
We'd better change it incase we forget to tell the ETL tools what experiment we want to use 
```
conf use experiment data more_out_than_in
```
This is saying to the config utility, **use** the **experiment** called **more_out_than_in** as a description for the **data**. Let's check that worked
```
conf show
```
Now, the current experiment specified for the data is `more_out_than_in`. Super.


### Processing the data with the ETL toolkit
Now we are ready to process the raw data and produce our training data. For a full breakdown of the processing steps and the order they must occur in, see the [docs](#Documentation). <br><br>
The first thing we need to do is select PPI's from the Biogrid dataset that are between humans only. Then select the format of the gene identifier that we want to be working with for this project (this is Entrez gene ID's btw). Luckily, all this is taken care of by the [BioGrid ETL utility](https://github.com/BMDuke/GraphStudio/blob/main/app/source/etl/biogrid.py). So all we have to do is tell the ETL CLI tool to process the biogrid data and were sweet. 
```
etl process biogrid
```
Now, this should have loaded in the raw BioGrid data, processed it and saved the result for us to use later. Not only that, but it should be archived and given a unique identifier based on the values of the parameters we have used in the experiment definition `more_out_than_in`. Lets have a look 
```
etl ls
```
According to me, this has the id `0df726c9cc7bd8`. Additionally, we can see from the table that the values of .3 and .9 were used for p and q respectively. Good stuff. <br><br>
Ok, but what does the processed data look like? Lets have a peek
```
etl head biogrid
```
Looking good. Now, I wonder how many connections there are in this network. And how many objects there are interacting. Let's see what we can find out. 
```
etl describe biogrid
```
Ok, that's heaps of info. Let's chew the fat on that another time. Better check there are no missing values or any other weird things going on in this dataset.
```
etl validate biogrid
```
Ok, sweet as, we're good to move onto the next step. 
<br>
The next data asset we needs to make is the transition probability graph for the PPI network. But that might take a while. And I want to finish writing this documentation before I go to bed tonight, so we might take a quick shortcut. **BUT** its important that you know that we can repeat the same steps illustrated above for any of the data assets we need to make in this project. Each one can be **processed**, **inspected** (head), **described** and **validated**. But for the sake of brevity, we are moving right along. 
```
etl process all
```
 
### Experiments in the model
Now that we have processed all the data, we are ready to train some models on it. If you have a peek in the [models](https://github.com/BMDuke/GraphStudio/tree/main/models) directory, you will see that there are subdirectories that correspond to various different classes of models. At present, these are `binaryClasifier`, `multiLabelClassifier` and `node2vec`. The directory `node2vec/` is the one where we define the various embedding architectures. A look in there will reveal that there is a `default.py` script which contains the definition for the default embedding architecture. We will use the default architecture to start with. <br><br>
A key point here, is that trained embedding models are dependent on both the **architecture** and the **data**. And trained multi-label classifiers and binary classifiers are dependent on their **architecture**, a particular **encoder** (trained embedder) and the **data**. Again, here we have a hierarchy of dependencies which we must manage, becuase it has non-trivial consequences for the final model. <br><br>
So to do this we again turn to our configuration file to manage these for us by defining experiments. Lets start by defining an experiment to train the encoder (node2vec) and telling the config manager to use it
```
conf new experiment node2vec my_first_encoder data=more_out_than_in epochs=1

conf use experiment node2vec my_first_encoder

conf show
```
Sweet, that looks about right. Here, we are telling the config utility to create a **new experiment** for **node2vec** called **my_first_encoder**, and that it should be trained on the data described by the experiment **more_out_than_in** for 1 **epoch**. <br>
**Important**: the value you pass to the data argument, is the **name of the experiment** that was used to generate the data. <br>
Then we are telling config to use that node2vec experiment unless otherwise specified. <br><br>
While were here, we could also define an expriment for a model which uses the trained encoder to embed its inputs. Let's create a new multi-label classifier experiment
```
conf new experiment multiLabelClassifier mlc_from_vec encoder=my_first_encoder

conf use experiment multiLabelClassifier mlc_from_vec
```
Here, we are telling the config to create a **new experiment** which is for a  **multiLabelClassifier** called **mlc_from_vec** which uses a trained encoder, described by the experiment **my_first_encoder**. <br>
**Important**: the value you pass to the encoder argument, is the **name of the experiment** that was used to train the encoder. <br>


### User defined models
Looking in the projects root directory, there are two main subdirrectories called `app/` and `models/`. The `app/` subdirectory contains all the source code for the project and any data assets and model weights which are generated. User's aren't expected to wade through here if they dont want to. Rather, users can put their model definitions in the `models/` directory under the appropriate subdir. These are then made available to the app when the docker container runs. This allows the user to include various architectures using a plugin-style pattern. <br><br>
We have previously seen the default architectures. The default architectures are alright for illustrative purposes, but it's likely that you will want to define your own custom architecture to experiment with creating information dense embeddings and other models. Supposing you have designed a new encoder architecture, called deep_encoder. This inherits from `tensorflow.keras.Model` and describes the computation in the `call()` method. For illustration lets pretend we have a deep_encoder (we will really just copy the default, but for illustration...). Outside of the docker container, in the host environment
```
cp models/node2vec/default.py models/node2vec/deep_encoder.py
```
Now fire the container up and tell the config manager about the new architecture by identifying it in an experiment
```
sudo docker-compose run graphstudio bash

conf new experiment node2vec custom_encoder architecture=deep_encoder data=more_out_than_in

conf use experiment node2vec custom_encoder

conf show
```

Here, we have told the config manager to create a **new experiment** for the **node2vec** embedding model, called **custom_encoder**. We have told it to use the architecture defined in the  **deep_encoder** file and to train it on the dataset described by the **more_out_than_in** experiment. <br>
**Note**: The value of `architecture` must be the same as the filename given to the model definition. This is how the plugin architecture works.<br>
Now we are ready to train the model.

### Training a model
All the hard work has now been done, we are now ready to train the model. This is simply done with the command
```
train node2vec
```
After training, the model is saved along with its weights and checkpoints. Like the data assets, it is saved under a UUID which is generated based on the parameters of the data it is trained on, the architecture used and training conditions used. To have a look at what models have been trained and the parameters that were used to train them
```
train ls
```
[Content table](#Contents-Table)

---

## Project Structure
The project is structured so that it has two main sections. In the `GraphStudio/` root dir, there are two subdirectories.The `app/` subdirectory contains all the source code for the project and any data assets and model weights which are generated. User's aren't expected to wade through here if they dont want to. Rather, users can put their model definitions in the `models/` directory under the appropriate subdir. This is the project structure in the **host context**<br><br>
When you spin up the container, the project has a slightly different structure. By looking through the [Dockerfile](https://github.com/BMDuke/GraphStudio/blob/main/Dockerfile), you can see that the project stucture in the **container context** is as if `app/` was the working directory in the host context. It just copies the contents of `app/` to the working directory in the container.
<br>

### Data persistence
By looking through the [docker-compose file](https://github.com/BMDuke/GraphStudio/blob/main/docker-compose.yml), we see that the the `cache/`, `conf/` and `data/` directories on the host machine are mounted into the container. This means that all the all data generated within the container are persisted on the host machine. 

### Custom models
We can also see that each of the subdirectories in the `models/` directory on the host machine are mounted to the corresponding subdirectory in the `app/source/models/` directory in the container. This provides users with an easy and less-error prone way to include custom architectures. <br>
**Note**: This means that you can have your custom architecture open in a text editor on your host machine and make changes which will instantly be reflected in the container environment. This may be useful for model prototyping and development.<br>
**Important**: Make sure you create custom models in the host context, because if you create the file within the container context it will be read-only on the host machine. 

[Content table](#Contents-Table)

---

## Documentation

### fetch
This is a CLI utility which fetches data from two public bioinformatics repositories, [BioGrid](https://thebiogrid.org/) and [MsigDB](https://www.gsea-msigdb.org/gsea/msigdb/). It fetches both datasets at once, so they can't be retrieved individually. <br>
**Important**: The email address that is assocuated with your MsigDB account is required to download the data.<br><br>
From within the container
```
fetch -e=<your email> [-v=<biogrid version>]

---------------------------------------------
Args:

    -e              Email address associated with MsigDB 
    --email         account. Required.

    -v              Biogrid version to download. Default is 
    --version       4.4.207. Optional.
```

### conf
This is a CLI utility that allows users to define experiments that desribe how to process data assets and train models. <br><br>
From within the container

```
conf new [experiment] [asset] name [**params]

---------------------------------------------
Creates a new config file or experiment

Args:

    experiment      Directive which indicates to the config 
                    manager whether the user wants to create 
                    a new config file or an experiment. This is 
                    passed as a literal 'experiment'. If this 
                    is provided then an asset must also be given 
                    to indicate what data/model the experiment
                    is for. If this is omitted, a new config
                    file will be created.
    
    asset           Name of the project resource that the 
                    experiment is describing. This must match 
                    top level keys in the config file eg.
                    node2vec, multiLabelClassifier etc...
                    
    name            The name of the config file / experiment. 
                    This shouldn't contain hyphens. Required.
    
    params          This then accepts a variable number of 
                    kwargs. These should be the keys associated
                    with the resource being created. They need the
                    format 'key=value' for example 'p=.5'.



conf use [experiment] [asset] name 

---------------------------------------------
Switches between active config file / experiment.

Args:

    experiment      Directive which indicates to the config 
                    manager whether the user wants to switch to 
                    a particular config file or a particular 
                    experiment within the current config file. 
                    This is passed as a literal 'experiment'.
                    If this is passed then it must be accompanied
                    with a value for asset also. 

    asset           The name of the project resource whose active
                    experiment is being changed. This must match 
                    top level keys in the config file eg.
                    node2vec, multiLabelClassifier etc...

    name            The name of config file / experiment that should
                    be activated. 



conf show [name] 

---------------------------------------------
Prints the current config file or a config file given by 'name'

Args:

    name            The name of the config file to display. Optional.
                    If no name is provided then this defaults to current.



conf current

---------------------------------------------
Prints the name of the current config file



conf edit [experiment] [asset] name [**params] 

---------------------------------------------
Edit the current values of a config file, or an experiment within a config 
file.

Args:

    experiment      Directive which indicates to the config 
                    manager whether the user wants to create 
                    a new config file or an experiment. This is 
                    passed as a literal 'experiment'. If this 
                    is provided then an asset must also be given 
                    to indicate what data/model the experiment
                    is for. If this is omitted modifications will
                    be made to the top level of the config file. 
    
    asset           Name of the project resource that is being
                    modified. This must match top level keys in 
                    the config file eg. node2vec, 
                    multiLabelClassifier etc...
                    
    name            The name of the config file / experiment. 
    
    params          These kwargs of the parameters you would like
                    to edit, along with their new values.They need 
                    the format 'key=value' for example 'p=.5'.    



conf delete [experiment] [asset] name 

---------------------------------------------
Delete a config file, or an experiment within a config file.

Args:

    experiment      Directive which indicates to the config 
                    manager whether the user wants to delete 
                    a particular config file or a particular 
                    experiment within the current config file. 
                    This is passed as a literal 'experiment'.
                    If this is passed then it must be accompanied
                    with a value for asset also. 

    asset           The name of the project resource to be deeted.
                    This must match top level keys in the config 
                    file eg. node2vec, multiLabelClassifier etc...

    name            The name of config file / experiment that should
                    be deleted. 
                      



```
### etl
This is a CLI utility that allows users to process the raw data into derrived data assets and finally training data sets. There is an order in which data assets must be produced as dependencies exist among them. Here, you can see the order in which they should be created, along with the parameter values that have been used to create them. <br><br>
Raw Data:<br>
| Name | Description | Dependencies |
| ---------- | ---------- | ---------- |
| MsigDB | Gene annotations. Used to created a multi-labelled dataset | None |
| Biogrid | Protein-protein dataset. Used to build interaction graph | None |

Processed Data:<br>
| Name | Description | Dependencies |
| ---------- | ---------- | ---------- |
| Gene IDs | This is a processed version of the MSigDB dataset. | None |
| Biogrid | Processes the raw biogrid data so that it only contains human-human interactions and gene id format is entrez | version |
| Transition Probabilties | Precalculated transition probabilities required to sample biased walks from the PPI graph. This is stored as a pickled nx graph and has a massive memory footprint. | version, p, q |
| walks | This is a dataset consisting of the biased walks sampled from the PPI graph. This indicates the topology of the graph. The nodes that are visited at each step of the walk are selected based on the transition probabilities computed at the previous step. | version, p, q, num_walks, walk_length |
| Skipgrams | This is a training dataset used to train the encoder for node embeddings. It is a TFRecords dataset that contains positive and negative skipgrams generated from the random walks  | version, p, q, num_walks, walk_length, negative_samples, window_size |
| Vertices | This is a training dataset used to train a multi-label classifier which takes entres gene IDs as inputs. It is a TFRecords dataset. | None |
| Edges | Coming soon... | version |

A more complete description of what each of the parameters to can be found in the [theory]() section, but a brief definition of the responsibilities of the parameters is given here.<br>
Parameters:<br>
| Name | Description |
| ---------- | ---------- |
| version | The version of the biogrid dataset that was used to build the PPI graph |
| p | Return parameter. Used to generate biased walks. Represents the liklihood of immediately revisiting an earlier node in the walk |
| q | In-out parameter. Used to generate biased walks. Controls whether the agent explores in a more BFS-like (q > 1) or DFS-like (q < 1) manner |
| num_walks | The number of times a biased walk should be initiated from each node |
| walk_length | The number of steps taken for each walk |
| window_size | The size of the window from which skipgrams are generated |
| negative_samples | The number of negative samples that should be generated for each the positive sample. This is used to optimise training. |

<br>

From within the container
```
etl process dataset[=experiment_name]

---------------------------------------------
Process a given data resource.

Args:

    dataset             The data resource to process. This will be one of the ones
                        given in the names column of the processed data table. This 
                        also takes a special value 'all' which will attempt to process
                        all data resources for a given experiment.

    experiment_name     If provided, it specifies which experiment definition
                        should be used to process the data. If this is not provided
                        then whatever experiment is provided by the 'current' key in 
                        the config file will be used.



etl describe dataset[=experiment_name]

---------------------------------------------
Describe a given data resource.

Args:

    dataset             The data resource to describe. This will be one of the ones
                        given in the names column of the processed data table.

    experiment_name     If provided, it specifies which experiment definition
                        should be used to describe the data. If this is not provided
                        then whatever experiment is provided by the 'current' key in 
                        the config file will be used.
                    


etl head dataset[=experiment_name] [n=value]

---------------------------------------------
Print the top n rows of a given data resource.

Args:

    dataset             The data resource to display. This will be one of the ones
                        given in the names column of the processed data table.

    experiment_name     If provided, it specifies which experiment definition
                        for the data to be displayed. If this is not provided
                        then whatever experiment is provided by the 'current' key in 
                        the config file will be used.                   

    n                   The number of records to be printed. Default is 5. Optional.



etl validate dataset[=experiment_name]

---------------------------------------------
Validate a given data resource.

Args:

    dataset             The data resource to validate. This will be one of the ones
                        given in the names column of the processed data table.

    experiment_name     If provided, it specifies which experiment definition
                        for the data to be validated. If this is not provided
                        then whatever experiment is provided by the 'current' key in 
                        the config file will be used.    



etl ls

---------------------------------------------
Display a table of all the data resources that have been created, their UUIDs, and the parameter values that were used to create them.

```

### train
This is a CLI utility that is used to train the models. Because all the details are handled by the configuration utility, this has a very simple interface.<br><br>
```
train train dataset[=experiment_name]

---------------------------------------------
train a given model. The syntax here could do with a tweak.

Args:

    dataset             The model to be trained. This will be one of:
                        node2vec:       for node2vec
                        mlc:            for multi-label classifier
                        bc:             for binary classifier

    experiment_name     If provided, it specifies which experiment definition
                        should be used to define the model. If this is not provided
                        then whatever experiment is provided by the 'current' key in 
                        the config file will be used.



train ls

---------------------------------------------
Display a table of all the models that have been trained, their UUIDs, and the parameter values that were used to create them.

```

### Data formats for custom models
#### Node2vec
The exact definition of the data for training the node2vec encoder will vary depending on the number of negative samples you decide to use. However, the basic structure of a training example will be something like this.
```
(
    x = {
        'target': tf.Tensor(shape=(1,)),
        'context': tf.Tensor(shape=(negative_samples + 1,))
    },

    y = tf.Tensor(shape=(negative_samples + 1,))
)
```
So what this means is that in your `call()` method, you should arrange your computation such that target (a scalar) is applied to every element of context (a vector) returning y_pred which is a vector. Einsum is a good way to do this!

#### Multi-label classification
A training example for MLC is something like this
```
(
    x = tf.tensor(shape=(1,)),

    y = tf.tensor(shape=(num_labels,))
)
```
where `num_labels` is the number of hallmarks in the MsigDB database. At the time of development, this was 50. <br><br>
y is a multi-hot encoding of labels. 

[Content table](#Contents-Table)

---

## Theory
I will try and write a comprehensive digest of the theory behind the models used in this project. However untill I do, you can find out more information about the methods used from the following resources. 
 - Snap Stanford: This is a resource produced by the creators of node2vec. This can be found [here](https://snap.stanford.edu/node2vec/).
 - Towards data science: This is is a nice break down of node2vec in an easy to understand way. This can be found [here](https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef)

[Content table](#Contents-Table)

---

## How to Approach Model Development

Coming soon...

[Content table](#Contents-Table)

---

## Examples

Coming soon...

[Content table](#Contents-Table)

---

## Contact

If you want to get in touch to talk about anything you have seen in this project please get in touch! Also I am more than happy to send trained weights and any intermediate data resources if processing them is not possible on your machine. :v:
- duke@bmdanalytics.co.uk
<!-- end of the list -->
[Content table](#Contents-Table)

---

## Credits
This work was inspired by the work done by A. Grover and J. Leskovec at Stanford. 
 - https://arxiv.org/abs/1607.00653
<!-- end of the list -->
I also based by ppi graph implementation and sampling strategy on the work of E. Cohen
 - https://github.com/eliorc/node2vec

[Content table](#Contents-Table)

---

































 <!---

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

--->