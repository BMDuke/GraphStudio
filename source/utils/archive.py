import os
import hashlib

import pandas as pd

from source.utils.config import Config

class Archive(object):
    '''
    This manages the look ups for the storage and retrieval 
    of datasets created in the project. This works in tandem 
    with the config tool to create an identification system
    for data assets created with given parameters. 

    It takes a config file on instantiation which it will use to 
    retrieve values


    '''

    data_dir = 'data/processed'
    filename = 'lookup.csv'

    hash_length = 7

    def __init__(self, config, lookup_fp=None, alt_dir=None, prune_file=True):
        self.config = config
        self.lookup_fp = lookup_fp # Used to create seperate lookup for testing

        if alt_dir:
            self.data_dir = alt_dir # Use alternative data directory if specified. 

        self._init_lookup()
        self._prune_lookup(prune_file=prune_file)

    def make_id(self, type, *params, add_to_lookup=False):
        '''
        Creates a hash ID based on parameter values. 
        Optionally adds it to the lookup table.

        DEPRECIATED

        '''
        id = self._make_hash(*params)

        if add_to_lookup:

            config = self.config.show()
            current = config['data']['current']
            experiment = config['data']['experiments'][current]

            params = {
                'version':config['data']['version'],
                'p': experiment['p'],
                'q': experiment['q'],
                'walk_length': experiment['walk_length'],
                'num_walks': experiment['num_walks'],
                'window_size': experiment['window_size'],
                'negative_samples': experiment['negative_samples']
            }

            self._add_to_lookup(id, type, params)
        
        return id

    def make_id_from_dict(self, asset, values, add_to_lookup=False):
        '''
        A new and improved version of make_id which allows you to generate
        ID's from parameter values directly.
        
        The alternative was passing parameter names and having the indirection
        of making the archive look up those parameter values. It also 
        allows you to make ID's for experiments other than the current ones
        in the config file.
        
        This is paired with the new method ._make_hash_from_dict, verbose,
        but avoid problems described above.
        '''

        id = self._make_hash_from_dict(asset, values)

        if add_to_lookup:

            self._add_to_lookup(id, asset, values)
        
        return id

    def lookup_id(self, id):
        '''
        Looks up a given file ID and returns its parameters and filepath
        '''
        return self._get_from_lookup(id)

    def _init_lookup(self):
        '''
        If the lookup table hasnt been created yet, make it
        '''

        filepath = self._get_lookup_filepath()

        try:

            if not os.path.exists(filepath):
                columns = ['id', 'type', 'version', 'p', 'q', 
                            'walk_length', 'num_walks', 'window_size', 
                            'negative_samples']
                df = pd.DataFrame(columns=columns)
                self._save_lookup(df)
        
        except Exception as e:

            print(f"ERROR: {e}")

    def _load_lookup(self):
        '''
        Loads the lookup
        '''

        filepath = self._get_lookup_filepath()

        try:

            lookup = pd.read_csv(filepath, index_col=False)
            return lookup
        
        except Exception as e:

            print(f"ERROR: {e}")
    
    def _save_lookup(self, lookup):
        '''
        Saves the lookup file
        '''
        filepath = self._get_lookup_filepath()

        try:

            lookup.to_csv(filepath, index=False)
            return lookup
        
        except Exception as e:

            print(f"ERROR: {e}")   


    def _add_to_lookup(self, id, type, params):
        '''
        Add an item to the lookup table. Id is a hash of param values 
        created by _make_hash. type is the type of data asset being created.
        params is a dict of parameter values
        '''    
        lookup = self._load_lookup()

        row = [id, type, params['version'], params['p'], params['q'], 
                params['walk_length'], params['num_walks'],
                params['window_size'], params['negative_samples']]
        
        lookup.loc[ len(lookup.index) ] = row       # Add row to bottom of table

        self._save_lookup(lookup)



    def _get_from_lookup(self, id):
        '''
        Retrieve an item by id from lookup
        '''
        lookup = self._load_lookup()

        row = lookup.loc[ lookup['id'] == id ]

        return row


    def _remove_from_lookup(self, id):
        '''
        Delete a given item from the lookup table
        '''
        lookup = self._load_lookup()

        filtered = lookup['id'] != id

        lookup = lookup[filtered]

        self._save_lookup(lookup)


    def _prune_lookup(self, prune_file=True):
        '''
        Get all identifiers of processed datasets. Remove any which are not 
        present in the lookup table. This means they have been deleted 
        by the user.
        '''

        # Select predicate
        if prune_file:
            predicate = lambda x: x.is_file()
        else:
            predicate = lambda x: x.is_dir()

        lookup = self._load_lookup()
        
        # List all subdirectories in the data_dir
        dirs = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        
        # Get all data assets by id 
        ids = []
        for d in dirs:
            with os.scandir(d) as directory:
                for item in directory:
                    if predicate(item):
                        id = item.name.split('.')[0]
                        ids.append(id)
        
        # Filter lookup table by ids present in file system
        lookup = lookup[ lookup['id'].isin(ids) ]

        self._save_lookup(lookup)

    
    def _make_hash(self, *params):
        '''
        This takes the names of the parameters that are used to 
        define the various datasets as arguments. It then uses
        the names as keys to retrieve the values from the config 
        file. The values are then used to make the hash

        *params         These are the names of parameters
                        as strings. Example: 'version',
                        not 4.4.207, 'p', not 0.5
        
        DEPRECIATED

        '''

        config = self.config.show()
        hash_master = hashlib.shake_128()
        hash_length = self.hash_length

        current = config['data']['current']
        
        # Hash of version - Basic PPI
        if 'version' in params:
            version = bytes(str(config['data']['version']), 'utf-8')
            hash_master.update(version)
            id = hash_master.hexdigest(hash_length)
        
        # Hash of p, q - Preprocessed graph weights
        if ('p' in params) or ('q' in params):
            p = bytes(str(config['data']['experiments'][current]['p']), 'utf-8')
            q = bytes(str(config['data']['experiments'][current]['q']), 'utf-8')
            hash_master.update(p)
            hash_master.update(q)
            id = hash_master.hexdigest(hash_length)

        # Hash of walk_length and num_walks - Walks
        if ('walk_length' in params) or ('num_walks' in params):
            walk_length = bytes(str(config['data']['experiments'][current]['walk_length']), 'utf-8')
            num_walks = bytes(str(config['data']['experiments'][current]['num_walks']), 'utf-8')
            hash_master.update(walk_length)
            hash_master.update(num_walks)
            id = hash_master.hexdigest(hash_length)      

        # Hash of window_size and negative_samples - Node Embeddings
        if ('window_size' in params) or ('negative_samples' in params):
            window_size = bytes(str(config['data']['experiments'][current]['window_size']), 'utf-8')
            negative_samples = bytes(str(config['data']['experiments'][current]['negative_samples']), 'utf-8')
            hash_master.update(window_size)
            hash_master.update(negative_samples)
            id = hash_master.hexdigest(hash_length)        

        return id              

    def _make_hash_from_dict(self, asset, values):
        '''
        New and imporved version of make_hash method. Now, rather than
        having to load the config file and extract the values, they are
        simpley provided directly by the caller. 

        values:             dict of parameter values

        '''
        
        # Initialise the hash
        hash_master = hashlib.shake_128()
        hash_length = self.hash_length   
        b = lambda x: bytes(str(x), 'utf-8')

        # Make byte values
        version = b(values.get('version'))
        p = b(values.get('p'))
        q = b(values.get('q'))
        walk_length = b(values.get('walk_length'))
        num_walks = b(values.get('num_walks'))
        window_size = b(values.get('window_size'))
        negative_samples = b(values.get('negative_samples'))

        # Make hash and return id
        hash_master.update(version) # biogrid dataset
        if asset == 'biogrid':
            return hash_master.hexdigest(hash_length)
        
        hash_master.update(p) # transition probs dataset
        hash_master.update(q)
        if asset == 'transition':
            return hash_master.hexdigest(hash_length)

        hash_master.update(walk_length) # walks dataset
        hash_master.update(num_walks) 
        if asset == 'walk':
            return hash_master.hexdigest(hash_length)        

        hash_master.update(window_size) # skipgram dataset
        hash_master.update(negative_samples) 
        if asset == 'skipgram':
            return hash_master.hexdigest(hash_length)     

        raise ValueError(f'Dataset not recognised: {asset}. Please choose from biogrid | transition | walk | skipgram')         

    def _get_lookup_filepath(self):
        '''
        Returns the filepath to the lookup table
        '''
        if self.lookup_fp is not None:
            return os.path.join(self.data_dir, self.lookup_fp)
        else:
            return os.path.join(self.data_dir, self.filename)














class ModelArchive(Archive):

    '''
    The main difference between this class and the base Archive class is that 
    the base archive class provides the option to generate ids at various hierarchies
    such as [verion], [version, p, q], [version, p, q, ...] where each level of 
    the hierarchy is a superset of the previous. This allows you to create hierarchies
    of dependencies amoung data assets. 
        The ModelArchive class, however, has a flat information structure. There are a 
    given number of fields [model, dataset, encoder, ...] which may or may not be 
    present, but in every case the final hash is a result of all those values. 
        One important detail of the ModelArchive class is that any model which uses trained
    weights from a node2vec model, is indirectly dependent on the following parameters:

    version, p, q, num_walks, walk_length, negative_samples, window_size, node2vec:encoder,
    node2vec:embedding_dim, node2vec:epochs 
    
    - which is a lot. Therefore, to simplify this dependency, the argument tranfer_id 
    is provided which is the uuid of a given node2vec trained model instance. When training
    a node2vec instance, transfer_id = None
    '''

    # Class attrs
    location = 'cache/tensorflow'
    columns = ['id', 'model', 'dataset', 'transfer_id', 'encoder', 'architecture', 
                'embedding_dim', 'split', 'epochs']
    column_offset = 4           # id + number of user provided values ^ user provides: model, dataset, transfer_id 
    missing_value = None

    def __init__(self, dir=None):

        if dir is None:
            dir = self.location
        
        self.config = Config()

        super().__init__(self.config, alt_dir=dir, prune_file=False)

    def make_id(self, model, experiment, dataset, transfer_id, add_to_lookup=False):
        '''
        Creates a hash ID based on parameter values. 
        Optionally adds it to the lookup table.
        '''

        accepted_args = ['node2vec', 'binary_classifier', 'multi_label']

        assert model in accepted_args, f'ERROR: argument {model} not recognised, select from {accepted_args}'
        
        id = self._make_hash(model, experiment, dataset, transfer_id)

        if add_to_lookup:

            params = self.config.get_experiment(model)

            self._add_to_lookup(id, model, dataset, transfer_id, params)
        
        return id
    
    def _init_lookup(self, columns=None):
        '''
        If the lookup table hasnt been created yet, make it.
        - columns:      List of column names that should be used to create 
                        the table
        '''

        if columns is None:
            columns = self.columns

        filepath = self._get_lookup_filepath()

        try:

            if not os.path.exists(filepath):
                df = pd.DataFrame(columns=columns)
                self._save_lookup(df)
        
        except Exception as e:

            print(f"ERROR: {e}")
    
    def _add_to_lookup(self, id, model, dataset, transfer_id, params):
        '''
        Add an item to the lookup table. Id is a hash of param values 
        created by _make_hash. model is the type of asset being created.
        params is a dict of parameter values
        '''    
        lookup = self._load_lookup()

        row = self._make_row(id, model, dataset, transfer_id, params)
        
        lookup.loc[ len(lookup.index) ] = row       # Add row to bottom of table

        self._save_lookup(lookup)
    
    def _make_row(self, id, model, dataset, transfer_id, params):
        '''
        This parses a params dictionary in a specified order providing 
        default values in the case where no value is provided and returns
        a row as a list which can be written to file.
        '''

        row = [id, model, dataset, transfer_id] # These are values which are not obtained from config 

        offset = self.column_offset
        for column in self.columns[offset:]:
            value = params.get(column, self.missing_value)
            row.append(value)
        
        return row

    def _make_hash(self, model, experiment, dataset, transfer_id):
        '''
        This takes the names of the parameters that are used to 
        define the various datasets as arguments. It then uses
        the names as keys to retrieve the values from the config 
        file. The values are then used to make the hash

        - model:        This is the type of model that is being 
                        archived, 'node2vec, 'binary_classifier'
                        etc...
        - dataset:      This is the id of the data asset used 

        It uses the current values specified in the config file to make
        the hash
        '''

        # Initialise the hash
        hash_master = hashlib.shake_128()
        hash_length = self.hash_length
        b = lambda x: bytes(str(x), 'utf-8')

        # Get the relevant config details and target params
        config = self.config.get_experiment(model, experiment)
        offset = self.column_offset
        params = self.columns[offset:]

        # Hash model type and dataset ID
        hash_master.update(b(model))
        hash_master.update(b(dataset))
        hash_master.update(b(transfer_id))

        # Hash values from config
        for param in params:
            value = config.get(param, self.missing_value)
            hash_master.update(b(value))
        
        return hash_master.hexdigest(hash_length)
        

    
if __name__ == "__main__":
    archive = Archive('c')
    # archive._prune_lookup()
    config = Config()

    model_archive = ModelArchive()

    # transfer_id  = model_archive.make_id('node2vec', '87c87ebk7', None, add_to_lookup=True)
    
    # model_archive.make_id('binary_classifier', '87c87ebk7', transfer_id, add_to_lookup=True)
    # model_archive.make_id('multi_label', '87c87ebk7', transfer_id, add_to_lookup=True)

    # print(model_archive._load_lookup())
    pass

     
        