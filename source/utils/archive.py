import os
import hashlib

import pandas as pd

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

    def __init__(self, config, lookup_fp=None, alt_dir=None):
        self.config = config
        self.lookup_fp = lookup_fp # Used to create seperate lookup for testing

        if alt_dir:
            self.data_dir = alt_dir # Use alternative data directory if specified. 

        self._init_lookup()
        self._prune_lookup()

    def make_id(self, type, *params, add_to_lookup=False):
        '''
        Creates a hash ID based on parameter values. 
        Optionally adds it to the lookup table.
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


    def _prune_lookup(self):
        '''
        Get all identifiers of processed datasets. Remove any which are not 
        present in the lookup table. This means they have been deleted 
        by the user.
        '''

        lookup = self._load_lookup()
        
        # List all subdirectories in the data_dir
        dirs = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        
        # Get all data assets by id 
        ids = []
        for d in dirs:
            with os.scandir(d) as directory:
                for item in directory:
                    if item.is_file():
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

        

    def _get_lookup_filepath(self):
        '''
        Returns the filepath to the lookup table
        '''
        if self.lookup_fp is not None:
            return os.path.join(self.data_dir, self.lookup_fp)
        else:
            return os.path.join(self.data_dir, self.filename)



    
if __name__ == "__main__":
    archive = Archive('c')
    # archive._prune_lookup()
    pass

     
        