from genericpath import isfile
import os
import pathlib
import shutil
import itertools
import multiprocessing
import concurrent.futures
import random
import csv
import ujson as json
from math import ceil

import networkx as nx
import pandas as pd
import tqdm

class Node2Vec:

    '''
    This class is used to model the interactions in the PPI
    dataset. It performs the edge preprocessing and random 
    walk generation required to create node embeddings, as 
    per: 
    - https://arxiv.org/pdf/1607.00653.pdf

    About: 
    A Node2Vec instance is a facade containing a networkx grpah 
    instance. It is parameterised by 4 parameters
     - p: Return paramater: used in generating transition probs. Controls
        the liklihood of immediately revisiting a node in the walk.
     - q: In-out parameter: used in generating transition probs. Controls 
        the behaviour of the random walk to be more like BFS (q > 1) or 
        more like DFS (q < 1). 
     - num_walks: used in generating the random walks, controls how many
        times a node is sampled
     - walk_length: used in generating the random walks, controls how
        many steps are taken when sampling
    These are used to perform the two main features of Node2Vec; 
    preprocessing transition probablities and generating random walks.
    This class also handles all the IO operations for graph objects. However, 
    it does not interface with the config or archive tools, so it is fully 
    controlled by the caller.  

    Public API:
     - load()                   Loads a saved n2v model. This can be in either
                                GML, GraphML or pickle format
     - save()                   This saves a n2v model. This can be either in 
                                GML, GraphML or pickle format
     - process_weights()        This procomputes the transition probabilities 
                                that are used to generate the random walks
     - generate_walks()         This takes a random walk through the graph where
                                the next step is biased by the transition probabilities.
                                This allows you to explore the topology of the graph. 
                                The walks are used in the same way as sentences are used
                                in a word2vec model. Skipgrams are generated from each
                                walk and these are used to train an embedding model.
     - set()                    Setter function which can be used to update p, q, num_walk
                                walk_length parameters
     - get()                    Getter for p, q, num_walk, walk_length values

    '''

    def __init__(self, edge_list=None, p=0.5, q=0.7, 
                    num_walks=10, walk_length=80, is_directed=False, verbose=True):
        '''
        Only undirected and directed graphs supported currently
        '''

        if is_directed:
            nx_generator = nx.DiGraph
        else:
            nx_generator = nx.Graph
        
        self.graph = nx_generator()

        if edge_list:
            if len(edge_list[0]) == 3: # edge list is shape(3 x n) => weighted
                self.graph.add_weighted_edges_from(edge_list)
            else:
                self.graph.add_edges_from(edge_list)

        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.is_directed = is_directed
        self.verbose = verbose

        self.second_order_key = 'secondOrderProbs'
        self.temp_dir = 'data/processed/walks/tempdir'

    def save(self, filepath, format='json'):
        '''
        Save self.graph in the desired format. Supported formats:
         - gml
         - graphml
         - pickle

        Filepath must not have a suffix eg. not data/graph/d56fu8.csv
        '''
        mapping = {
            'gml': nx.write_gml,
            'graphml': nx.write_graphml_lxml,
            'pickle': nx.write_gpickle,
            'json': self._write_json
        }

        assert format in mapping, f'Method unrecognised: {format} - save files with "gml", "graphml" or "pickle"'

        writer = mapping[format]

        filepath_with_suffix = '.'.join([filepath, format])

        try: 

            writer(self.graph, filepath_with_suffix)

        except Exception as e:

            print(f"ERROR: {e} - node2vec instance .save()")
            raise

    def load(self, filepath, format='json'):
        '''
        Loads a nx graph instance into self.graph. Supported formats:
         - gml
         - graphml
         - pickle

        Filepath must not have a suffix eg. not data/graph/d56fu8.csv
        '''
        mapping = {
            'gml': nx.read_gml,
            'graphml': nx.read_graphml,
            'pickle': nx.read_gpickle,
            'json': self._read_json
        }

        assert format in mapping, f'Method unrecognised: {format} - load files with "gml", "graphml" or "pickle"'
        assert filepath is not None, "Filepath not provided"      

        reader = mapping[format]

        filepath_with_suffix = '.'.join([filepath, format])

        try: 

            self.graph = reader(filepath_with_suffix)
            self.is_directed = self.graph.is_directed()

        except Exception as e:

            print(f"ERROR: {e} - node2vec instance .save()")
            raise

    def get(self, *args):
        '''
        Return the values of the graphs parameters as a dict. 
        It accept a variable number of arguments which are the names
        of parameter values. Example 'p', 'q'... which are used to 
        filter the results.
        Input: str
        '''

        parameter_values = {}

        if not args:

            args = ['p', 'q', 'num_walks', 'walk_length', 'is_directed'] # If no parameters give, return all except graph
        
        for arg in args:

            assert arg in self.__dict__, f'ERROR: {arg} not found in Node2Vec'

            parameter_values[arg] = self.__dict__[arg]
        
        return parameter_values

    def set(self, **kwargs):
        '''
        Set the graphs parameters to the values given or raise an
        error if the key is unknown
        '''

        for attribute, value in kwargs.items():

            assert attribute in self.__dict__, f'ERROR: attribute "{attribute}" not found in Node2Vec'

            setattr(self, attribute, value)

    def process_weights(self):
        '''
        > _normalise_edge_weights: Pass over all nodes and normalise them 
            - must handle directed and undirected graphs
        > _calculate_transition_probabilities: calculate transition probabilities 
            from the p, q values
        '''

        graph = self.graph

        # Normalise any weighted edges
        graph = self._normalise_edges(graph)

        # Calculate transition probabilities
        graph = self._calculate_transition_probabilities(graph)

        # Update self
        self.graph = graph                           

    def _normalise_edges(self, graph):
        '''
        This scales any values for weighted edges leaving from a given node to
        the range 0:1. 
        For this to work, the graph must be a directed graph, so
        if the graph is not already a directed graph, it will be converted to one.
        If the graph is unweighted, edges are taken to be 1. 
        '''

        graph = graph.to_directed() # Convert to directed graph

        if self.verbose:
            nodes = tqdm.tqdm(graph.nodes, desc="Normalising edge weights")
        else: 
            nodes = graph.nodes

        for node in nodes:

            first_degree_nighbours = [n for n in graph.neighbors(node)]

            # Calculate normalising constant for nodes outgoing edges
            normalising_constant = sum([graph[node][neighbour].get('weight', 1.) 
                                            for neighbour in first_degree_nighbours])

            # Apply normalising constant to edges
            for neighbour in first_degree_nighbours:
                normalised_prob = graph[node][neighbour].get('weight', 1.) / normalising_constant
                graph[node][neighbour]['weight'] = normalised_prob
        
        return graph

    def _calculate_transition_probabilities(self, graph):
        '''
        This calculates the normalised transition probabilities for a 
        directed graph and applies the values to the graph in place.
        Note: This cannot be parralellised because multiple processes each
            get a copy of an object so you would end up with modifications
            made to multiple copies which would then need reconciling
        '''

        if self.verbose:
            nodes = tqdm.tqdm(graph.nodes, desc="Calculating transition probabilities")
        else: 
            nodes = graph.nodes

        # Calculate transition probabilities
        for source in nodes:
            
            first_degree_nighbours = [n for n in graph.neighbors(source)]

            for neighbour in first_degree_nighbours:

                second_degree_neighbours = [n for n in graph.neighbors(neighbour)]

                unnormalised_transition_probabilities = {}

                for destination in second_degree_neighbours:

                    weight = graph[neighbour][destination].get('weight')
                    weight = float(weight)

                    # Calculate the modified weights
                    if destination == source:
                        transition_prob = weight * (1. / self.p) # Backwards probability

                    elif destination in first_degree_nighbours:
                        transition_prob = weight * 1. # Node is neighbour of source

                    else:
                        transition_prob = weight * (1. / self.q) # BFS/DFS toggle

                    unnormalised_transition_probabilities[destination] = transition_prob         

                # Normalise second degree transition probabilities
                normalised_transition_probabilities = {}
                normalising_constant = sum(unnormalised_transition_probabilities.values())

                for key, value in unnormalised_transition_probabilities.items():
                    normalised_transition_probabilities[key] = value / normalising_constant

                key = self.second_order_key
                graph[source][neighbour][key] = normalised_transition_probabilities  

        return graph       

    def generate_walks(self, filepath=None):
        '''
        This function generates the random walks that are used to create the skipgrams.
        This essentially samples the nodes in the graph based on the topology and 
        the transition probabilities.

        If no filepath is specified, the results are stored in memory and returned to 
        the caller. 

        If a filepath is specified, the results are written to disk and the results are
        returned to the caller if they are small enough
        '''

        graph = self.graph

        # Create directories to persist walks if filepath provided
        if filepath:
            temp_dir = self.temp_dir
            result_fp = '.'.join([filepath, 'csv'])
            os.mkdir(temp_dir) # Create a temporary directory 
            pathlib.Path.touch(pathlib.Path(result_fp), exist_ok=False) # Create result file 

        # Split the nodes into chunks
        num_cpus = multiprocessing.cpu_count()
        nodes = graph.nodes
        chunksize = ceil(len(nodes) / num_cpus )
        chunks = self._make_chunks(chunksize, nodes)

        # Process the chunks
        func = self._sample # Function to be applied to the graph (makes walks)
        walks = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            
            futures = []

            for i, chunk in enumerate(chunks):
                fp = None
                if filepath:
                    fp = os.path.join(temp_dir, str(i))
                future = executor.submit(func, chunk, graph, fp) # If filepath is provided, walks are saved to that location
                futures.append(future)

            for future in futures:
                walks.extend(future.result())    

        # Combine results if filepath provided
        if filepath:
            self._combine_results(temp_dir, result_fp)
            self._cleanup(temp_dir)

        # Return walks or sample of walks
        if filepath: # filepath was given therefore walks were written to disk and nothing returned
            return pd.read_csv(result_fp, nrows=1000, header=None)
        else:
            return walks
    
    def _make_chunks(self, n, iterable):
        '''
        Returns an iterable of chunks of size n of the original iterable
        '''
        it = iter(iterable)

        while True:
            chunk = list(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    def _sample(self, nodes, graph, filepath=None):
        '''
        Draw samples from the graph. This generates the random walks. 
        If a directory is specified then the results are saved to that 
        location, else they are returned to the caller.
        '''

        num_walks = self.num_walks
        walk_length = self.walk_length

        walks = []

        if self.verbose:
            num_walks = tqdm.tqdm(range(num_walks), desc=f'Generating random walks - chunksize: {len(nodes)} walks: {num_walks} length: {walk_length}')
        else:
            num_walks = range(num_walks)

        for _ in num_walks:

            random.shuffle(nodes)

            for source in nodes:

                sample = [source]

                # Select a node from the neighborhood of source
                neighbors = [n for n in graph.neighbors(source)]
                probabilities = [graph[source][n]['weight'] for n in neighbors]
                neighbor = random.choices(neighbors, weights=probabilities, k=1)[0]
                sample.append(neighbor)

                # Generate samples
                start_node = source             # Node 0 in 2nd order markovian
                current_node = neighbor         # Node 1 in 2nd order markovian
                next_node = None                # Node 2 in 2nd order markovian. We are sampling for this one.

                while len(sample) < walk_length:
                    
                    key = self.second_order_key
                    second_order_probs = graph[start_node][current_node][key]
                    neighbors, probabilities = [], []

                    for key, value in second_order_probs.items():
                        neighbors.append(key)
                        probabilities.append(value)

                    if len(neighbors) == 0:
                        continue

                    next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
                    sample.append(next_node)

                    start_node = current_node
                    current_node = next_node
                    next_node = None
                
                walks.append(sample)

        if filepath:
            self._to_csv(walks, filepath)
            return []
        
        return walks

    def _to_csv(self, data, filepath):
        '''
        Writes an dataset out to a specific filepath as a csv
        '''

        try:

            f_out = f'{filepath}.csv'
            dataframe = pd.DataFrame(data)
            dataframe.to_csv(f_out, index=False, header=False)
        
        except Exception as e: 

            print(f"ERROR: {e}") 
            raise

    def _combine_results(self, temp_dir, result_filepath):
        '''
        This function lists all the files in a directory and merges
        them into one file specified by the result_filepath
        '''

        files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]

        try: 
            
            for file in files:
                filepath = os.path.join(temp_dir, file)
                with open(filepath, 'r') as f_in, open(result_filepath, 'a') as f_out:
                    results = csv.writer(f_out)
                    walks = csv.reader(f_in)
                    for walk in walks:
                        results.writerow(walk)
        
        except Exception as e:

            print(f"ERROR: {e}") 
            raise

    def _cleanup(self, temp_dir):
        '''
        This deletes a directory and all its contents 
        '''
        try:
            
            shutil.rmtree(temp_dir)
        
        except Exception as e:

            print(f"ERROR: {e}") 
            raise

    def _write_json(self, graph, filepath):
        '''
        This is a function to handle converting an nx graph to json
        format and then writing it out to a file. 
        '''

        json_data = nx.readwrite.json_graph.node_link_data(graph)

        try: 

            with open(filepath, 'w') as json_out:
                json.dump(json_data, json_out, indent=4)

        except Exception as e:

            print(f"ERROR: {e}") 
            raise

    def _read_json(self, filepath):
        '''
        This is a function to handle loading a json file and 
        converting it into a nx graph. 
        '''

        # json_data = nx.readwrite.json_graph.node_link_data(graph)

        try: 

            with open(filepath, 'r') as json_in:
                json_data = json.load(json_in)
            graph = nx.readwrite.json_graph.node_link_graph(json_data)
            return graph

        except Exception as e:

            print(f"ERROR: {e}") 
            raise
        

if __name__ == "__main__":
    n2v = Node2Vec()
    # print('\n', n2v.get())    
    # n2v.set(p=100, q='testing_q_value')
    # print('\n', n2v.get('p', 'q'))    
    # n2v.set(p=100, q='testing_q_value', d=0.1)


