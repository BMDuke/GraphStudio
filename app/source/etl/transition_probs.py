import os
import random
import time

import psutil
import gc

import networkx as nx
from prettytable import PrettyTable

from source.graphs.n2v import Node2Vec
from source.etl.biogrid import BioGrid
from source.etl.etl import ETL

class TransitionProb(ETL):
    
    '''
    This is a class that orchestrates calculating transition probabilities
    for PPI graphs. It uses the Node2Vec class to handle the weight processing,
    using parameter values from the config manager. 

    About:
    The input to this tool is the edge list calculated from a given version 
    of the BioGrid data. This initially assumes that it is a directed 
    graph and is unweighted. The transition probabilities are caluculated in 
    two steps:
     - edge normalisation
     - search term encoding
    Both of these are handled by the Node2Vec class. The adidtional functionality 
    provided by this class is the arrangement of versioning, configuration 
    management and archive management. 

    Public API:
     - process()       Reads in the raw biogrid data, filters it for 
                        human proteins and then selects the enrtez columns     
     - describe()      Summarizes the processed data
     - validate()      Validate the processed data
     - head()          Print the top n rows of the processed data    

    '''

    # Class attrs
    source_dir = 'data/processed/biogrid'
    destination_dir = 'data/processed/transition_probs'

    graph_format = 'pickle'


    def __init__(self, debug=False, verbose=True):
        '''
        Config file is required to create an ID for the transition
        probabilities that are calculated. This depends on 
        the biogrid version, p and q. 
        '''
        
        super().__init__()

        self.debug = debug
        self.verbose = verbose
    
    def process(self, experiment='current'):
        '''
        This function handles retrieving configuration values from 
        the config tool and checkng whether the required and target 
        resources exist. If the required resources exist and the 
        target resource does not exist, the target resource is 
        created, saved and returned. This can be a long computation. 

        Things to do: 
        > This could do with a refactor to make more testable
        '''

        # Getting config values 
        biogrid_version = self._get_biogrid_version()
        p, q = self._get_p_q_values(experiment=experiment)

        # Load the config and get the current biogrid version
        biogrid_id = self._make_uuid('biogrid', experiment=experiment)
        transition_id = self._make_uuid('transition', experiment=experiment)

        # Check biogrid data exists
        biogrid_fp = os.path.join(self.source_dir, f"{biogrid_id}.csv")

        assert os.path.exists(biogrid_fp), f"ERROR: biogrid dataset version {biogrid_version} not found. Have you downloaded the data?"

        # Check if TP's have already been calculated for this biogrid 
        # version and these p-q values
        filename = f'{transition_id}.{self.graph_format}'
        filepath = os.path.join(self.destination_dir, filename) # File extension will cause error in n2v
        tps_exist = os.path.exists(filepath)

        if tps_exist:
            print(f'CONFLICT: Graph of biogrid version {biogrid_version} already exists for p: {p} q: {q}\n> {filename}')
            return self._load_n2v_graph(experiment)

        # Get the edge list
        edge_list = self._load_biogrid()

        # Create a new node2vec instance from the edge list
        n2v = Node2Vec(edge_list=edge_list, is_directed=False)
        n2v.set(p=p, q=q)
        if self.verbose:
            print("Number of nodes: ", nx.number_of_nodes(n2v.graph))
            print("Number of edges: ", nx.number_of_edges(n2v.graph))

        # Process the weights
        n2v.process_weights()
        
        # Save the results
        self._save_n2v_graph(n2v, experiment)

        return n2v
    
    def describe(self, n2v=None, experiment='current'):
        '''
        This calculates various metrics to decribe a processed node2vec
        instance that has been saved. It describes the config parameters
        that were used to create it, a selection of graph metrics and 
        its space taken on the OS. 
        '''
        
        # Config - Get details of the current config 
        biogid_version = self._get_biogrid_version()
        p, q = self._get_p_q_values(experiment=experiment)

        config_details = PrettyTable()
        conf_fieldnames = ['Biogrid version', 'p', 'q'] # Set title
        config_details.field_names = conf_fieldnames
        config_details.add_row([biogid_version, p, q]) # Add row

        # Graph - Load nx graph and get details    
        if not n2v:
            if self.verbose:
                print('Loading node2vec graph...')
            n2v = self._load_n2v_graph(experiment)
        num_nodes = nx.number_of_nodes(n2v.graph)
        num_edges = nx.number_of_edges(n2v.graph)
        num_attr = nx.number_attracting_components(n2v.graph)
        num_isol = nx.number_of_isolates(n2v.graph)
        density = nx.density(n2v.graph)

        graph_details = PrettyTable()
        graph_fieldnames = ['# nodes', '# edges', 'Degree'] # Set title
        graph_details.field_names = graph_fieldnames
        graph_details.add_row([num_nodes, num_edges, density]) # Add row

        graph_details_2 = PrettyTable()
        graph_fieldnames_2 = ['# attracting components', '# isolates'] # Set title
        graph_details_2.field_names = graph_fieldnames_2
        graph_details_2.add_row([num_attr, num_isol])  # Add row

        # OS - Get the filesize of the saved graph 
        uuid = '.'.join([self._make_uuid('transition', experiment=experiment), self.graph_format])
        url = os.path.join(self.destination_dir, uuid)
        size = os.stat(url).st_size / 1024**3

        os_details = PrettyTable()
        os_fieldnames = ['Filesize GB'] # Set title
        os_details.field_names = os_fieldnames
        os_details.add_row([size]) # Add row

       
        # Display
        if self.verbose:
            print("\nConfiguration Details:")
            print(config_details, '\n')
            print('Graph Details:')
            print(graph_details, '\n')
            print(graph_details_2, '\n')
            print('File System Details')
            print(os_details, '\n')


    def validate(self, n2v=None, experiment='current'):
        '''
        This function runs some tests designed to validate the graph.
        It tests whether all first and second degree edges have been 
        normalise, that is their sum is 1. 
        '''
        
        if not n2v:
            if self.verbose:
                print('Loading node2vec graph...')            
            n2v = self._load_n2v_graph(experiment)

        # Take a random sample from the nodes in the graph
        nodes = random.sample(n2v.graph.nodes, 100)

        # Check all outbound edges sum to 1 for all nodes
        # Check transition probs sum to 1
        outboud_sum_to_1 = True
        tp_sum_to_1 = True

        tollerance = 0.0001
        key = n2v.second_order_key

        for node in nodes:
            total = 0
            for neighbour in n2v.graph.neighbors(node):
                total += n2v.graph[node][neighbour]['weight']
                tp_total = sum([v for k, v in n2v.graph[node][neighbour][key].items()])
                if (abs(1 - tp_total) > tollerance):
                    if tp_total == 0: # Neighbour has no neighbours
                        continue         
                    tp_sum_to_1 = False
            if (abs(1 - total) > tollerance):
                if len(n2v.graph[node]) == 0: # Node has no neighbours
                    continue
                outboud_sum_to_1 = False
        
        validation = PrettyTable()
        validation.field_names = ['Criteria', 'Result']
        validation.add_row( ['Edges are normalised', outboud_sum_to_1 ] )
        validation.add_row( ['Transition probabilities are normalised', tp_sum_to_1 ] )

        if self.verbose:
            print('Graph validation - n=100')
            print(validation)

        return not any([outboud_sum_to_1, tp_sum_to_1 ])
            
            

    def head(self, nrows=5, n2v=None, experiment='current'):
        '''
        This function attempts to provide standard 'head' functionality as is 
        comonly used with tabular data to the n2v graph. This isnt a perfect match
        although it adequately describes the significant information for the project. 
        '''

        if not n2v:
            if self.verbose:
                print('Loading node2vec graph...')            
            n2v = self._load_n2v_graph(experiment)

        # Get first n edges
        edges = []

        for i in range(nrows): # Sample the edges
            edge = self._sample_n2v_edge(n2v)
            edges.append(edge)

        # Make table
        head = PrettyTable()
        fieldnames = ['Source', '1st Degree', '1st Degree TP', '2nd Degree TPs'] # Set title
        head.field_names = fieldnames
        for edge in edges:          # Write rows to the table
            from_node = edge[0]
            to_node = edge[1]
            weight = n2v.graph[from_node][to_node]['weight']
            second_order_sample = self._get_tp_sample(from_node, to_node, n2v, limit=5)
            head.add_row([from_node, to_node, weight, second_order_sample[0]]) # Add row     
            for sample in second_order_sample[1:]:
                head.add_row(['', '', '', sample]) # Add row     

        if self.verbose:
            print(head)   



    def _load_biogrid(self):
        '''
        This loads the biogrid data using the BioGrid ETL utility.
        It then transforms it from a pandas dataframe to a list of 
        lists and returns it to the caller. 
        '''

        biogrid = BioGrid(verbose=False)

        data = biogrid._load_ppi().applymap(int) # _load_ppi returns pandas DF

        return data.values.tolist()

    def _save_n2v_graph(self, n2v, experiment):
        '''
        Just a light wrapper to add some interactivity when saving
        the n2v graph. Writing to JSON can take a while
        '''
        if self.verbose:
            print('Saving graph... ')

        transition_id = self._make_uuid('transition', experiment=experiment, add_to_lookup=True)
        destination = os.path.join(self.destination_dir, transition_id)

        n2v.save(destination, format=self.graph_format)

    def _load_n2v_graph(self, experiment):
        '''
        Just a light wrapper to add some interactivity when saving
        the n2v graph. Writing to JSON can take a while
        '''
        if self.verbose:
            print('Loading graph... ')        

        transition_id = self._make_uuid('transition', experiment=experiment)
        destination = os.path.join(self.destination_dir, transition_id)            

        n2v = Node2Vec()
        n2v.load(destination, format=self.graph_format)     

        return n2v    

    def _sample_n2v_edge(self, n2v):
        '''
        Selects a random edge from the graph
        '''

        edge = None
        attempts = 10

        while (not edge) and (attempts > 0):
            sample = random.sample(n2v.graph.nodes(), 1)[0]
            neighbours = list(n2v.graph.neighbors(sample))
            if neighbours:
                edge =  [sample, neighbours[0]]    
            attempts -= 1    
        
        return edge

    def _get_tp_sample(self, from_node, to_node, n2v, limit=10):
        '''
        For a given 1st degree edge in the graph, draw random 
        samples from the 2nd degree transition probabilities. 
        '''

        key = n2v.second_order_key

        tps = n2v.graph[from_node][to_node][key]

        output = []

        for k, v in tps.items():
            if limit > 0:
                line = f'{k:<11} {round(v, 6)}'
                output.append(line)
                limit -= 1
            else:
                line = f'Plus {len(tps.items()) - limit} more records...'
                output.append(line)
                break
        
        output.append('') # apply some spacing

        return output

if __name__ == "__main__":
    tp = TransitionProb(debug=False)
    # n2v = tp.process()
    # tp.head(n2v)
    # tp.describe(n2v)
    # tp.validate(n2v)

