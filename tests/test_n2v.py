import os
import shutil
import unittest
import math
from pathlib import Path
from random import choice
from string import ascii_uppercase

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pandas as pd
import numpy as np

from source.graphs.n2v import Node2Vec
from tests.dummy_data.scripts.graph_dd import test_graphs

# Configure unittest
unittest.TestLoader.sortTestMethodsUsing = None

class TestNode2Vec(unittest.TestCase):

    '''
    Tests
    - __init__
        - ✓ Correctly initialises graph or digraph
    - .save()
        - ✓ errors correctly handled
        - ✓ suffix correctly added to filepath
        - ✓ graph correctly saved to location
    - .load()
        - ✓ errors correctly handled
        - ✓ suffix correctly added to filepath
        - ✓ graph correctly loaded into instance
    - .get()
        - ✓ All params returned when no argument are passed
        - ✓ specific params returned when arguments are passes
        - ✓ error raised when specified param doesnt exist
    - set()
        - ✓ Attribute values are correctly updated
        - ✓ errors are handled if attribute is not found
    - _normalise_edges()
        - ✓ undirected graphs are converted to directed graphs
        - ✓ edge normalisation is correct for directed graphs
        - ✓ edge normalisation is correct for undirected graphs
    - _calculate_transition_probabilities()
        - ✓ Calulation is correct
        - ✓ Transition probability sum is equal to one
    - _make_chunks()
        - ✓ Iterable is correctly divided 
    - _sample()
        - ✓ Output shape is correct
        - ✓ A single node with an edge to itself creates a sample of only itself
        - ✓ Node frequencies reflect transition probabilities
        - ✓ There are no null values in the sample
    - to_csv()
        - ✓ File written to location
    - combine_results()
        - ✓ Copying is accurate
    - _cleanup()
        - ✓ desired directory is removed with files
        - ✓ desired directory is removed without files
    - generate_Walks()
        - Concurrent execution is successful
        - Walks are returned when no filepath is provided
        - walks are persisted when a filepath is provided 
        - pd df is retruned when a filepath is provided
            
        '''

    @classmethod
    def setUpClass(cls) -> None:    

        # Make a directory as an IO test space
        cls.temp_dir = 'tests/dummy_data/graphs'
        os.mkdir(cls.temp_dir)
        
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        
        # Remove IO test space directory
        shutil.rmtree(cls.temp_dir)

        return super().tearDownClass()

    def setUp(self) -> None:

        return super().setUp()



    def test_1__init__(self):
        '''
        Testing for:
        > Accurate initialisation of graph or digraph
        '''

        # Get test graphs
        undirected = test_graphs['star_undirected_weighted']()
        directed = test_graphs['star_directed_weighted']()

        # Initialise directed or undirected graphs
        undirected_as_undirected = Node2Vec(edge_list=undirected['edge_list'], is_directed=False)
        undirected_as_directed = Node2Vec(edge_list=undirected['edge_list'], is_directed=True)
        directed_as_undirected = Node2Vec(edge_list=directed['edge_list'], is_directed=False)
        directed_as_directed = Node2Vec(edge_list=directed['edge_list'], is_directed=True)        

        # ++++ Testing ++++
        self.assertFalse(undirected_as_undirected.graph.is_directed())
        self.assertTrue(undirected_as_directed.graph.is_directed())
        self.assertFalse(directed_as_undirected.graph.is_directed()) # This passes meaning that directed -> undirected transformation is lossy
        self.assertTrue(directed_as_directed.graph.is_directed())

    def test_2_save(self):
        '''
        Testing for:
        > error handling
        > siffix addition
        > correctness
        '''

        # Get test graphs
        graph = test_graphs['star_undirected_unweighted']()

        # Initialise directed or undirected graphs
        n2v = Node2Vec(edge_list=graph['edge_list'], is_directed=False)      

        # Save the graph in all three supported formats 
        test_name = 'test_saving'
        filepath = os.path.join(self.temp_dir, test_name)
        n2v.save(filepath, format='gml')
        n2v.save(filepath, format='graphml')
        n2v.save(filepath, format='pickle')

        # List all files in the temp_dir
        files = [os.path.join(self.temp_dir, f) 
                    for f in os.listdir(self.temp_dir) 
                    if os.path.isfile(os.path.join(self.temp_dir, f))]
        
        # Filter results that match this test
        files = [f for f in files if f.find(test_name) != -1]
        
        # ++++ Testing ++++
        self.assertTrue(any([f.endswith('gml') for f in files])) # Suffix correctly added - correctness implicit as files exist
        self.assertTrue(any([f.endswith('graphml') for f in files])) # Suffix correctly added   - correctness implicit as files exist
        self.assertTrue(any([f.endswith('pickle') for f in files])) # Suffix correctly added - correctness implicit as files exist
        with self.assertRaises(AssertionError):
            n2v.save(filepath, format='made_up_extension') # Error correctly handled

    def test_3_load(self):
        '''
        Testing for:
        > error handling
        > siffix addition
        > correctness
        '''

        # Get test graphs
        graph = test_graphs['star_directed_weighted']() # Uses directed weighted graph to ensure no information is lost

        # Initialise directed or undirected graphs
        n2v = Node2Vec(edge_list=graph['edge_list'], is_directed=True)      

        # Save the graph in all three supported formats 
        test_name = 'test_loading'
        filepath = os.path.join(self.temp_dir, 'test_loading')
        n2v.save(filepath, format='gml')
        n2v.save(filepath, format='graphml')
        n2v.save(filepath, format='pickle')

        # List all files in the temp_dir
        files = [os.path.join(self.temp_dir, f) 
                    for f in os.listdir(self.temp_dir) 
                    if os.path.isfile(os.path.join(self.temp_dir, f))]
        
        # Filter results that match this test
        files = [f for f in files if f.find(test_name) != -1]

        # Load the saved graphs
        g1 = Node2Vec(is_directed=True)
        g2 = Node2Vec(is_directed=True)
        g3 = Node2Vec(is_directed=True)

        g1.load(filepath, format='gml')
        g2.load(filepath, format='graphml')
        g3.load(filepath, format='pickle')

        # Define the edge matching callable
        edge_match = iso.numerical_edge_match('weight', 1)
        
        # ++++ Testing ++++
        self.assertTrue(nx.is_isomorphic(n2v.graph, g1.graph, edge_match=edge_match)) # The loaded graph is an isomorphic match to the one that was saved - File extension success implicit
        self.assertTrue(nx.is_isomorphic(n2v.graph, g2.graph, edge_match=edge_match)) # The loaded graph is an isomorphic match to the one that was saved - File extension success implicit
        self.assertTrue(nx.is_isomorphic(n2v.graph, g3.graph, edge_match=edge_match)) # The loaded graph is an isomorphic match to the one that was saved - File extension success implicit
        with self.assertRaises(AssertionError):
            n2v.load(filepath, format='made_up_extension') # Error correctly handled    

    def test_4_get(self):
        '''
        Testing for:
        - All params returned when no argument are passed
        - specific params returned when arguments are passes
        - error raised when specified param doesnt exist
        '''

        # Initialise n2v instance with specific values
        p = 1
        q = 5
        num_walks = None
        walk_length = 'happy'
        is_directed = False

        n2v = Node2Vec(p=p, q=q, num_walks=num_walks, 
                        walk_length=walk_length, is_directed=is_directed)      

        # Get specific values
        get_p = n2v.get('p')
        get_q = n2v.get('q')
        get_num_walks = n2v.get('num_walks')
        get_walk_length = n2v.get('walk_length')
        get_is_directed = n2v.get('is_directed')
        get_p_and_p = n2v.get('p', 'q')
        get_all = n2v.get()
        
        # ++++ Testing ++++
        self.assertEqual(get_p['p'], p ) 
        self.assertEqual(get_q['q'], q ) 
        self.assertEqual(get_num_walks['num_walks'], num_walks ) 
        self.assertEqual(get_walk_length['walk_length'], walk_length ) 
        self.assertEqual(get_is_directed['is_directed'], is_directed )  
        self.assertTrue('p' in get_p_and_p.keys())
        self.assertTrue(p in get_p_and_p.values())                   
        self.assertTrue('q' in get_p_and_p.keys())
        self.assertTrue(q in get_p_and_p.values())    
        self.assertEqual(get_all['p'], p) 
        self.assertEqual(get_all['q'], q) 
        self.assertEqual(get_all['num_walks'], num_walks) 
        self.assertEqual(get_all['walk_length'], walk_length) 
        self.assertEqual(get_all['is_directed'], is_directed)   
        with self.assertRaises(AssertionError):
            _ = n2v.get('missing_parameter')                  

    def test_5_set(self):
        '''
        Testing for:
        - Attribute values are correctly updated
        - errors are handled if attribute is not found
        '''

        # Initialise n2v instance with default values
        p = 1
        q = 5
        num_walks = None
        walk_length = 'happy'
        is_directed = False

        n2v = Node2Vec()      

        # Get default values
        get_p_1 = n2v.get('p')
        get_q_1 = n2v.get('q')
        get_num_walks_1 = n2v.get('num_walks')
        get_walk_length_1 = n2v.get('walk_length')
        get_is_directed_1 = n2v.get('is_directed')

        # Set new values
        n2v.set(p=p)
        n2v.set(q=q)
        n2v.set(num_walks=num_walks, walk_length=walk_length)

        # Get default values
        get_p_2 = n2v.get('p')
        get_q_2 = n2v.get('q')
        get_num_walks_2 = n2v.get('num_walks')
        get_walk_length_2 = n2v.get('walk_length')
        get_is_directed_2 = n2v.get('is_directed')        

        # ++++ Testing ++++
        self.assertNotEqual(get_p_1['p'], get_p_2['p'] ) 
        self.assertNotEqual(get_q_1['q'], get_q_2['q'] ) 
        self.assertNotEqual(get_num_walks_1['num_walks'], get_num_walks_2['num_walks'] ) 
        self.assertNotEqual(get_walk_length_1['walk_length'], get_walk_length_2['walk_length'] )  

        self.assertEqual(get_p_2['p'], p ) 
        self.assertEqual(get_q_2['q'], q ) 
        self.assertEqual(get_num_walks_2['num_walks'], num_walks ) 
        self.assertEqual(get_walk_length_2['walk_length'], walk_length ) 

    def test_6_normalise_edges(self):
        '''
        Testing for:
        - undirected graphs are converted to directed graphs
        - edge normalisation is correct for directed graphs
        - edge normalisation is correct for undirected graphs
        '''

        # Get test graphs
        star_undirected_unweighted = test_graphs['star_undirected_unweighted']()
        star_directed_unweighted = test_graphs['star_directed_unweighted']()
        star_undirected_weighted = test_graphs['star_undirected_weighted']()
        star_directed_weighted = test_graphs['star_directed_weighted']()

        # Create n2v instances
        uu = Node2Vec(edge_list=star_undirected_unweighted['edge_list'], is_directed=False, verbose=False)
        du = Node2Vec(edge_list=star_directed_unweighted['edge_list'], is_directed=True, verbose=False)
        uw = Node2Vec(edge_list=star_undirected_weighted['edge_list'], is_directed=False, verbose=False)
        dw = Node2Vec(edge_list=star_directed_weighted['edge_list'], is_directed=True, verbose=False)  

        # Get graphs
        uu_graph = uu.graph
        du_graph = du.graph
        uw_graph = uw.graph
        dw_graph = dw.graph

        # Apply normalisation function to graphs
        uu_graph_normalised = uu._normalise_edges(uu_graph)
        du_graph_normalised = du._normalise_edges(du_graph)
        uw_graph_normalised = uw._normalise_edges(uw_graph)
        dw_graph_normalised = dw._normalise_edges(dw_graph)
      

        # ++++ Testing ++++
        self.assertTrue(uu_graph_normalised.is_directed()) # undirected graphs converted to directed graphs
        self.assertTrue(uw_graph_normalised.is_directed()) # undirected graphs converted to directed graphs
        for answer in star_undirected_unweighted['answers']:
            node_from = answer[0][0]
            node_to = answer[0][1]
            answ = answer[1]
            self.assertEqual(uu_graph_normalised[node_from][node_to]['weight'], answ) # Normalisation procedure is correct - undirected graph
        for answer in star_directed_unweighted['answers']:
            node_from = answer[0][0]
            node_to = answer[0][1]
            answ = answer[1]
            self.assertEqual(du_graph_normalised[node_from][node_to]['weight'], answ) # Normalisation procedure is correct - directed graph
        for answer in star_undirected_weighted['answers']:
            node_from = answer[0][0]
            node_to = answer[0][1]
            answ = answer[1]
            self.assertEqual(uw_graph_normalised[node_from][node_to]['weight'], answ) # Normalisation procedure is correct - undirected graph        
        for answer in star_directed_weighted['answers']:
            node_from = answer[0][0]
            node_to = answer[0][1]
            answ = answer[1]
            self.assertEqual(dw_graph_normalised[node_from][node_to]['weight'], answ) # Normalisation procedure is correct - directed graph                 

    def test_7_calculate_transition_probabilities(self):
        '''
        Testing for:
        - Calulation is correct
        - Transition probability sum is equal to one
        '''

        # Define some parameter values
        p1 = 0.5
        p2 = 0.9
        q1 = 0.7
        q2 = 0.2

        # Get test graphs
        transition_unweighted = test_graphs['transition_unweighted']
        transition_weighted = test_graphs['transition_weighted']

        # Generate expected answers
        p1q1_unweighted = transition_unweighted(p1, q1)
        p1q1_weighted = transition_weighted(p1, q1)
        p2q2_unweighted = transition_unweighted(p2, q2)
        p2q2_weighted = transition_weighted(p2, q2)

        # Create n2v instances
        p1q1_u = Node2Vec(edge_list=p1q1_unweighted['edge_list'], p=p1, q=q1, is_directed=True, verbose=False)
        p1q1_w = Node2Vec(edge_list=p1q1_weighted['edge_list'], p=p1, q=q1, is_directed=True, verbose=False)
        p2q2_u = Node2Vec(edge_list=p2q2_unweighted['edge_list'], p=p2, q=q2, is_directed=True, verbose=False)
        p2q2_w = Node2Vec(edge_list=p2q2_weighted['edge_list'], p=p2, q=q2, is_directed=True, verbose=False)

        # Get graphs
        p1q1_u_graph = p1q1_u.graph
        p1q1_w_graph = p1q1_w.graph
        p2q2_u_graph = p2q2_u.graph
        p2q2_w_graph = p2q2_w.graph

        # Apply normalisation function to graphs
        p1q1_u_graph_normalised = p1q1_u._normalise_edges(p1q1_u_graph)
        p1q1_w_graph_normalised = p1q1_w._normalise_edges(p1q1_w_graph)
        p2q2_u_graph_normalised = p2q2_u._normalise_edges(p2q2_u_graph)
        p2q2_w_graph_normalised = p2q2_w._normalise_edges(p2q2_w_graph)

        # Compute transition probabilities
        p1q1_u_graph_trans_prob = p1q1_u._calculate_transition_probabilities(p1q1_u_graph_normalised)
        p1q1_w_graph_trans_prob = p1q1_w._calculate_transition_probabilities(p1q1_w_graph_normalised)
        p2q2_u_graph_trans_prob = p2q2_u._calculate_transition_probabilities(p2q2_u_graph_normalised)
        p2q2_w_graph_trans_prob = p2q2_w._calculate_transition_probabilities(p2q2_w_graph_normalised)

      

        # ++++ Testing ++++
        for answer in p1q1_unweighted['answers']: # Calculation is correct
            node_from = answer[0][0]
            node_to = answer[0][1]
            node_end = answer[0][2]
            answ = answer[1]
            self.assertEqual(round(p1q1_u_graph_trans_prob[node_from][node_to]['second_order_probs'][node_end], 7), round(answ, 7)) # Normalisation procedure is correct - undirected graph
        
        for answer in p1q1_weighted['answers']: # Calculation is correct
            node_from = answer[0][0]
            node_to = answer[0][1]
            node_end = answer[0][2]
            answ = answer[1]
            self.assertEqual(round(p1q1_w_graph_trans_prob[node_from][node_to]['second_order_probs'][node_end], 7), round(answ, 7)) # Normalisation procedure is correct - undirected graph
        
        for answer in p2q2_unweighted['answers']: # Calculation is correct
            node_from = answer[0][0]
            node_to = answer[0][1]
            node_end = answer[0][2]
            answ = answer[1]
            self.assertEqual(round(p2q2_u_graph_trans_prob[node_from][node_to]['second_order_probs'][node_end], 7), round(answ, 7)) # Normalisation procedure is correct - undirected graph                        
        
        for answer in p2q2_weighted['answers']: # Calculation is correct
            node_from = answer[0][0]
            node_to = answer[0][1]
            node_end = answer[0][2]
            answ = answer[1]
            self.assertEqual(round(p2q2_w_graph_trans_prob[node_from][node_to]['second_order_probs'][node_end], 7), round(answ, 7)) # Normalisation procedure is correct - undirected graph

        for n in p1q1_u_graph_trans_prob.nodes: # Transition probabilities sum to 1
            for m in p1q1_u_graph_trans_prob[n].keys():
                self.assertGreater(sum(p1q1_u_graph_trans_prob[n][m]['second_order_probs'].values()), 0.999)

        for n in p1q1_w_graph_trans_prob.nodes: # Transition probabilities sum to 1
            for m in p1q1_w_graph_trans_prob[n].keys():
                self.assertGreater(sum(p1q1_w_graph_trans_prob[n][m]['second_order_probs'].values()), 0.999)

        for n in p2q2_u_graph_trans_prob.nodes: # Transition probabilities sum to 1
            for m in p2q2_u_graph_trans_prob[n].keys():
                self.assertGreater(sum(p2q2_u_graph_trans_prob[n][m]['second_order_probs'].values()), 0.999)

        for n in p2q2_w_graph_trans_prob.nodes: # Transition probabilities sum to 1
            for m in p2q2_w_graph_trans_prob[n].keys():
                self.assertGreater(sum(p2q2_w_graph_trans_prob[n][m]['second_order_probs'].values()), 0.999)

    def test_8_make_chunks(self):
        '''
        Testing for:
        - Iterable is correctly divided 
        '''

        # Make instance of node2vec
        n2v = Node2Vec()           

        # Make iterable
        iterable = [i for i in range(100)] 

        # Chunksizes
        smallest = 1
        small = 7
        medium = 13
        large = 51
        largest = 200

        # Make chunks
        smallest_chunks = n2v._make_chunks(smallest, iterable)
        small_chunks = n2v._make_chunks(small, iterable)
        medium_chunks = n2v._make_chunks(medium, iterable)
        large_chunks = n2v._make_chunks(large, iterable)
        largest_chunks = n2v._make_chunks(largest, iterable)

        # Expected number of chunks
        smallest_num_chunks = math.ceil(100/smallest)
        small_num_chunks = math.ceil(100/small)
        medium_num_chunks = math.ceil(100/medium)
        large_num_chunks = math.ceil(100/large)
        largest_num_chunks = math.ceil(100/largest)


        # ++++ Testing ++++
        counter = 0 # Smallest
        for i in smallest_chunks:
            self.assertTrue(len(i) == smallest or len(i) == (100 % smallest))
            counter += 1
        self.assertEqual(counter, smallest_num_chunks)

        counter = 0 # Small
        for i in small_chunks:
            self.assertTrue(len(i) == small or len(i) == (100 % small))
            counter += 1
        self.assertEqual(counter, small_num_chunks)

        counter = 0 # Medium 
        for i in medium_chunks:
            self.assertTrue(len(i) == medium or len(i) == (100 % medium))
            counter += 1
        self.assertEqual(counter, medium_num_chunks)

        counter = 0 # Large
        for i in large_chunks:
            self.assertTrue(len(i) == large or len(i) == (100 % large))
            counter += 1
        self.assertEqual(counter, large_num_chunks)

        counter = 0 # Largest
        for i in largest_chunks:
            self.assertTrue(len(i) == largest or len(i) == (100 % largest))
            counter += 1
        self.assertEqual(counter, largest_num_chunks)                


    def test_9_sample(self):
        '''
        Testing for:
        - Output shape is correct
        - Only nodes reachable from a given node appear in the sample 
        - There are no null values in the sample
        - A single node with an edge to itself creates a sample of only itself
        - Node frequencies reflect transition probabilities
        ''' 

        # Get test graphs
        singleton = test_graphs['singleton']()
        triangle = test_graphs['triangle']()
        line = test_graphs['line']()

        # Make instance of node2vec
        n2v_singleton = Node2Vec(singleton['edge_list'], verbose=False)       
        n2v_triangle = Node2Vec(triangle['edge_list'], verbose=False)       
        n2v_line = Node2Vec(line['edge_list'], verbose=False)       

        # Preprocess the weights
        n2v_singleton.process_weights()
        n2v_triangle.process_weights()
        n2v_line.process_weights()        

        # Get graphs
        singleton_graph = n2v_singleton.graph
        triangle_graph = n2v_triangle.graph
        line_graph = n2v_line.graph

        # Define num_walks and walk_length
        num_walks_1 = 1
        num_walks_2 = 2

        walk_length_1 = 3 # Walk will always have minimum of 2 nodes in it
        walk_length_2 = 10
        walk_length_3 = 100

        # Set params and draw samples    
        # 1
        n2v_singleton.set(num_walks=num_walks_1, walk_length=walk_length_1)
        n2v_triangle.set(num_walks=num_walks_1, walk_length=walk_length_1)
        n2v_line.set(num_walks=num_walks_1, walk_length=walk_length_1)

        walks_11_singleton = n2v_singleton._sample(list(singleton_graph.nodes), singleton_graph)
        walks_11_triangle = n2v_triangle._sample(list(triangle_graph.nodes), triangle_graph)
        walks_11_line = n2v_line._sample(list(line_graph.nodes), line_graph)

        # 2
        n2v_singleton.set(num_walks=num_walks_1, walk_length=walk_length_2)
        n2v_triangle.set(num_walks=num_walks_1, walk_length=walk_length_2)
        n2v_line.set(num_walks=num_walks_1, walk_length=walk_length_2)
                
        walks_12_singleton = n2v_singleton._sample(list(singleton_graph.nodes), singleton_graph)
        walks_12_triangle = n2v_triangle._sample(list(triangle_graph.nodes), triangle_graph)
        walks_12_line = n2v_line._sample(list(line_graph.nodes), line_graph)    

        # 3
        n2v_singleton.set(num_walks=num_walks_1, walk_length=walk_length_3)
        n2v_triangle.set(num_walks=num_walks_1, walk_length=walk_length_3)
        n2v_line.set(num_walks=num_walks_1, walk_length=walk_length_3)
                
        walks_13_singleton = n2v_singleton._sample(list(singleton_graph.nodes), singleton_graph)
        walks_13_triangle = n2v_triangle._sample(list(triangle_graph.nodes), triangle_graph)
        walks_13_line = n2v_line._sample(list(line_graph.nodes), line_graph)     

        # 4 
        n2v_singleton.set(num_walks=num_walks_2, walk_length=walk_length_2)
        n2v_triangle.set(num_walks=num_walks_2, walk_length=walk_length_2)
        n2v_line.set(num_walks=num_walks_2, walk_length=walk_length_2)
                
        walks_22_singleton = n2v_singleton._sample(list(singleton_graph.nodes), singleton_graph)
        walks_22_triangle = n2v_triangle._sample(list(triangle_graph.nodes), triangle_graph)
        walks_22_line = n2v_line._sample(list(line_graph.nodes), line_graph)               

        # Aggregate
        table = [

            [walks_11_singleton, num_walks_1, len(singleton_graph.nodes), walk_length_1],
            [walks_11_triangle, num_walks_1, len(triangle_graph.nodes), walk_length_1],
            [walks_11_line, num_walks_1, len(line_graph.nodes), walk_length_1],
            [walks_12_singleton, num_walks_1, len(singleton_graph.nodes), walk_length_2],
            [walks_12_triangle, num_walks_1, len(triangle_graph.nodes), walk_length_2],
            [walks_12_line, num_walks_1, len(line_graph.nodes), walk_length_2],
            [walks_13_singleton, num_walks_1, len(singleton_graph.nodes), walk_length_3],
            [walks_13_triangle, num_walks_1, len(triangle_graph.nodes), walk_length_3],
            [walks_13_line, num_walks_1, len(line_graph.nodes), walk_length_3],
            [walks_22_singleton, num_walks_2, len(singleton_graph.nodes), walk_length_2],
            [walks_22_triangle, num_walks_2, len(triangle_graph.nodes), walk_length_2],
            [walks_22_line, num_walks_2, len(line_graph.nodes), walk_length_2]

            ]

        # ++++ Testing ++++
        # Shape, Frequencies & Null values:
        counts = {}

        types = ['singleton', 'triangle', 'line']
        idx = 0

        for row in table:
            # Get graph type
            graph_type = types[idx % 3]
            idx += 1

            # Get values
            sample = row[0]
            num_walks = row[1]
            num_nodes = row[2]
            walk_length = row[3]
            rows = len(sample)
            cols = len(sample[0])

            # Run tests
            self.assertEqual(rows, num_walks * num_nodes) # Shape check - row
            self.assertEqual(cols, walk_length) # Shape check - col
            for i in sample:
                for j in i:
                    self.assertIsNotNone(j) # Checks no null values
                    # Count node frequencies
                    if graph_type not in counts.keys():
                        counts[graph_type] = {}
                    if j not in counts[graph_type].keys():
                        counts[graph_type][j] = 1
                    else:
                        counts[graph_type][j] += 1
        
        # Single node self loop graph loops
        self.assertEqual(len(counts['singleton'].keys()), 1)

        # Frequency reflects expected rates
        node_by_frequency = [k for k, v in sorted(counts['line'].items(), key=lambda item: item[1], reverse=True)]
        self.assertTrue(node_by_frequency[0] in line['answers'])
        self.assertTrue(node_by_frequency[1] in line['answers'])
            


    def test_11_to_csv(self):
        '''
        Testing for:
        - File written to location
        ''' 

        # Make tabular data to write out
        walks = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]    
            ]
        
        # Create a n2v instance
        n2v = Node2Vec()

        # Create a filename
        test_name = 'test_to_csv'
        filepath = os.path.join(self.temp_dir, test_name)

        # Save file
        n2v._to_csv(walks, filepath)

        # ++++ Testing ++++
        self.assertTrue(os.path.exists(f'{filepath}.csv'))

    def test_12_combine_results(self):
        '''
        Testing for:
        - Copying is accurate
        ''' 

        # Create a temporary directory to work in
        _dir = 'combine_results_test'
        temp_dir = os.path.join(self.temp_dir, _dir)
        result_fp = os.path.join(temp_dir, 'results.csv')
        os.mkdir(temp_dir)

        # Generate a random dataset
        data = []
        for i in range(30):
            row = [i]           # Give each row a unique id
            for j in range(9):
                random_string = ''.join(choice(ascii_uppercase) for i in range(10))
                row.append(random_string)
            data.append(row)
        data = pd.DataFrame(data)

        # Split the dataset into partitions
        splits = np.array_split(data, 3)

        # Save each partition in the temp dir
        for i, split in enumerate(splits):
            fp = os.path.join(temp_dir, f'{i}.csv')
            pd.DataFrame(split).to_csv(fp, index=False, header=False)

        # Call _combine_results on the temp dir
        n2v = Node2Vec()
        n2v._combine_results(temp_dir, result_fp)

        # Load the result in sort rows by unique id
        data_ = pd.read_csv(result_fp, header=None).sort_values(by=[0]).reset_index(drop=True)

        # ++++ Testing ++++
        self.assertTrue(data.equals(data_))  


    def test_13_cleanup(self):
        '''
        Testing for:
        - desired directory is removed with files
        - desired directory is removed without files
        ''' 

        # Create 2 temporary directories to work in
        _dir1 = 'directory'
        _dir2 = 'directory_with_files'
        temp_dir1 = os.path.join(self.temp_dir, _dir1)
        temp_dir2 = os.path.join(self.temp_dir, _dir2)
        os.mkdir(temp_dir1)
        os.mkdir(temp_dir2)        
        
        # Populate 1 directory with files
        for i in range(10):
            Path(os.path.join(temp_dir2, f'file-{i}.txt')).touch()

        # Call clean up on both directories
        n2v = Node2Vec()
        n2v._cleanup(temp_dir1)
        n2v._cleanup(temp_dir2)

        # ++++ Testing ++++
        self.assertFalse(os.path.exists(temp_dir1)) # removed without files
        self.assertFalse(os.path.exists(temp_dir2)) # removed with files      



    def test_14_generate_walks(self):
        '''
        Testing for:
        - Concurrent execution is successful
        - Walks are returned when no filepath is provided
        - walks are persisted when a filepath is provided 
        - pd df is retruned when a filepath is provided
        ''' 

        # Create a temporary directory
        _dir = 'walk_generation'
        temp_dir = os.path.join(self.temp_dir, _dir)
        filepath = os.path.join(self.temp_dir, 'results')
        # os.mkdir(temp_dir)

        # Create a random graph with 100 nodes
        graph = nx.barabasi_albert_graph(100, 10)
        edges = [[i, j] for i, j in graph.edges()]

        # Process weights
        n2v = Node2Vec(edge_list=edges, verbose=False, walk_length=10)
        n2v.temp_dir = temp_dir
        n2v.process_weights()

        # Generate Walks
        walks1 = n2v.generate_walks() 
        walks2 = n2v.generate_walks(filepath=filepath)
      

        # ++++ Testing ++++
        self.assertTrue(isinstance(walks1, list)) # Walks are returned when no filepath is provided
        self.assertGreater(len(walks1), 0) # Walks are returned when no filepath is provided
        self.assertIsInstance(walks2, pd.DataFrame) # walks are persisted when a filepath is provided 
        '''
        Successful concurrent execution is implicit in any of these tests passing
        as it is the method used to produce the results.
        Also, the fact that a pdDF is returned for walks2 shows the success that
        the data was persisted becuase it is read back in and returned as a pdDf
        '''


if __name__ == "__main__":
    TestNode2Vec.main()

