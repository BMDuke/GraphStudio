
## Test graphs for normalisation unit tests
'''
Edge lists:
These are represented as a tuple (n, m). For a directed graph this represents
a node from node n to m. For an undirected graph it represents the existence
of an edge between node n and m. 
For the weighted case it is (n, m, weight)

Answers: 
These are the hand calculated values expected to be produced by the output of 
the function. Note undirected graphs are converted into directed graphs by the
program which means  that there are answers for directed edges even in the 
undirected case. 
These are calculated by adding up the weights of edges leaving a node and 
dividing each edge by that number.

'''
#A
def star_undirected_unweighted():
    return  {
        'edge_list': [  (1, 2),
                        (1, 3),
                        (1, 4),
                        (1, 5),
                        (1, 6)  ],
        'answers': [    ((1, 2), 0.2),
                        ((1, 3), 0.2),
                        ((1, 4), 0.2),
                        ((1, 5), 0.2),
                        ((1, 6), 0.2),
                        ((6, 1), 1),
                        ((5, 1), 1),
                        ((4, 1), 1),
                        ((3, 1), 1),
                        ((2, 1), 1) ]
    }

# B
def star_directed_unweighted():
    return  {
        'edge_list': [  (1, 2),
                        (1, 3),
                        (1, 4),
                        (1, 5),
                        (1, 6),
                        (6, 1),
                        (5, 1),
                        (4, 1),
                        (3, 1),
                        (2, 1)  ],
        'answers': [    ((1, 2), 0.2),
                        ((1, 3), 0.2),
                        ((1, 4), 0.2),
                        ((1, 5), 0.2),
                        ((1, 6), 0.2),
                        ((6, 1), 1),
                        ((5, 1), 1),
                        ((4, 1), 1),
                        ((3, 1), 1),
                        ((2, 1), 1) ]
    }

# C
def star_undirected_weighted():
    return  {
        'edge_list': [  (1, 2, 5),
                        (1, 3, 4),
                        (1, 4, 3),
                        (1, 5, 2),
                        (1, 6, 1)  ],
        'answers': [    ((1, 2), 1/3),
                        ((1, 3), 4/15),
                        ((1, 4), 1/5),
                        ((1, 5), 2/15),
                        ((1, 6), 1/15),
                        ((6, 1), 1),
                        ((5, 1), 1),
                        ((4, 1), 1),
                        ((3, 1), 1),
                        ((2, 1), 1) ]
    }

# D
def star_directed_weighted():
    return {
        'edge_list': [  (1, 2, 5),
                        (1, 3, 4),
                        (1, 4, 3),
                        (1, 5, 2),
                        (1, 6, 1),
                        (6, 1, 10),
                        (5, 1, 10),
                        (4, 1, 10),
                        (3, 1, 10),
                        (2, 1, 10)  ],
        'answers': [    ((1, 2), 1/3),
                        ((1, 3), 4/15),
                        ((1, 4), 1/5),
                        ((1, 5), 2/15),
                        ((1, 6), 1/15),
                        ((6, 1), 1),
                        ((5, 1), 1),
                        ((4, 1), 1),
                        ((3, 1), 1),
                        ((2, 1), 1) ]
    }





## Test graphs for transition probability calculation
'''
Edge lists:
These are represented as a tuple (n, m). For a directed graph this represents
a node from node n to m. For an undirected graph it represents the existence
of an edge between node n and m. 
For the weighted case it is (n, m, weight)

Answers: 
These are the hand calculated values for the transition probabilities. These
are calculated by multiplying the normalised edge weights by either 1/p, 1/q or 1
depending on the local network topology as defined in https://arxiv.org/pdf/1607.00653.pdf
Because these values are dependent on 3 nodes, these are given as a tripplet (n, m, l),
which can be read n to m to l. We can calculating the transition probabilities for edge
(m, l) having come from node n. 
The answer is calculated as:
normalised_weight * search_bias / normalising constant from nodes in neighbourhood

P & Q values:
We also define p and q vaules that are used to compute the transition probabilities
'''

# E
def transition_unweighted(p, q):
    return  {
        'edge_list': [  (1, 2),
                        (1, 5),
                        (1, 6),
                        (1, 7),
                        (2, 1),
                        (2, 3),
                        (2, 4),
                        (2, 5),
                        (3, 2),
                        (4, 2),
                        (5, 2),
                        (5, 1),
                        (6, 1),
                        (7, 1),  ],
        'answers': [    ((1, 2, 3), ((1/4) * 1/q) / (((1/4) * 1/q) + ((1/4) * 1/q) + ((1/4) * 1) + ((1/4) * 1/p))),
                        ((1, 2, 4), ((1/4) * 1/q) / (((1/4) * 1/q) + ((1/4) * 1/q) + ((1/4) * 1) + ((1/4) * 1/p))),
                        ((1, 2, 5), ((1/4) * 1) / (((1/4) * 1/q) + ((1/4) * 1/q) + ((1/4) * 1) + ((1/4) * 1/p))),
                        ((1, 2, 1), ((1/4) * 1/p) / (((1/4) * 1/q) + ((1/4) * 1/q) + ((1/4) * 1) + ((1/4) * 1/p))),
                        ((1, 7, 1), (1 * 1/p) / (1 * 1/p)),
                        ((1, 6, 1), (1 * 1/p) / (1 * 1/p)),
                        ((5, 1, 7), ((1/4) * 1/q) / (((1/4) * 1/q) + ((1/4) * 1/q) + ((1/4) * 1) + ((1/4) * 1/p))),
                        ((5, 1, 6), ((1/4) * 1/q) / (((1/4) * 1/q) + ((1/4) * 1/q) + ((1/4) * 1) + ((1/4) * 1/p))),
                        ((5, 1, 5), ((1/4) * 1/p) / (((1/4) * 1/q) + ((1/4) * 1/q) + ((1/4) * 1) + ((1/4) * 1/p))),
                        ((5, 1, 2), ((1/4) * 1) / (((1/4) * 1/q) + ((1/4) * 1/q) + ((1/4) * 1) + ((1/4) * 1/p))), ]
    }

# F
def transition_weighted(p, q):
    return  {
        'edge_list': [  (1, 2, 3),
                        (1, 5, 4),
                        (1, 6, 2),
                        (1, 7, 5),
                        (2, 1, 3),
                        (2, 3, 4),
                        (2, 4, 3),
                        (2, 5, 1),
                        (3, 2, 4),
                        (4, 2, 3),
                        (5, 2, 1),
                        (5, 1, 4),
                        (6, 1, 2),
                        (7, 1, 5),  ],
        'answers': [    ((1, 2, 3), ((4/11) * 1/q) / (((4/11) * 1/q) + ((3/11) * 1/q) + ((1/11) * 1) + ((3/11) * 1/p))),
                        ((1, 2, 4), ((3/11) * 1/q) / (((4/11) * 1/q) + ((3/11) * 1/q) + ((1/11) * 1) + ((3/11) * 1/p))),
                        ((1, 2, 5), ((1/11) * 1) / (((4/11) * 1/q) + ((3/11) * 1/q) + ((1/11) * 1) + ((3/11) * 1/p))),
                        ((1, 2, 1), ((3/11) * 1/p) / (((4/11) * 1/q) + ((3/11) * 1/q) + ((1/11) * 1) + ((3/11) * 1/p))),
                        ((1, 7, 1), ((1) * 1/p) / ((1) * 1/p)),
                        ((1, 6, 1), ((1) * 1/p) / ((1) * 1/p)),
                        ((5, 1, 7), ((5/14) * 1/q) / (((5/14) * 1/q) + ((2/14) * 1/q) + ((4/14) * 1/p) + ((3/14) * 1))),
                        ((5, 1, 6), ((2/14) * 1/q) / (((5/14) * 1/q) + ((2/14) * 1/q) + ((4/14) * 1/p) + ((3/14) * 1))),
                        ((5, 1, 5), ((4/14) * 1/p) / (((5/14) * 1/q) + ((2/14) * 1/q) + ((4/14) * 1/p) + ((3/14) * 1))),
                        ((5, 1, 2), ((3/14) * 1) / (((5/14) * 1/q) + ((2/14) * 1/q) + ((4/14) * 1/p) + ((3/14) * 1))) ]
    }    

## Tests graphs for walk generation
'''
Edge lists:
These are represented as a tuple (n, m). For a directed graph this represents
a node from node n to m. For an undirected graph it represents the existence
of an edge between node n and m. 
For the weighted case it is (n, m, weight)

Answers: 
The answers included in this section are non-deterministic, and rather are used to
get an idea of the correctness of the sampling procedures (random walk generator) by 
seeing whether the most frequent nodes to appear in the sample coroborate with those
that we would expect. for this reason it will be represented as a list of the nodes we 
expect to see
'''

# G
def singleton():
    return  {
        'edge_list': [  (1, 1, 1)  ],
        'answers': [    1    ]
    } 

# G
def triangle():
    return  {
        'edge_list': [  (1, 2, 1),
                        (2, 3, 1),
                        (3, 1, 5),
                        (1, 3, 1),
                        (3, 2, 1),
                        (2, 1, 5)  ],
        'answers': [    1    ]
    }  

# H
def line():
    return  {
        'edge_list': [  (1, 2, 1),
                        (2, 3, 1),
                        (3, 4, 5),
                        (4, 5, 1),
                        (5, 6, 1),
                        (6, 5, 1),
                        (5, 4, 1),
                        (4, 3, 5),
                        (3, 2, 1),
                        (2, 1, 1),  ],
        'answers': [    3, 4    ]
    }        

## Aggregate object for export
test_graphs = {
    'star_undirected_unweighted':star_undirected_unweighted,
    'star_directed_unweighted':star_directed_unweighted,
    'star_undirected_weighted':star_undirected_weighted,
    'star_directed_weighted':star_directed_weighted,
    'transition_unweighted':transition_unweighted,
    'transition_weighted':transition_weighted,
    'singleton':singleton,
    'triangle':triangle,
    'line':line
}    