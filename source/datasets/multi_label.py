import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import pandas as pd
import numpy as np
import tqdm

class MultiLabelDataset(object):

    '''
    This class is responsible for the creation and management of multi label
    dataset objects which are built using the tensorflow TFRecord dataset. 

    About: 
    This class takes a list of labelled objects and returns a table of objects 
    where labels are binary encoded (multihot encoding). It then saves this as a
    TFRecods dataset allowing it to be loaded back in.


    Public API:
     - create_dataset()         This takes a list of tuples of labelled objects
                                (object_id, [label_1,  ..., label_n]) and comverts 
                                it into a multihot encoded tf dataset
     - create_encoder()         This generates the encoder which multi-hot-encodes
                                the labels of examples. It creates the decoder as a 
                                side effect.
     - encode_example()         This multi-hot-encodes a single example
     - decode_example()         This converts an encoded example back into class labels
     - save_dataset()           This saved a dataset as a tfrecords dataset
     - data()                   This (loads and) returns a tfrecords dataset of
                                {
                                    x: object id,
                                    y: multi-hot encoding
                                }

    '''



    def __init__(self, pairs=None, filepath=None, verbose=True):
        '''
        This takes either pairs or tables as initilisation arguments. 
        - pairs:            Individually labelled objects
                            [id1, class1]
                            [id2, class1]
                            ....
        - filepath:         Location of a dataset

        '''

        self.pairs = pairs
        self.filepath = filepath
        self.verbose = verbose

    def create_dataset(self, pairs=None):
        '''
        Transform a list of (object, label) tuples into a multihot 
        encoded table. 
         - pairs:      list of tuples, list of lists, ndarray
        '''

        if pairs is None:
            pairs = self.pairs
        
        assert pairs is not None, 'ERROR: no value provided for .encode(pairs= ). Expecting list of labelled ids [(id, label),...]'

        # Group pairs by object identifier
        group = self._group_by_identifier(pairs)

        # Create an encoder
        encoder = self.create_encoder(pairs)

        # Encode 
        slices = {'x':[], 'y':[]} 
        for identifier, labels in group.items():
            slices['x'].append(str(identifier))
            slices['y'].append(encoder(labels))

        # Make dataset
        dataset = tf.data.Dataset.from_tensor_slices(slices)

        return dataset
    
    def create_encoder(self, pairs):
        '''
        This takes the list of labelled objects [(label, object_id),...]
        and generates an encoder which multi-hot encodes a set of labels.
        ie. [label_0, label_3] => [1,0,0,1,0].

        This also creates a decoder as a side effect based on the vocabulary of the 
        encoder
        '''

        # Extract and sort labels to get consistent encoder
        labels = [label for label, _ in pairs]
        labels = set(labels)
        labels = sorted(labels)

        # Create encoder
        encoder = tf.keras.layers.StringLookup(output_mode='multi_hot', num_oov_indices=0)
        encoder.adapt(labels)

        # Add to object
        self.encoder = encoder
        self.decoder = self._create_decoder(encoder)

        return encoder
    
    def encode_example(self, example):
        '''
        Assumes example is {id:[label1, label2]}
        '''

        if len(example) > 1:
            raise ValueError(example, 'Length > 1. Can only encode single examples')

        x = list(example.keys())[0]
        y = list(example.values())[0]

        example = {
            'x': str(x),
            'y': self.encoder(y)
        }

        return example
    
    def decode_example(self, example):
        '''
        Assumes example is {x: 'id', 'y': Tensor}
        '''
       
        if len(example) > 2:
            raise ValueError(example, 'Length > 2. Can only dencode single examples')

        x = example['x']
        y = example['y']

        example = {
            'x': int(x),
            'y': self.decoder(y)
        }

        return example
    
    def save_dataset(self, dataset, filepath):
        '''
        This function write a TF dataset out to disk as a TFRecords dataset
        '''

        try:
            
            with tf.io.TFRecordWriter(filepath) as file_writer:

                for record in tqdm.tqdm(dataset):

                    proto = self._serialize_example(record)

                    file_writer.write(proto)
        
        except Exception as e:

            print(f'ERROR: {e}')
            raise

    def data(self, filepath=None):
        '''
        This loads a TFRecords dataset and returns it to the user
        '''

        if filepath is None:
            if self.filepath is None:
                raise ValueError('Multilabelled gene dataset not provided')
            else:
                filepath = self.filepath
                
        dataset = tf.data.TFRecordDataset([self.filepath])
        
        dataset = dataset.map(self._deserialize_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(1024).cache().prefetch(tf.data.AUTOTUNE)

        return dataset

    
    def _create_decoder(self, encoder):
        '''
        This creates the corresponding decoder for the encoder that is
        created above. It indexes the vocabulary that is created by the 
        multi-hot encoder and returns the values of the matching labels. 
        
        This encloses the vocabulary in a function and returns the function. 
        '''

        vocab = encoder.get_vocabulary()

        def decoder(example):
            hot_indices = np.argwhere(example == 1.0)[..., 0]
            return np.take(vocab, hot_indices)

        return decoder

    def _group_by_identifier(self, pairs):
        '''
        Assumes [label, identifier] order and returns a dict of 
        {
            identifier: ...,
            labels: [label_1, ..., label_n]
        }
        '''

        group = {}

        for label, identifier in pairs:
            if identifier not in group.keys():
                group[identifier] = [label]
            else:
                group[identifier].append(label)       

        return group 

    def _serialize_example(self, example):
        '''
        This is assumed to be applied to every element in a TF dataset
        using an interface such as .map()
        '''

        x, y = example['x'], example['y']

        feature = {
            'x': self._tensor_feature(x), 
            'y': self._tensor_feature(y)
        }  

        proto = tf.train.Example(features=tf.train.Features(feature=feature))

        return proto.SerializeToString()
    
    def _deserialize_example(self, example):
        '''
        This deserialises an example proto from a tfrecord file and returns the 
        example as an (x, y) tuple
        '''

        description = {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
        }

        example = tf.io.parse_single_example(example, description)

        x = tf.io.parse_tensor(example['x'], out_type=tf.string)
        y = tf.io.parse_tensor(example['y'], out_type=tf.float32) 

        return (x, y)



    def _tensor_feature(self, tensor):
        '''
        Serialise a tensor into a proto feature
        '''

        serialized_tensor = tf.io.serialize_tensor(tensor)

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_tensor.numpy()]))        







if __name__ == "__main__":
    ml = MultiLabelDataset()
    # df = pd.read_csv('data/processed/gene_ids/gene_ids_labelled.csv')
    # print(df.head())
    # print(df.to_numpy())
    # data = ml.create_dataset(df.to_numpy())
    # enc_examp = ml.encode_example({3726: ['HALLMARK_XENOBIOTIC_METABOLISM', 'HALLMARK_WNT_BETA_CATENIN_SIGNALING']})
    # print(enc_examp)
    # print(len(enc_examp))
    # dec_examp = ml.decode_example(enc_examp)
    # print(dec_examp)

    # print(data.take(1).element_spec['x'])
    fp = 'data/processed/vertices'
    # ml.save_dataset(data, os.path.join(fp, 'mlc'))
    ml = MultiLabelDataset(filepath=os.path.join(fp, 'mlc'))
    dataset = ml.data()
    for i in dataset.take(1):
        print(i)

