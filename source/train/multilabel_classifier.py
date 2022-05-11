import os
import shutil

import tensorflow as tf

from source.train.model import Model
from source.train.node2vec import Node2Vec
from source.datasets.text import TextDataset
from source.datasets.multi_label import MultiLabelDataset
from source.etl.vertices import Vertices
from source.etl.msig import MSig

class MultiLabelClassifier(Model):

    # Class attrs
    data_dir = 'data/processed/vertices'
    walk_dir = 'data/processed/walks'    
    source_dir = 'source/models/multiLabelClassifier'

    def __init__(self):
        '''
        '''
        super().__init__()
    
    def train(self, experiment):
        '''
        mlc/bc:
        - ✓ Get specified mlc/bc experiment config
        - ✓ Get associated node2vec experment config
        - ✓ Get associated skipgram experiment config
        - ✓ Load n2v instance
        - ✓ Load associated data asset
        - ✓ Create new mlc/bc instance from n2v
        - ✓ Create callbacks
        - ✓ Create new experiment directory
        - ✓ Compile model
        - ✓ Train model
        - ✓ Save model       
        ''' 

        model = 'multi_label' # Must be in config experiment mapping keys

        # Get multilabel classifier experimental config
        try:
            
            model_conf = self.config.get_experiment(model, experiment)
        
        except Exception as e:

            print(f'ERROR: Experiment name \'{experiment}\' not found for {model}.\n {e}')
            raise

        # Get associated node2vec experimental config
        encoder_experiment = model_conf.get('encoder')
        try:
            
            encoder_conf = self.config.get_experiment('node2vec', encoder_experiment)
        
        except Exception as e:

            print(f'ERROR: Experiment name \'{encoder_experiment}\' not found for node2vec.\n {e}')
            raise        


        # Extract the associated data config
        data_experiment = encoder_conf.get('data')
        try:
            
            data_conf = self.config.get_experiment('skipgram', data_experiment)
        
        except Exception as e:

            print(f'ERROR: Experiment name \'{data_experiment}\' not found for data.\n {e}')
            raise


        # Create the transfer ID from the encoder config and the data config
        # This is the uuid for a given experiment in the encoder
        skipgram_params = list(data_conf.keys())
        skipgram_uuid = self.data_archive.make_id('skipgram', *skipgram_params)
        transfer_id = self.model_archive.make_id('node2vec', encoder_experiment, skipgram_uuid, None)

        # Get text vectorization layer
        walks_uuid = self.data_archive.make_id('walk', *filter(lambda x: not ((x == 'window_size') or (x == 'negative_samples')),skipgram_params))
        walks_fp = os.path.join(self.walk_dir, f'{walks_uuid}.csv')
        assert os.path.exists(walks_fp), f'ERROR: {walks_uuid} walks dataset does not exist. Have you processed the dataset?'        

        text_vectorizer = TextDataset(walks_fp, '').text_vectorizer()
        
        # Check encoder experiment exists
        encoder_root = os.path.join(self.base_dir, transfer_id) 
        assert os.path.exists(encoder_root), f'ERROR: encoder \'{encoder_experiment}\' does not exist. Have you trained the encoder?'

        # Load the node2vec instance and extract the embedding layer
        encoder_model = os.path.join(encoder_root, 'model')
        node2vec = Node2Vec()
        node2vec._load_model(encoder_model)
        encoder = node2vec.model.get_layer(name='w2v_embedding') 
        
        # Calculate the number of labels in the dataset
        msig = MSig()
        labelled_genes = msig._load_msig()
        num_labels = len(set([label for label, _ in labelled_genes.values]))

        # Load the training data
        training_data_filename = Vertices(self.config).out_file
        training_data_fp = os.path.join(self.data_dir, training_data_filename)
        multilabel_dataset = MultiLabelDataset(filepath=training_data_fp, num_labels=num_labels)

        # Split the data into train and test sets
        split = model_conf.get('split')
        rows = len(list(multilabel_dataset.data().unbatch().as_numpy_iterator()))
        batch_size = [batch[0].shape[0] for batch in multilabel_dataset.data().take(1)][0]
        train_size, test_size = round(split * rows), round((1 - split) * rows)
        data = multilabel_dataset.data().unbatch()
        train = data.take(train_size).batch(batch_size)
        test = data.skip(train_size).batch(batch_size)

        # Generate a new UUID for this traning experiment
        dataset_name = training_data_filename.split('.')[0]
        experiment_id = self.model_archive.make_id(model, experiment, dataset_name, transfer_id, add_to_lookup=True)

        # Create experiment directory
        root, log_dir, chckpt_dir, model_dir = self._new_experiment(experiment_id) 

        # Create callbacks
        tensorboard_callback = self._get_callback_tensorboard(log_dir)
        checkpoint_callback = self._get_callback_checkpoints(chckpt_dir, 'accuracy')

        # Get model architecture and check it exists
        architecture = model_conf.get('architecture')
        model_fp = os.path.join(self.source_dir, f'{architecture}.py')
        assert os.path.exists(model_fp), f'ERROR: File {architecture}.py not found.'        

        # Import model and instantiate
        tf_model = self._import_model(model_fp)
        multilabel_classifier = tf_model(text_vectorizer, encoder, num_labels)

        # Compile model
        multilabel_classifier.compile_model()

        # Train model
        try: 
            multilabel_classifier.fit(  train.unbatch(),
                                        epochs=5,
                                        # epochs=model_conf.get('epochs'),
                                        callbacks=[checkpoint_callback, tensorboard_callback]
                                    )
        except Exception as e:
            shutil.rmtree(root)
            print(e)
            exit()

        # Save model
        self._save_model(multilabel_classifier, model_dir)

        # Evaluate model
        for example in test.unbatch().take(5):
            y_hat = multilabel_classifier(example[0], training=False)
            print(y_hat)
            print(example[1])

        # Launch tensorboard
        self._launch_tensorboard_(log_dir)




    def test(self, experiment):
        '''
        '''
        pass

    def predict(self, experiment):
        '''
        '''
        pass        





if __name__ == "__main__":
    multilabel_classifier = MultiLabelClassifier()
    multilabel_classifier.train('default')
  