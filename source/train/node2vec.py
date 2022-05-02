import os
import shutil

import tensorflow as tf

from source.train.model import Model
from source.datasets.text import TextDataset

class Node2Vec(Model):

    # Class attrs
    data_dir = 'data/processed/skipgrams'
    walk_dir = 'data/processed/walks'
    source_dir = 'source/models/node2vec'

    def __init__(self):
        '''
        '''
        super().__init__()
    
    def train(self, experiment):
        '''
        n2v:
        - ✓ Get specified n2v experiement config
        - ✓ Get associated skipgram config
        - ✓ Create new n2v instance
        - ✓ Load skipgram dataset
        - ✓ Create callbacks
        - ✓ Create new experiment directory
        - ✓ Compile model
        - ✓ Train model
        - ✓ Save Model        
        ''' 

        model = 'node2vec'

        # Get node2vec experimental config
        try:
            
            model_conf = self.config.get_experiment(model, experiment)
        
        except Exception as e:

            print(f'ERROR: Experiment name \'{experiment}\' not found for node2vec.\n {e}')
            raise

        # Extract the associated data config
        data_experiment = model_conf.get('data')
        try:
            
            data_conf = self.config.get_experiment('skipgram', data_experiment)
        
        except Exception as e:

            print(f'ERROR: Experiment name \'{data_experiment}\' not found for data.\n {e}')
            raise

        # Make data uuid and assert the file exists 
        skipgram_params = list(data_conf.keys())

        walks_uuid = self.data_archive.make_id('walk', *filter(lambda x: not ((x == 'window_size') or (x == 'negative_samples')),skipgram_params))
        walks_fp = os.path.join(self.walk_dir, f'{walks_uuid}.csv')
        assert os.path.exists(walks_fp), f'ERROR: {walks_uuid} walks dataset does not exist. Have you processed the dataset?'

        skipgram_uuid = self.data_archive.make_id('skipgram', *skipgram_params)
        skipgram_fp = os.path.join(self.data_dir, f'{skipgram_uuid}.tfrecord')
        assert os.path.exists(skipgram_fp), f'ERROR: {skipgram_uuid} skipgram dataset does not exist. Have you processed the dataset?'

        # Extract experiment architecture from model config and assert exists
        architecture = model_conf.get('architecture')
        model_fp = os.path.join(self.source_dir, f'{architecture}.py')
        assert os.path.exists(model_fp), f'ERROR: File {architecture}.py not found.'

        # Load skipgram dataset
        skipgrams = TextDataset(walks_fp, skipgram_fp, negative_samples=data_conf.get('negative_samples'))

        # Generate a uuid for this training experiment
        transfer_id = None
        experiment_id = self.model_archive.make_id(model, experiment, skipgram_uuid, transfer_id, add_to_lookup=True)

        # Create experiment directory
        root, log_dir, chckpt_dir, model_dir = self._new_experiment(experiment_id) 
        
        # Create callbacks
        tensorboard_callback = self._get_callback_tensorboard(log_dir)
        checkpoint_callback = self._get_callback_checkpoints(chckpt_dir, 'accuracy')

        # Import model and instantiate
        tf_model = self._import_model(model_fp)
        vocab_size = skipgrams.text_vectorizer().vocabulary_size()
        embedding_dim = model_conf.get('embedding_dim')
        negative_samples = data_conf.get('negative_samples')
        node2vec = tf_model(vocab_size, embedding_dim, negative_samples)

        # Compile model
        node2vec.compile_model()

        # Train model
        node2vec.fit(   skipgrams.data(), 
                        # epochs=model_conf.get('epochs'),
                        epochs=1, 
                        callbacks=[tensorboard_callback, checkpoint_callback]
                    )

        # Save model
        self._save_model(node2vec, model_dir)

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
    node2vec = Node2Vec()
    node2vec.train('default')
  