import tensorflow as tf

class Model(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, negative_samples):
        '''
        '''

        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples

        self._create_embedding_layers()

    def compile_model(self):
        '''
        '''
        self.compile(
            optimizer='adam',
            loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )        
    
    def call(self, pair):
        '''
        '''
        target, context = pair.get('target'), pair.get('context')
        
        if (len(target.shape) == 2):
            target = tf.squeeze(target, axis=1)
        
        word_embed = self.target_embedding(target)
        context_embed = self.context_embedding(context)

        dots = tf.einsum('be,bce->bc', word_embed, context_embed)

        return dots        

    def _create_embedding_layers(self):
        '''
        '''
        vocab_size = self.vocab_size
        embedding_dim = self.embedding_dim
        negative_samples = self.negative_samples


        self.target_embedding = tf.keras.layers.Embedding(
                            vocab_size,
                            embedding_dim,
                            input_length=1,
                            name='w2v_embedding'
                        )
        
        self.context_embedding = tf.keras.layers.Embedding(
                                    vocab_size,
                                    embedding_dim,
                                    input_length=negative_samples + 1        
        )