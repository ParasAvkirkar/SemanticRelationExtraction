import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...

        self.bidirectional_layer = layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True), merge_mode='concat')

        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...

        M = tf.tanh(rnn_outputs)
        alpha = tf.matmul(M, self.omegas)
        alpha = tf.nn.softmax(alpha, axis=1)

        ### TODO(Students) END

        return alpha

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...

        shapes = inputs.get_shape().as_list()
        batch_size = shapes[0]
        time_steps = shapes[1]

        final_embed = tf.concat([word_embed, pos_embed], axis=2)

        hidden_states = self.bidirectional_layer(final_embed, training=training)

        attention = self.attn(hidden_states)
        r = tf.multiply(hidden_states, attention)
        r = tf.reduce_sum(r, axis=1)
        h_star = tf.tanh(r)

        # h_star_dim_1 = h_star.get_shape().as_list()[1]
        # h_star_dim_2 = h_star.get_shape().as_list()[2]

        # h_star = tf.reshape(h_star, [batch_size, h_star_dim_1 * h_star_dim_2])
        logits = self.decoder(h_star)

        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...
        ### TODO(Students END

    def call(self, inputs, pos_inputs, training):
        raise NotImplementedError
        ### TODO(Students) START
        # ...
        ### TODO(Students END
