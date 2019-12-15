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

        self.bidirectional_layer = layers.Bidirectional(layers.GRU(units=hidden_size, return_sequences=True), merge_mode='concat')

        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...

        M = tf.tanh(rnn_outputs)

        alpha = tf.tensordot(M, self.omegas, axes=[2, 0])
        alpha = tf.nn.softmax(alpha, axis=1)
        r = tf.multiply(rnn_outputs, alpha)
        r = tf.reduce_sum(r, axis=1)
        h_star = tf.tanh(r)

        ### TODO(Students) END

        return h_star

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        # pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...

        shapes = inputs.get_shape().as_list()
        batch_size = shapes[0]
        time_steps = shapes[1]

        sequence_mask = tf.cast(inputs != 0, tf.float32)

        # final_embed = tf.concat([word_embed, pos_embed], axis=2)
        final_embed = word_embed

        hidden_states = self.bidirectional_layer(final_embed, training=training, mask=sequence_mask)

        h_star = self.attn(hidden_states)

        logits = self.decoder(h_star)

        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # self.hidden_size = hidden_size
        self.training = training


        # window-size = 2
        self.first_cnn_layer = layers.Conv1D(128, kernel_size=(2), input_shape=(None, self.embed_dim*2), activation="tanh")
        self.first_max_pool = layers.GlobalMaxPool1D()

        # window-size = 3
        self.second_cnn_layer = layers.Conv1D(128, kernel_size=(3), input_shape=(None, self.embed_dim*2), activation="tanh")
        self.second_max_pool = layers.GlobalMaxPool1D()

        # window-size = 4
        self.third_cnn_layer = layers.Conv1D(128, kernel_size=(4), input_shape=(None, self.embed_dim*2), activation="tanh")
        self.third_max_pool = layers.GlobalMaxPool1D()
        #

        self.num_classes = len(ID_TO_CLASS)

        self.dropout_layer = layers.Dropout(0.5)
        self.decoder = layers.Dense(units=self.num_classes)
        # self.omegas = tf.Variable(tf.random.normal((hidden_size * 2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students END

    def call(self, inputs, pos_inputs, training):
        ### TODO(Students) START
        # ...

        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        shapes = inputs.get_shape().as_list()
        batch_size = shapes[0]
        time_steps = shapes[1]

        final_embed = tf.concat([word_embed, pos_embed], axis=2)

        first_conv_output = self.first_cnn_layer(final_embed)
        first_max_output = self.first_max_pool(first_conv_output)

        second_conv_output = self.second_cnn_layer(final_embed)
        second_max_output = self.second_max_pool(second_conv_output)

        third_conv_output = self.third_cnn_layer(final_embed)
        third_max_output = self.third_max_pool(third_conv_output)

        # fourth_conv_output = self.fourth_cnn_layer(final_embed)
        # fourth_max_output = self.fourth_max_pool(fourth_conv_output)

        # final_max_pool = tf.concat([first_max_output, second_max_output, third_max_output, fourth_max_output], axis=-1)
        final_max_pool = tf.concat([first_max_output, second_max_output, third_max_output], axis=-1)
        # final_max_pool = first_max_output

        if training:
            final_max_pool = self.dropout_layer(final_max_pool)

        logits = self.decoder(final_max_pool)
        ### TODO(Students END

        return {'logits': logits}
