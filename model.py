
"""
Run in tf 2.0 environment.
"""
import tensorflow as tf


def train_input_fn(train_source_tensor, train_target_tensor, buffer_size, EPOCH, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((train_source_tensor, train_target_tensor))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(EPOCH)
    return dataset


def encoder(source_tensor, encoder_state, vocab_size, embed_size, units, activation, dropout_rate):
    #Encoder + 4*LSTM layer
    #TODO: verify LSTM parameters
    embed_func = tf.keras.layers.Embedding(vocab_size, embed_size, name="Encoder/embedding")
    # by default, initial state is zero in lstm implementation.
    # The size of initial state should be [batch_size,enc_units]
    lstm_func_with_init_1 = tf.keras.layers.LSTM(units = units, activation = 'tanh', kernel_initializer='glorot_uniform',
                                     recurrent_initializer='glorot_uniform', dropout = dropout_rate,
                                     return_sequences=True, return_state=True, name="Encoder/Lstm1")
    lstm_func_with_init_2 = tf.keras.layers.LSTM(units = units, activation = 'tanh', kernel_initializer='glorot_uniform',
                                     recurrent_initializer='glorot_uniform', dropout = dropout_rate,
                                     return_sequences=True, return_state=True, name="Encoder/Lstm2")
    lstm_func_with_init_3 = tf.keras.layers.LSTM(units = units, activation = 'tanh', kernel_initializer='glorot_uniform',
                                     recurrent_initializer='glorot_uniform', dropout = dropout_rate,
                                     return_sequences=True, return_state=True, name="Encoder/Lstm3")
    lstm_func_with_init_4 = tf.keras.layers.LSTM(units = units, activation = 'tanh', kernel_initializer='glorot_uniform',
                                     recurrent_initializer='glorot_uniform', dropout = dropout_rate,
                                     return_sequences=True, return_state=True, name="Encoder/Lstm4")

    x = embed_func(source_tensor)
    #TODO : check how initial state helps LSTM learn
    x, h1, c1 = lstm_func_with_init_1(x, initial_state=encoder_state[0])
    x, h2, c2 = lstm_func_with_init_2(x, initial_state=encoder_state[1])
    x, h3, c3 = lstm_func_with_init_3(x, initial_state=encoder_state[2])
    x, h4, c4 = lstm_func_with_init_4(x, initial_state=encoder_state[3])
    return_state = [[h1, c1], [h2, c2], [h3, c3], [h4, c4]]
    train_layer_var = [embed_func, lstm_func_with_init_1, lstm_func_with_init_2,
                       lstm_func_with_init_3, lstm_func_with_init_4]
    trainable_var = []
    for layer_name in train_layer_var:
        # assume that layer_name.trainable_weights is a instance of list
        trainable_var += layer_name.trainable_weights
    return x, return_state, trainable_var



class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, units, method, batch_size, dropout_rate):

        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.units = units

        self.embed_func = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.lstm_func_with_init_1 = tf.keras.layers.LSTM(units=units, activation='tanh',
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='glorot_uniform', dropout=dropout_rate,
                                                     return_sequences=True, return_state=True)
        self.lstm_func_with_init_2 = tf.keras.layers.LSTM(units=units, activation='tanh',
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='glorot_uniform', dropout=dropout_rate,
                                                     return_sequences=True, return_state=True)
        self.lstm_func_with_init_3 = tf.keras.layers.LSTM(units=units, activation='tanh',
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='glorot_uniform', dropout=dropout_rate,
                                                     return_sequences=True, return_state=True)
        self.lstm_func_with_init_4 = tf.keras.layers.LSTM(units=units, activation='tanh',
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='glorot_uniform', dropout=dropout_rate,
                                                     return_sequences=True, return_state=True)
        self.hidden_state_dense = tf.keras.layers.Dense(embed_size, activation='tanh')
        # updated hidden_state available for now, then reduce to y_t
        self.output_state_dense = tf.keras.layers.Dense(vocab_size)


    def call(self, target_tensor, state, out, attention_state):
        # TODO : check input tensor shape
        # TODO : check concat result
        # TODO : what's difference between LSTM output and state?
        # TODO : why initialize h_t state(attention_state) to 0?
        x = self.embed_func(target_tensor)
        x = tf.concat([x, attention_state],axis=-1)
        # TODO: check why this happen


        x, h1, c1 = self.lstm_func_with_init_1(x, initial_state=state[0])
        # tf.print(state[0][0].shape)
        # tf.print(state[0][1].shape)
        # tf.print(state[1][0].shape)
        # tf.print(state[1][0].shape)
        x, h2, c2 = self.lstm_func_with_init_2(x, initial_state=state[1])
        x, h3, c3 = self.lstm_func_with_init_3(x, initial_state=state[2])
        x, h4, c4 = self.lstm_func_with_init_4(x, initial_state=state[3])
        return_state = [[h1, c1], [h2, c2], [h3, c3], [h4, c4]]
        # TODO: Verify size after this stage
        # TODO: add another two scoring, adapt to three functions
        score = tf.matmul(out, x, transpose_b=True)
        global_alignment = tf.nn.softmax(score, axis=1)
        global_context = tf.reduce_sum(global_alignment * out, axis=1)
        hidden_state = tf.concat([tf.expand_dims(global_context, 1), x],axis = -1)
        # TODO : Verify activation
        # TODO : change dense layer
        # TODO : Verify size after this stage
        hidden_state = self.hidden_state_dense(hidden_state)
        output_state = self.output_state_dense(hidden_state)
        output_state = tf.squeeze(output_state, axis=1)
        # train_layer_var = [embed_func, lstm_func_with_init_1, lstm_func_with_init_2, lstm_func_with_init_3,
        #                    lstm_func_with_init_4, hidden_state_dense, output_state_dense]
        # trainable_var = []
        # for idx, layer_name in enumerate(train_layer_var):
        #     # assume that layer_name.trainable_weights is a instance of list
        #     trainable_var += layer_name.trainable_weights
        # return output_state, return_state, hidden_state, trainable_var
        return output_state, return_state, hidden_state








