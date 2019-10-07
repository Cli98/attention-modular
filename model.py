import tensorflow as tf


def train_input_fn(train_source_tensor, train_target_tensor, buffer_size, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((train_source_tensor, train_target_tensor))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(BATCH_SIZE)
    print('dataset shape (batch_size, max_len):', dataset)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def encoder(features, vocab_input_size, embedding_dim, units, batch_size, dropout_rate):
    # TODO: what about sequence_length?
    # TODO: change multi-lstm ceil
    input_layer = tf.contrib.layers.embed_sequence(
        features,
        vocab_input_size,
        embedding_dim,
        initializer="uniform")
    stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper
                                                     (tf.nn.rnn.BasicLSTMCell(units),output_keep_prob=1-dropout_rate)
                                                     for _ in range(4)])
    initial_state = stacked_lstm_cell.zero_state(batch_size, dtype=tf.float32)
    output, final_states = tf.nn.dynamic_rnn(
        stacked_lstm_cell, input_layer, sequence_length=None, dtype=tf.float32, initial_state = initial_state)

    #ceil state, hidden state
    return output, final_states



def decoder(source_output, source_state, target_input, h_pre_t, vocab_input_size, embedding_dim, units, dropout_rate):
    input_layer = tf.contrib.layers.embed_sequence(
        target_input,
        vocab_input_size,
        embedding_dim,
        initializer="uniform")
    input_layer_2 = tf.concat([input_layer, h_pre_t], axis=-1)
    stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper
                                                     (tf.nn.rnn.BasicLSTMCell(units),output_keep_prob=1-dropout_rate)
                                                     for _ in range(4)])
    decode_output, final_states = tf.nn.dynamic_rnn(
        stacked_lstm_cell, input_layer_2, sequence_length=None, dtype=tf.float32, initial_state = source_state)
    #ceil state, hidden state
    merged_output = source_output + decode_output
    score = tf.layers.dense(merged_output, units, activation="tanh")
    score = tf.layers.dense(score, 1,activation=None)
    softmax_score = tf.nn.softmax(score, axis=1)
    context_vector = tf.reduce_sum(source_output*softmax_score, axis = 1)
    merged_output_2 = tf.concat([tf.expand_dims(context_vector, 1), decode_output], axis=-1)
    h_bar_t = tf.layers.dense(merged_output_2, embedding_dim, activation='tanh')
    y_t = tf.keras.layers.Dense(h_bar_t, vocab_input_size, activation=None)
    # TODO: check shape of y_t
    y_t = tf.squeeze(y_t, axis=1)
    return y_t, final_states, h_bar_t

def model_fn(features, mode, param, train_target_tokenizer, vocab_input_size,
             embedding_dim, units, batch_size, dropout_rate):
    isTrain = mode == tf.estimator.ModeKeys.TRAIN
    h_t = tf.zeros((batch_size, 1, embedding_dim))
    source_output, source_state = encoder(features['x'], vocab_input_size, embedding_dim, units,
                                          batch_size, dropout_rate)
    dec_input = tf.expand_dims([train_target_tokenizer.word_index['<s>']] * batch_size, 1)

    if mode == tf.estimator.ModeKeys.TRAIN:
        for idx in range(1, features['y'].shape[1]):
            prediction, target_state, h_t = decoder(source_output, source_state, dec_input, h_t,
                                                vocab_input_size, embedding_dim, units, dropout_rate)
            mask = tf.math.logical_not(tf.math.equal(features['y'][:, idx], 0))
            loss = tf.losses.SparseCategoricalCrossentropy(features['y'][:, idx], prediction, from_logits=True, reduction='none')
            mask = tf.cast(mask, dtype=loss.dtype)
            loss *= mask
            loss = tf.reduce_mean(loss)
            dec_input = tf.expand_dims(features['y'][:, idx], 1)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)




