import tensorflow as tf


def train_input_fn(train_source_tensor, train_target_tensor, buffer_size, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((train_source_tensor, train_target_tensor))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(BATCH_SIZE)
    print('dataset shape (batch_size, max_len):', dataset)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def encoder(vocab_input_size, embedding_dim, units, batch_size, dropout_rate):
    pass


def decoder():
    pass