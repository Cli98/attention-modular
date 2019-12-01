
"""
Run in tf 2.0 environment.
"""
import tensorflow as tf
from functools import partial


def train_input_fn(train_source_tensor, train_target_tensor, buffer_size, EPOCH, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((train_source_tensor, train_target_tensor))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(EPOCH)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def generator_input_fn(train_source_tensor, train_target_tensor, buffer_size, EPOCH, BATCH_SIZE, MAX_LEN):
    dataset = tf.data.Dataset.from_generator(generator_wrapper(train_source_tensor,
                      train_target_tensor,
                      BATCH_SIZE,
                      limit=None), (tf.int32, tf.int32), ([MAX_LEN],[MAX_LEN]))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(EPOCH)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def generator_wrapper(source_data,
                      target_data,
                      batch_size,
                      limit=None):
    return partial(nmt_generator, source_data,
                      target_data,
                      batch_size,
                      limit=None)

def nmt_generator(source_data,
                      target_data,
                      batch_size,
                      limit=None):
    total_amount_processed = 0
    for i, (lin, lout) in enumerate(zip(source_data, target_data)):
        #yield data_in, len_in, data_out, len_out
        yield lin[::-1], lout
        #len_in.append(len(in_text))
        #len_out.append(len(out_text))
        total_amount_processed += 1
        if limit and total_amount_processed>limit:
            break

def process_sentence(sequence, vocab, pad_length):
    # This function I referred from https://github.com/thisisiron/nmt-attention-tf/blob/master/utils.py
    # TODO : check what's the result of this?
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n',
                                                           oov_token='<unk>')
    lang_tokenizer.word_index = vocab
    tensor = lang_tokenizer.texts_to_sequences(sequence)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=pad_length, padding='post')
    return tensor, lang_tokenizer

def nmt_generator_from_data(source_data_path = r"./data/train.en",
                      target_data_path = r'./data/train.de',
                      batch_size = 128,
                      limit=None,
                      source_vocabulary_data = None, target_vocabulary_data = None,
                      pad_length = 90
                            ):
    with open(source_data_path, "r") as f_in, open(target_data_path) as f_out:
        prev_batch = 0
        data_in    = []
        data_out   = []
        #len_in     = []
        #len_out    = []
        total_amount_processed = 0
        for i, (lin, lout) in enumerate(zip(f_in, f_out)):
            if i - prev_batch >= batch_size:
                prev_batch = i
                #yield data_in, len_in, data_out, len_out
                train_source_tensor, train_source_tokenizer = process_sentence(data_in,source_vocabulary_data,pad_length)
                train_target_tensor, train_target_tokenizer = process_sentence(data_out,target_vocabulary_data,pad_length)
                yield train_source_tensor, train_source_tokenizer, train_target_tensor, train_target_tokenizer
                data_in  = []
                data_out = []
                #len_in   = []
                #len_out  = []
            in_text = lin[::-1]
            out_text = lout
            #len_in.append(len(in_text))
            #len_out.append(len(out_text))
            data_in.append(in_text)
            data_out.append(out_text)
            total_amount_processed += 1
            if limit and total_amount_processed>limit:
                break

        if (i + 1) % batch_size == 0:
            #yield data_in, len_in, data_out, len_out
            train_source_tensor, train_source_tokenizer = process_sentence(data_in, source_vocabulary_data, pad_length)
            train_target_tensor, train_target_tokenizer = process_sentence(data_out, target_vocabulary_data, pad_length)
            yield train_source_tensor, train_source_tokenizer, train_target_tensor, train_target_tokenizer

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, units, dropout_rate, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
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


    def call(self, source_tensor, encoder_state, activation):
        #Encoder + 4*LSTM layer
        #TODO: verify LSTM parameters

        # by default, initial state is zero in lstm implementation.
        # The size of initial state should be [batch_size,enc_units]
        x = self.embed_func(source_tensor)
        #TODO : check how initial state helps LSTM learn
        x, h1, c1 = self.lstm_func_with_init_1(x, initial_state=encoder_state[0])
        x, h2, c2 = self.lstm_func_with_init_2(x, initial_state=encoder_state[1])
        x, h3, c3 = self.lstm_func_with_init_3(x, initial_state=encoder_state[2])
        x, h4, c4 = self.lstm_func_with_init_4(x, initial_state=encoder_state[3])
        return_state = [[h1, c1], [h2, c2], [h3, c3], [h4, c4]]
        # train_layer_var = [embed_func, lstm_func_with_init_1, lstm_func_with_init_2, lstm_func_with_init_3, lstm_func_with_init_4]
        # trainable_var = []
        # for layer_name in train_layer_var:
        #     # assume that layer_name.trainable_weights is a instance of list
        #     trainable_var += layer_name.trainable_weights
        # for val in trainable_var:
        #     watcher.watch(val)
        # return x, return_state, trainable_var, watcher
        return x, return_state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_size, self.units))


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








