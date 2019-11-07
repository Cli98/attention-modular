import tensorflow as tf
from data_preprocess import load_test_data
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from model import train_input_fn, Encoder, Decoder
import os
import time
import numpy as np
np.random.seed(2019)
tf.random.set_seed(2019)

parser = ArgumentParser(description='test mode')
parser.add_argument('--batch-size', help='batch size <default: 32>', metavar='INT',
                    type=int, default=1)
parser.add_argument('--epoch', help='epoch number <default: 10>', metavar='INT',
                    type=int, default=10)
parser.add_argument('--embeddingDim', help='embedding dimension',
                    metavar='INT', type=int, default=1000)
parser.add_argument('--maxLen', help='max length of a sentence',
                    metavar='INT', type=int, default=90)
parser.add_argument('--units', help='units', metavar='INT',
                    type=int, default=1000)
parser.add_argument('--learning_rate', help='learning rate',
                    metavar='REAL', type=float, default=0.001)
parser.add_argument('--dropout', help='dropout probability',
                    metavar='REAL', type=float, default=.2)
parser.add_argument('--method', help='content-based function',
                    metavar='STRING', default='concat')
parser.add_argument('--gpuNum', help='GPU number to use',
                    metavar='INT', type=int, default=3)
parser.add_argument('--checkpoint', help='The dir to save checkpoint',
                    metavar='STRING', type=str, default="./checkpoint")
parser.add_argument('--limit', help='only use part of the data to train model',
                    metavar='INT', type=int, default=300)

args = parser.parse_args()

def test():
    #Dropout rate is 0 in test setup?
    BATCH_SIZE = args.batch_size
    EMBED_DIM = args.embeddingDim
    MAXLEN = args.maxLen
    NUM_UNITS = args.units
    CKPT = args.checkpoint
    LEARNING_RATE = args.learning_rate
    EPOCH = 1
    DROPOUT = 0
    start_word = "<s>"
    end_word = "</s>"

    test_source_tensor, test_source_tokenizer, test_target_tensor, test_target_tokenizer = \
        load_test_data(test_translate_from=r"./data/newstest2015.en", test_translate_to=r'./data/newstest2015.de',
           vocab_from=r'./data/vocab.50K.en', vocab_to=r'./data/vocab.50K.de',
           pad_length=90, limit=6000)

    vocab_source_size = len(test_source_tokenizer.word_index) + 1
    print("vocab_input_size: ",vocab_source_size)
    vocab_target_size = len(test_target_tokenizer.word_index) + 1
    print("vocab_target_size: ", vocab_target_size)
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    buffer_size = len(test_source_tensor)

    test_steps = len(test_source_tensor) // BATCH_SIZE

    ckpt = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)


    dataset = train_input_fn(test_source_tensor, test_target_tensor, buffer_size, EPOCH, BATCH_SIZE)
    apply_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    encoder = Encoder(vocab_source_size, EMBED_DIM, NUM_UNITS, dropout_rate = DROPOUT, batch_size=BATCH_SIZE)
    decoder = Decoder(vocab_target_size, EMBED_DIM, NUM_UNITS, batch_size= BATCH_SIZE, method= None, dropout_rate=DROPOUT)


    def test_wrapper(source, target):
        # source_out, source_state, source_trainable_var, tape = encoder(source, encoder_state, vocab_source_size,
        #                                                          EMBED_DIM, NUM_UNITS, activation="tanh",
        #                                                          dropout_rate = DROPOUT)
        result = ""
        source_out, source_state = encoder(source, encoder_state, activation="tanh")

        initial = tf.expand_dims([test_target_tokenizer.word_index[start_word]] * BATCH_SIZE, 1)
        attention_state = tf.zeros((BATCH_SIZE, 1, EMBED_DIM))
        # cur_total_loss is a sum of loss for current steps, namely batch loss
        cur_total_loss, cur_loss = 0, 0
        for i in range(target.shape[1]):
            output_state, source_state, attention_state = decoder(initial, source_state, source_out,
                                                                  attention_state)
            # TODO: check for the case where target is 0
            # 0 should be the padding value in target.
            # I assumed that there should not be 0 value in target
            # for safety reason, we apply this mask to final loss
            # Mask is a array contains binary value(0 or 1)
            current_ind = tf.argmax(output_state[0])
            result += test_target_tokenizer[current_ind]
            if test_target_tokenizer[current_ind] == end_word:
                break
            initial = tf.expand_dims(target[:, i], 1)

        return result

    encoder_hidden = encoder.initialize_hidden_state()
    encoder_ceil = encoder.initialize_cell_state()
    encoder_state = [[encoder_hidden, encoder_ceil], [encoder_hidden, encoder_ceil],
                     [encoder_hidden, encoder_ceil], [encoder_hidden, encoder_ceil]]
    # TODO : Double check to make sure all re-initialization is performed
    result_by_batch = []
    for idx, data in enumerate(dataset.take(test_steps)):

        source, target = data
        result_by_batch.append(test_wrapper(source, target))
    return result_by_batch

if __name__ == "__main__":
    test()

