import tensorflow as tf
from data_preprocess import load_test_data, load_training_data, load_data
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from model import train_input_fn, Encoder, Decoder
import os
import time
import numpy as np
from tqdm import tqdm
import nltk
np.random.seed(2019)
tf.random.set_seed(2019)

parser = ArgumentParser(description='test mode')
parser.add_argument('--batch-size', help='batch size <default: 32>', metavar='INT',
                    type=int, default=1)
parser.add_argument('--epoch', help='epoch number <default: 10>', metavar='INT',
                    type=int, default=1)
parser.add_argument('--embeddingDim', help='embedding dimension',
                    metavar='INT', type=int, default=500)
parser.add_argument('--maxLen', help='max length of a sentence',
                    metavar='INT', type=int, default=90)
parser.add_argument('--units', help='units', metavar='INT',
                    type=int, default=500)
parser.add_argument('--learning_rate', help='learning rate',
                    metavar='REAL', type=float, default=0.1)
parser.add_argument('--dropout', help='dropout probability',
                    metavar='REAL', type=float, default=.2)
parser.add_argument('--method', help='content-based function',
                    metavar='STRING', default='concat')
parser.add_argument('--gpuNum', help='GPU number to use',
                    metavar='INT', type=int, default=3)
parser.add_argument('--checkpoint', help='The dir to save checkpoint',
                    metavar='STRING', type=str, default="./checkpoint")
parser.add_argument('--limit', help='only use part of the data to train model',
                    metavar='INT', type=int, default=6000)

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
           pad_length=90, limit=args.limit)
    #test_source_tensor, test_source_tokenizer, test_target_tensor, test_target_tokenizer = \
    #    load_data(pad_length = MAXLEN, limit=None)
    print(len(test_source_tensor))
    vocab_source_size = len(test_source_tokenizer.word_index) + 1
    print("vocab_input_size: ",vocab_source_size)
    vocab_target_size = len(test_target_tokenizer.word_index) + 1
    print("vocab_target_size: ", vocab_target_size)
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    buffer_size = len(test_source_tensor)

    test_steps = len(test_source_tensor) // BATCH_SIZE
    #print(test_source_tensor[0])
    #print("Type: ",type(test_source_tensor))
    #for ele in test_target_tensor:
    #    print(type(ele))
    #print("dir of test_target__token: ",dir(test_target_tokenizer))
    #print(test_target_tokenizer.index_word)
    dataset = train_input_fn(test_source_tensor, test_target_tensor, buffer_size, EPOCH, BATCH_SIZE)

    # apply_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    encoder = Encoder(vocab_source_size, EMBED_DIM, NUM_UNITS, dropout_rate = DROPOUT, batch_size=BATCH_SIZE)
    decoder = Decoder(vocab_target_size, EMBED_DIM, NUM_UNITS, batch_size= BATCH_SIZE, method= None, dropout_rate=DROPOUT)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=10)
    ckpt.restore(manager.latest_checkpoint)
    per_epoch_loss, per_epoch_plex = 0, 0
    def test_wrapper(source, target):
        # source_out, source_state, source_trainable_var, tape = encoder(source, encoder_state, vocab_source_size,
        #                                                          EMBED_DIM, NUM_UNITS, activation="tanh",
        #                                                          dropout_rate = DROPOUT)
        result = ""
        source_out, source_state = encoder(source, encoder_state, activation="tanh")

        initial = tf.expand_dims([test_target_tokenizer.word_index[start_word]] * BATCH_SIZE, 1)
        attention_state = tf.zeros((BATCH_SIZE, 1, EMBED_DIM))
        apply_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        # cur_total_loss is a sum of loss for current steps, namely batch loss
        cur_total_loss, cur_total_plex, cur_loss = 0, 0, 0
        #print("word_index: ", test_target_tokenizer.word_index,len(test_target_tokenizer.word_index))
        #print("index_word: ",test_target_tokenizer.index_word)
        for i in range(target.shape[1]):
            output_state, source_state, attention_state = decoder(initial, source_state, source_out,
                                                                  attention_state)
            # TODO: check for the case where target is 0
            # 0 should be the padding value in target.
            # I assumed that there should not be 0 value in target
            # for safety reason, we apply this mask to final loss
            # Mask is a array contains binary value(0 or 1)
            #print(output_state.numpy().shape)
            #print(output_state[0].numpy())
            cur_loss = apply_loss(target[:, i], output_state)
            perplex = tf.nn.sparse_softmax_cross_entropy_with_logits(target[:, i], output_state)
            current_ind = tf.argmax(output_state[0])
            mask = tf.math.logical_not(tf.math.equal(target[:, i], 0))
            mask = tf.cast(mask, dtype=cur_loss.dtype)
            cur_loss *= mask
            perplex *= mask
            cur_total_loss += tf.reduce_mean(cur_loss)
            cur_total_plex += tf.reduce_mean(perplex)
            #tf.print("check current id: ",current_ind)
            #tf.print(test_target_tokenizer.index_word[29])
            #print(current_ind.numpy())
            if current_ind.numpy()==0:
                # 0 is for pad value, we don't need to record it
                continue
            result += test_target_tokenizer.index_word[current_ind.numpy()]
            if test_target_tokenizer.index_word[current_ind.numpy()] == end_word:
                break
            initial = tf.expand_dims(target[:, i], 1)
        batch_loss = cur_total_loss / target.shape[1]
        batch_perplex = cur_total_plex / target.shape[1]
        return batch_loss, batch_perplex, result

    encoder_hidden = encoder.initialize_hidden_state()
    encoder_ceil = encoder.initialize_cell_state()
    encoder_state = [[encoder_hidden, encoder_ceil], [encoder_hidden, encoder_ceil],
                     [encoder_hidden, encoder_ceil], [encoder_hidden, encoder_ceil]]
    # TODO : Double check to make sure all re-initialization is performed
    result_by_batch = []
    for idx, data in tqdm(enumerate(dataset.take(test_steps)),total=args.limit):
        source, target = data
        batch_loss, batch_perplex, result = test_wrapper(source, target)
        with open("checkpoint/test_logger.txt", "a") as filelogger:
            print("The validation loss in batch "+str(idx)+" is : ", str(batch_loss.numpy() / (idx + 1.0)),file=filelogger)
            print("The validation perplex in batch " + str(idx) + " is : ", str(batch_perplex.numpy() / (idx + 1.0)),
                  file=filelogger)
        per_epoch_loss += batch_loss
        per_epoch_plex += batch_perplex
        assert type(result)==str
        result_by_batch.append(result)
        #if idx>=3:
        #    break
    with open("checkpoint/test_logger.txt", "a") as filelogger:
        print("The validation loss is : ",str(per_epoch_loss.numpy()/(idx+1.0)),file=filelogger)
        print("The validation perplex is: ",str(tf.exp(per_epoch_plex).numpy()/(idx+1.0)),file=filelogger)
    return test_target_tokenizer, result_by_batch

if __name__ == "__main__":
    token, result_by_batch = test()
    target_test_data = load_training_data(r'./data/newstest2015.de', limit=args.limit)
    BLEU = 0
    #print(result_by_batch)
    for pred, true in zip(result_by_batch,target_test_data[:4]):
        #print(true)
        #print(pred)
        BLEU+=nltk.translate.bleu_score.sentence_bleu(true,pred)
    with open("checkpoint/test_logger.txt", "a") as filelogger:
        print("The average BLEU-4 score is: ",str(BLEU/args.limit*100),file=filelogger)

