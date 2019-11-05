import tensorflow as tf
from data_preprocess import load_data
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from model import train_input_fn, encoder, Decoder
import os
import time
import numpy as np
np.random.seed(2019)
tf.random.set_seed(2019)

parser = ArgumentParser(description='training mode')
parser.add_argument('--batch-size', help='batch size <default: 32>', metavar='INT',
                    type=int, default=128)
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

#TODO: figure out a way to place all work on gpunum
#TODO: Fix tensorflow initialization and compare
#TODO: move train wrapper function and check result
#TODO: check variable of var and gradient
def train():
    # args is a global variable in this task
    BATCH_SIZE = args.batch_size
    EPOCH = args.epoch
    EMBED_DIM = args.embeddingDim
    MAXLEN = args.maxLen
    NUM_UNITS = args.units
    LEARNING_RATE = args.learning_rate
    DROPOUT = args.dropout
    METHOD = args.method
    GPUNUM = args.gpuNum
    CKPT = args.checkpoint
    LIMIT = args.limit
    start_word = "<s>"
    end_word = "</s>"
    #Here, tokenizer saves all info to split data.
    #Itself is not a part of data.
    train_source_tensor, train_source_tokenizer, train_target_tensor, train_target_tokenizer = \
        load_data(pad_length = MAXLEN, limit=LIMIT)
    buffer_size = len(train_source_tensor)
    train_source_tensor, val_source_tensor, train_target_tensor, val_target_tensor = \
        train_test_split(train_source_tensor, train_target_tensor, random_state=2019)

    #TODO: check if we need target tokenizer
    training_steps = len(train_source_tensor)//BATCH_SIZE
    vocab_source_size = len(train_source_tokenizer.word_index) + 1
    print("vocab_input_size: ",vocab_source_size)
    vocab_target_size = len(train_target_tokenizer.word_index) + 1
    print("vocab_target_size: ", vocab_target_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # optimizer.learning_rate.assign(0.1)


    # set up checkpoint
    if not os.path.exists(CKPT):
        os.makedirs(CKPT)
    else:
        print("Warning: current Checkpoint dir already exist! ",
              "\nPlease consider to choose a new dir to save your checkpoint!")
    checkpoint = tf.train.Checkpoint(optimzier = optimizer)
    checkpoint_prefix = os.path.join(CKPT, "ckpt")

    dataset = train_input_fn(train_source_tensor, train_target_tensor, buffer_size, EPOCH, BATCH_SIZE)
    apply_loss =  tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    decoder = Decoder(vocab_target_size, EMBED_DIM, NUM_UNITS, batch_size= BATCH_SIZE, method= None, dropout_rate=DROPOUT)


    for epoch in range(EPOCH):
        per_epoch_loss = 0
        start = time.time()
        encoder_hidden = tf.zeros((BATCH_SIZE, NUM_UNITS))
        encoder_ceil = tf.zeros((BATCH_SIZE, NUM_UNITS))
        encoder_state = [[encoder_hidden, encoder_ceil], [encoder_hidden, encoder_ceil],
                         [encoder_hidden, encoder_ceil], [encoder_hidden, encoder_ceil]]
        # TODO : Double check to make sure all re-initialization is performed
        for idx, data in enumerate(dataset.take(training_steps)):
            # merge_to_function
            # print(dir(optimizer))
            # print("check learning rate at here: ",optimizer.learning_rate)
            source, target = data
            with tf.GradientTape() as tape:
                source_out, source_state, source_var = encoder(source, encoder_state, vocab_source_size,
                                                               EMBED_DIM, NUM_UNITS, activation="tanh",
                                                               dropout_rate=DROPOUT)
                initial = tf.expand_dims([train_target_tokenizer.word_index[start_word]] * BATCH_SIZE, 1)
                attention_state = tf.zeros((BATCH_SIZE, 1, EMBED_DIM))
                # cur_total_loss is a sum of loss for current steps, namely batch loss
                cur_total_loss, cur_loss = 0, 0
                for i in range(target.shape[1]):
                    output_state, source_state, attention_state = decoder(initial, source_state, source_out,
                                                                          attention_state)
                    # TODO: check for the case where target is 0
                    cur_loss = apply_loss(target[:, i], output_state)
                    # 0 should be the padding value in target.
                    # I assumed that there should not be 0 value in target
                    # for safety reason, we apply this mask to final loss
                    # Mask is a array contains binary value(0 or 1)
                    mask = tf.math.logical_not(tf.math.equal(target[:, i], 0))
                    mask = tf.cast(mask, dtype=cur_loss.dtype)
                    cur_loss *= mask
                    cur_total_loss += tf.reduce_mean(cur_loss)
                    initial = tf.expand_dims(target[:, i], 1)
                    # print(cur_loss)
                    # print(cur_total_loss)
            batch_loss = cur_total_loss / target.shape[1]
            # print("Turn loss: i,loss: ", i, batch_loss.numpy(), tf.print(batch_loss), type(batch_loss))
            # compute loss
            variables = source_var + decoder.trainable_variables
            # print("check variable: ", len(variables))
            # print("check variable: ", len(variables), variables[12:])

            gradients = tape.gradient(cur_total_loss, variables)
            sum_g = [ele.numpy().sum() for ele in gradients[6:13]]
            print(sum_g)
            # print("check gradient: ", len(gradients))
            # print("check gradient: ", len(gradients), gradients[12:])
            optimizer.apply_gradients(zip(gradients, variables))
            # end of function
            per_epoch_loss += batch_loss
            if idx % 10 == 0:
               print('Epoch {}/{} Batch {}/{} Loss {:.4f}'.format(epoch + 1,
                                                                  EPOCH,
                                                                  idx + 10,
                                                                  training_steps,
                                                                  batch_loss.numpy()))


        print('Epoch {}/{} Total Loss per epoch {:.4f} - {} sec'.format(epoch + 1,
                                                                        EPOCH,
                                                                        per_epoch_loss / training_steps,
                                                                        time.time() - start))
        # TODO: for evaluation add bleu score
        if epoch % 10 == 0:
            print('Saving checkpoint for each 10 epochs')
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == "__main__":
    print("Current version of tensorflow is: "+tf.__version__)
    train()
