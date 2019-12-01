import tensorflow as tf
from data_preprocess import load_data, load_vocabulary_data, nmt_generator_from_data, process_sentence
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from model import train_input_fn, Encoder, Decoder, generator_input_fn
import os
import time
import numpy as np
np.random.seed(2019)
tf.random.set_seed(2019)
# TODO: Roll back code base and change loss, 1. change to reduce_mean, 2. use cur_loss/ cur_total_loss

parser = ArgumentParser(description='training mode')
parser.add_argument('--batch-size', help='batch size <default: 32>', metavar='INT',
                    type=int, default=128)#128
parser.add_argument('--epoch', help='epoch number <default: 10>', metavar='INT',
                    type=int, default=12)
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
                    metavar='INT', type=int, default=1)
parser.add_argument('--checkpoint', help='The dir to save checkpoint',
                    metavar='STRING', type=str, default="./checkpoint")
parser.add_argument('--limit', help='only use part of the data to train model',
                    metavar='INT', type=int, default=300)#None

args = parser.parse_args()

#TODO: figure out a way to place all work on gpunum
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
    source_vocabulary_data = load_vocabulary_data(filepath=r'./data/vocab.50K.en', limit=None)
    target_vocabulary_data = load_vocabulary_data(filepath=r'./data/vocab.50K.de', limit=None)
    buffer_size = BATCH_SIZE*100

    #train_source_tensor, train_source_tokenizer, train_target_tensor, train_target_tokenizer = \
    #    load_data(pad_length = MAXLEN, limit=LIMIT)

    #train_source_tensor, val_source_tensor, train_target_tensor, val_target_tensor = \
    #    train_test_split(train_source_tensor, train_target_tensor, random_state=2019)

    vocab_source_size = len(source_vocabulary_data)
    print("vocab_input_size: ",vocab_source_size)
    vocab_target_size = len(target_vocabulary_data)
    print("vocab_target_size: ", vocab_target_size)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    # dataset = generator_input_fn(train_source_tensor, train_target_tensor, buffer_size, EPOCH, BATCH_SIZE, MAXLEN)
    apply_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    encoder = Encoder(vocab_source_size, EMBED_DIM, NUM_UNITS, dropout_rate = DROPOUT, batch_size=BATCH_SIZE)
    decoder = Decoder(vocab_target_size, EMBED_DIM, NUM_UNITS, batch_size= BATCH_SIZE, method= None, dropout_rate=DROPOUT)
    # set up checkpoint
    if not os.path.exists(CKPT):
        os.makedirs(CKPT)
    else:
        print("Warning: current Checkpoint dir already exist! ",
              "\nPlease consider to choose a new dir to save your checkpoint!")
    checkpoint = tf.train.Checkpoint(optimizer = optimizer, encoder=encoder,
                                     decoder=decoder)
    checkpoint_prefix = os.path.join(CKPT, "ckpt")


    def train_wrapper(source, target, train_target_tokenizer):
        # with tf.GradientTape(watch_accessed_variables=False) as tape:
        with tf.GradientTape() as tape:
            # source_out, source_state, source_trainable_var, tape = encoder(source, encoder_state, vocab_source_size,
            #                                                          EMBED_DIM, NUM_UNITS, activation="tanh",
            #                                                          dropout_rate = DROPOUT)
            source_out, source_state = encoder(source, encoder_state, activation="tanh")

            initial = tf.expand_dims([train_target_tokenizer.word_index[start_word]] * BATCH_SIZE, 1)
            attention_state = tf.zeros((BATCH_SIZE, 1, EMBED_DIM))
            # cur_total_loss is a sum of loss for current steps, namely batch loss
            cur_total_loss, cur_total_plex, cur_loss = 0, 0, 0
            for i in range(1, target.shape[1]):
                output_state, source_state, attention_state = decoder(initial, source_state, source_out, attention_state)
                # TODO: check for the case where target is 0
                cur_loss = apply_loss(target[:, i], output_state)
                perplex = tf.nn.sparse_softmax_cross_entropy_with_logits(target[:, i], output_state)
                # 0 should be the padding value in target.
                # I assumed that there should not be 0 value in target
                # for safety reason, we apply this mask to final loss
                # Mask is a array contains binary value(0 or 1)
                mask = tf.math.logical_not(tf.math.equal(target[:, i], 0))
                mask = tf.cast(mask, dtype=cur_loss.dtype)
                cur_loss *= mask
                perplex *= mask
                cur_total_loss += tf.reduce_mean(cur_loss)
                cur_total_plex += tf.reduce_mean(perplex)
                initial = tf.expand_dims(target[:, i], 1)
                # print(cur_loss)
                # print(cur_total_loss)
        batch_loss = cur_total_loss / target.shape[1]
        batch_perplex = cur_total_plex / target.shape[1]
        ## debug
        variables = encoder.trainable_variables + decoder.trainable_variables
        # print("check variable: ", len(variables))
        #variables = encoder.trainable_variables
        # print("check var:", len(variables), variables[12:])
        gradients = tape.gradient(cur_total_loss, variables)
        # print("check gradient: ", len(gradients))
        # g_e = [type(ele) for ele in gradients if not isinstance(ele, tf.IndexedSlices)]
        # sum_g = [ele.numpy().sum() for ele in gradients if not isinstance(ele, tf.IndexedSlices)]

        # print(len(gradients), len(sum_g))
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        optimizer.apply_gradients(zip(clipped_gradients, variables))
        return batch_loss, batch_perplex
    # print(len(train_source_tensor),BATCH_SIZE,training_steps,LIMIT)
    for epoch in range(EPOCH):
        per_epoch_loss, per_epoch_plex = 0, 0
        start = time.time()
        encoder_hidden = encoder.initialize_hidden_state()
        encoder_ceil = encoder.initialize_cell_state()
        encoder_state = [[encoder_hidden, encoder_ceil], [encoder_hidden, encoder_ceil],
                         [encoder_hidden, encoder_ceil], [encoder_hidden, encoder_ceil]]
        # TODO : Double check to make sure all re-initialization is performed
        gen = \
            nmt_generator_from_data(source_data_path=r"./data/train.en", target_data_path=r'./data/train.de',
                         source_vocabulary_data=source_vocabulary_data, target_vocabulary_data=target_vocabulary_data, pad_length=90, batch_size=BATCH_SIZE,
                         limit=args.limit)
        for idx, data in enumerate(gen):
            train_source_tensor, train_source_tokenizer, train_target_tensor, train_target_tokenizer = data
            source, target = tf.convert_to_tensor(train_source_tensor), tf.convert_to_tensor(train_target_tensor)
            #print(type(source))
            #print("The shape of source is: ",source.shape,train_source_tensor.shape)
            #print(source[0])
            #print(source[1])
            cur_total_loss, batch_perplex = train_wrapper(source, target, train_target_tokenizer)
            per_epoch_loss += cur_total_loss
            per_epoch_plex += tf.exp(batch_perplex).numpy()
            if idx % 10 == 0:
                # print("current step is: "+str(tf.compat.v1.train.get_global_step()))
                # print(dir(optimizer))
                with open("checkpoint/logger.txt","a") as filelogger:
                    print("current learning rate is:"+str(optimizer._learning_rate),file=filelogger)
                    print('Epoch {}/{} Batch {} Loss {:.4f} perplex {:.4f}'.format(epoch + 1,
                                                                       EPOCH,
                                                                       idx + 10,
                                                                       cur_total_loss.numpy(),
                                                                        tf.exp(batch_perplex).numpy()),
                          file=filelogger)
                # tf.print(step)
                # print(dir(step))
                # print(int(step))

        with open("checkpoint/logger.txt", "a") as filelogger:
            print('Epoch {}/{} Total Loss per epoch {:.4f} - {} sec'.format(epoch + 1,
                                                                        EPOCH,
                                                                        per_epoch_loss / (idx+1.0),
                                                                        time.time() - start),file=filelogger)
            print("Epoch perplex: ",str(per_epoch_plex),file=filelogger)
        # TODO: for evaluation add bleu score
        if epoch % 1 == 0:
            print('Saving checkpoint for each epochs')
            checkpoint.save(file_prefix=checkpoint_prefix)

        if epoch == 3:
            optimizer._learning_rate = LEARNING_RATE / 10.0


if __name__ == "__main__":
    print("Current version of tensorflow is: "+tf.__version__)
    print("check if gpu is available: "+str(tf.test.is_gpu_available()))
   # gpus = tf.config.experimental.list_physical_devices('GPU')
   #  with tf.device('/device:GPU:0'):
   #  	print(tf.test.gpu_device_name())
    train()

