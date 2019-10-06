import tensorflow as tf
from data_preprocess import load_training_data
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split


def train():
    parser = ArgumentParser(description='training mode')
    parser.add_argument('--batch-size', help='batch size <default: 32>', metavar='INT',
                        type=int, default=32)
    parser.add_argument('--epoch', help='epoch number <default: 10>', metavar='INT',
                        type=int, default=10)
    parser.add_argument('--embeddingDim', help='embedding dimension <default: 256>',
                        metavar='INT', type=int, default=256)
    parser.add_argument('--maxLen', help='max length of a sentence <default: 90>',
                        metavar='INT', type=int, default=90)
    parser.add_argument('--units', help='units <default: 512>', metavar='INT',
                        type=int, default=512)
    parser.add_argument('--learning_rate', help='learning rate <default: 0.001>',
                        metavar='REAL', type=float, default=0.001)
    parser.add_argument('--dropout', help='dropout probability <default: 0.2>',
                        metavar='REAL', type=float, default=.2)
    parser.add_argument('--method', help='content-based function <default: concat>',
                        metavar='STRING', default='concat')
    parser.add_argument('--gpuNum', help='GPU number to use <default: 0>',
                        metavar='INT', type=int, default=0)

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    EPOCH = args.epoch
    EMBED_DIM = args.embeddingDim
    MAXLEN = args.maxLen
    NUM_UNITS = args.units
    LEARNING_RATE = args.learning_rate
    DROPOUT = args.dropout
    METHOD = args.method
    GPUNUM = args.gpuNum

    train_source_tensor, train_source_tokenizer, train_target_tensor, train_target_tokenizer = load_training_data()
    buffer_size = len(train_source_tensor)
    train_source_tensor, val_source_tensor, train_target_tensor, val_target_tensor = \
        train_test_split(train_source_tensor, train_target_tensor)
    vocab_input_size = len(train_source_tokenizer.word_index) + 1
    vocab_target_size = len(train_target_tokenizer.word_index) + 1





if __name__ == "__main__":
    pass
