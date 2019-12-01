"""
This scipt helps load data from raw file
"""
import tensorflow as tf
from model import nmt_generator_from_data


def load_training_data(filepath, start_word='<s> ', stop_word=' </s>', limit=None):
    # manually pad start and stop word to distinguish between sentence
    lines = []
    with open(filepath) as file_loader:
        for idx, line in enumerate(file_loader):
            if limit and limit<idx:
                break
            lines.append(start_word + line.lower().strip() + stop_word)
    return lines


def load_vocabulary_data(filepath, limit = None):
    voc_dict = {}
    with open(filepath) as file_loader:
        for idx, line in enumerate(file_loader):
            if limit and limit<idx:
                break
            voc_dict[line.replace("\n", "")] = idx + 1
    return voc_dict


def load_dictionary_data(filepath):
    ## TODO: Check if we need this function
    with open(filepath) as file_loader:
        for idx, line in enumerate(file_loader):
            print(line)
    return


def process_sentence(sequence, vocab, pad_length):
    # This function I referred from https://github.com/thisisiron/nmt-attention-tf/blob/master/utils.py
    # TODO : check what's the result of this?
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n',
                                                           oov_token='<unk>')
    lang_tokenizer.word_index = vocab
    tensor = lang_tokenizer.texts_to_sequences(sequence)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=pad_length, padding='post')
    return tensor, lang_tokenizer


def load_data(train_translate_from=r"./data/train.en", train_translate_to=r'./data/train.de',
                       vocab_from=r'./data/vocab.50K.en', vocab_to=r'./data/vocab.50K.de',
                       pad_length=90, limit = 6000):
    # TODO: load test data
    print("Loading training data to memory")
    source_training_data = load_training_data(train_translate_from, limit=limit)
    target_training_data = load_training_data(train_translate_to, limit=limit)
    print("Loading vocabulary data to memory")
    source_vocabulary_data = load_vocabulary_data(vocab_from, limit=limit)
    target_vocabulary_data = load_vocabulary_data(vocab_to, limit=limit)
    # print("The length of source: ",len(source_training_data))
    print("start to process data")
    train_source_tensor, train_source_tokenizer = process_sentence(source_training_data, source_vocabulary_data,
                                                                   pad_length)
    train_target_tensor, train_target_tokenizer = process_sentence(target_training_data, target_vocabulary_data,
                                                                   pad_length)
    train_source_tokenizer.index_word = {}
    train_target_tokenizer.index_word = {}
    convert_vocab(train_source_tokenizer, source_vocabulary_data)
    convert_vocab(train_target_tokenizer, target_vocabulary_data)
    print("Data has been loaded to memory!")
    return train_source_tensor, train_source_tokenizer, train_target_tensor, train_target_tokenizer


def corpus_generator(train_translate_from=r"./data/train.en", train_translate_to=r'./data/train.de',
                       source_vocabulary_data = None, target_vocabulary_data = None, pad_length=90, batch_size = 128,
                     limit = None):
    source_training_data, target_training_data = nmt_generator_from_data(
        train_translate_from, train_translate_to, batch_size, source_vocabulary_data, target_vocabulary_data,
        pad_length)

    train_source_tokenizer.index_word = {}
    train_target_tokenizer.index_word = {}
    convert_vocab(train_source_tokenizer, source_vocabulary_data)
    convert_vocab(train_target_tokenizer, target_vocabulary_data)
    yield train_source_tensor, train_source_tokenizer, train_target_tensor, train_target_tokenizer


def load_test_data(test_translate_from=r"./data/newstest2015.en", test_translate_to=r'./data/newstest2015.de',
                       vocab_from=r'./data/vocab.50K.en', vocab_to=r'./data/vocab.50K.de',
                       pad_length=90, limit = 6000):
    """
    In inference period, assume no vocab data available
    :param test_translate_from:
    :param test_translate_to:
    :param vocab_from:
    :param vocab_to:
    :param pad_length:
    :param limit:
    :return:
    """
    print("Loading test data to memory")
    source_test_data = load_training_data(test_translate_from, limit=limit)
    target_test_data = load_training_data(test_translate_to, limit=limit)
    print("Loading vocabulary data to memory")
    source_vocabulary_data = load_vocabulary_data(vocab_from, limit=limit)
    target_vocabulary_data = load_vocabulary_data(vocab_to, limit=limit)
    print("start to process data")
    test_source_tensor, test_source_tokenizer = process_sentence(source_test_data, source_vocabulary_data,
                                                                   pad_length)
    test_target_tensor, test_target_tokenizer = process_sentence(target_test_data, target_vocabulary_data,
                                                                   pad_length)
    test_source_tokenizer.index_word = {}
    test_target_tokenizer.index_word = {}
    convert_vocab(test_source_tokenizer, source_vocabulary_data)
    convert_vocab(test_target_tokenizer, target_vocabulary_data)
    return test_source_tensor, test_source_tokenizer, test_target_tensor, test_target_tokenizer


def convert_vocab(tokenizer, vocab):
    for key, val in vocab.items():
        assert val!=0
        tokenizer.index_word[val] = key
    return


def print_assumptions():
    assumpt = """
    1. This implementation add start and stop word, as it helps to distinguish the boundary of each setence.\n
    2. No removal for Punctuation ,as tf build in function can achieve this.
    3. Author in paper mention that they filter out sentence length>50,so we set max length as 50.
    
    """
    print(assumpt)
    return


if __name__ == "__main__":
    train_source_tensor, train_source_tokenizer, train_target_tensor, train_target_tokenizer = load_data()

