"""
This scipt helps load data from raw file
"""
import tensorflow as tf


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
    print("start to process data")
    train_source_tensor, train_source_tokenizer = process_sentence(source_training_data, source_vocabulary_data,
                                                                   pad_length)
    train_target_tensor, train_target_tokenizer = process_sentence(target_training_data, target_vocabulary_data,
                                                                   pad_length)
    print("Data has been loaded to memory!")
    return train_source_tensor, train_source_tokenizer, train_target_tensor, train_target_tokenizer


def convert_vocab(tokenizer, vocab):
    for key, val in vocab.items():
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

