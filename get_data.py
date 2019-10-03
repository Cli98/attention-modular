import os

import wget


def create_folder():
    """
    create required folder to save data, pretrain weight and output
    :return:
    """
    if not os.path.exists(os.path.join(".", "data")):
        os.makedirs(os.path.join(".", "data"))


def get_train_data(out_folder="./data"):
    # refer to this link for wget tutorial
    # https://pypi.org/project/wget
    # https://stackoverflow.com/questions/24346872/python-equivalent-of-a-given-wget-command
    English_train_url = r"https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en"
    German_train_url = r"https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de"
    wget.download(English_train_url, out=out_folder)
    wget.download(German_train_url, out=out_folder)


def get_vocab_data(out_folder="./data"):
    English_vocab_url = r"https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en"
    German_vocab_url = r"https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de"
    wget.download(English_vocab_url, out=out_folder)
    wget.download(German_vocab_url, out=out_folder)


def get_dictionary_data(out_folder="./data"):
    dictionary_url = r"https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/dict.en-de"
    wget.download(dictionary_url, out=out_folder)


def get_test_data(out_folder="./data"):
    test_link = [
        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en",
        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de",
        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en",
        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de",
        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en",
        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de",
        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en",
        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de"
    ]
    for link in test_link:
        wget.download(link, out=out_folder)


def get_data():
    if not os.path.exists(os.path.join(".", "data")):
        create_folder()
        get_train_data()
        get_vocab_data()
        get_dictionary_data()
        get_test_data()
