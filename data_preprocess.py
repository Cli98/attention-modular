"""
This scipt helps load data from raw file
"""

def load_training_data(filepath, start_word = '<s> ', stop_word = ' </s>'):
    # manually pad start and stop word to distinguish between sentence
    lines = []
    with open(filepath) as file_loader:
        for line in file_loader:
            lines.append(start_word+line.lower().strip()+stop_word)
    return lines


def load_vocabulary_data(filepath):
    voc_dict = {}
    with open(filepath) as file_loader:
        for idx, line in enumerate(file_loader):
            voc_dict[idx+1] = line.replace("\n","")
    return voc_dict


def load_dictionary_data(filepath):
    with open(filepath) as file_loader:
        for idx, line in enumerate(file_loader):
            print(line)
    return


def process_setence(setence_list):
    pass


def load_data(train_translate_from, train_translate_to, vocab_from, vocab_to, test_file_list = []):
    # TODO: load test data
    print("Loading training data to memory")
    source_training_data = load_training_data(train_translate_from)
    target_training_data = load_training_data(train_translate_to)
    print("Loading vocabulary data to memory")
    source_vocabulary_data = load_vocabulary_data(vocab_from)
    target_vocabulary_data = load_vocabulary_data(vocab_to)



def print_assumptions():
    assumpt = """
    1. This implementation add start and stop word, as it helps to distinguish the boundary of each setence.\n
    2. No removal for Punctuation ,as tf build in function can achieve this.
    3. Author in paper mention that they filter out sentence length>50.
    
    """
    print(assumpt)
    return


if __name__=="__main__":
    # create_dataset(path="./data/train.en")
    l = load_dictionary_data(filepath = "./data/dict.en-de")