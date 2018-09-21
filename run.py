import zipfile
import logging
import torch
from glove import GloVeModel
from tools import SpacyTokenizer, Dictionary

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

FILE_PATH = './data/short_story.txt'
MODLE_PATH = './model/glove.pt'
LANG = 'en_core_web_sm'
EMBEDDING_SIZE = 128
CONTEXT_SIZE = 3
NUM_EPOCH = 50
LEARNING_RATE = 0.00001


def read_data(file_path, type='file'):
    """ Read data into a string

    Args:
        file_path (str): path for the data file
    """
    text = None
    if type is 'file':
        with open(file_path, mode='r', encoding='utf-8') as fp:
            text = fp.read()
    else:
        with zipfile.ZipFile(file_path) as fp:
            text = fp.read(fp.namelist()[0]).decode()
    return text


def preprocess(file_path):
    """ Get corpus and vocab_size from raw text

    Args:
        file_path (str): raw file path

    Returns:
        corpus (list): list of idx words
        vocab_size (int): vocabulary size
    """

    # preprocess read raw text
    text = read_data(FILE_PATH)
    logging.info("read raw data")

    # init base model
    tokenizer = SpacyTokenizer(LANG)
    dictionary = Dictionary()

    # build corpus
    doc = tokenizer.tokenize(text)
    logging.info("after generate tokens from text")
    dictionary.update(doc)
    logging.info("after generate dictionary")
    corpus = dictionary.corpus(doc)
    vocab_size = dictionary.vocab_size

    return corpus, vocab_size


def train_glove_model():
    # preprocess
    corpus, vocab_size = preprocess(FILE_PATH)

    # specify device type
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init vector model
    logging.info("init model hyperparameter")
    model = GloVeModel(EMBEDDING_SIZE, CONTEXT_SIZE, vocab_size)
    model.to(device)
    model.fit(corpus)
    model.train(NUM_EPOCH, learning_rate=LEARNING_RATE)

    # save model for evaluation
    torch.save(model.state_dict(), MODLE_PATH)


if __name__ == '__main__':
    train_glove_model()
