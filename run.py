import zipfile
import logging
import pickle
import torch
from glove import GloVeModel
from tools import SpacyTokenizer, Dictionary

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

FILE_PATH = './data/text8.zip'
MODLE_PATH = './model/glove.pt'
DOC_PATH = './data/corpus.pickle'
COMATRIX_PATH = './data/comat.pickle'
LANG = 'en_core_web_sm'
EMBEDDING_SIZE = 128
CONTEXT_SIZE = 3
NUM_EPOCH = 100
BATHC_SIZE = 512
LEARNING_RATE = 0.01


def read_data(file_path, type='file'):
    """ Read data into a string

    Args:
        file_path (str): path for the data file
    """
    text = None
    if type is 'file':
        with open(file_path, mode='r', encoding='utf-8') as fp:
            text = fp.read()
    elif type is 'zip':
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
    # text = read_data(FILE_PATH, type='zip')
    # logging.info("read raw data")

    # init base model
    # tokenizer = SpacyTokenizer(LANG)
    dictionary = Dictionary()

    # build corpus
    # doc = tokenizer.tokenize(text)
    # logging.info("after generate tokens from text")

    # save doc
    # with open(DOC_PATH, mode='wb') as fp:
    #     pickle.dump(doc, fp)
    # logging.info("tokenized documents saved!")
    # load doc
    with open(DOC_PATH, 'rb') as fp:
        doc = pickle.load(fp)

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

    # fit corpus to count cooccurance matrix
    model.fit(corpus)

    cooccurance_matrix = model.get_coocurrance_matrix()
    # saving cooccurance_matrix
    with open(COMATRIX_PATH, mode='wb') as fp:
        pickle.dump(cooccurance_matrix, fp)

    model.train(NUM_EPOCH, device, learning_rate=LEARNING_RATE)

    # save model for evaluation
    torch.save(model.state_dict(), MODLE_PATH)


if __name__ == '__main__':
    train_glove_model()
