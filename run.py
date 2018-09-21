import zipfile
import torch
from glove import GloVeModel
from tools import SpacyTokenizer, Dictionary

FILE_PATH = './data/text8.zip'
MODLE_PATH = './model/glove.pt'
LANG = 'en_core_web_sm'
EMBEDDING_SIZE = 128
CONTEXT_SIZE = 3
NUM_EPOCH = 50


def read_data(file_path):
    """ Read data into a string

    Args:
        file_path (str): path for the data file
    """
    with zipfile.ZipFile(file_path) as fp:
        text = fp.read(fp.namelist()[0])
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

    # init base model
    tokenizer = SpacyTokenizer(LANG)
    dictionary = Dictionary()

    # build corpus
    doc = tokenizer.tokenize(text)
    dictionary.update(doc)
    corpus = dictionary.corpus(doc)
    vocab_size = dictionary.vocab_size

    return corpus, vocab_size


def train_glove_model():
    # preprocess
    corpus, vocab_size = preprocess(FILE_PATH)

    # specify device type
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init vector model
    model = GloVeModel(EMBEDDING_SIZE, CONTEXT_SIZE, vocab_size)
    model.to(device)
    model.fit(corpus)
    model.train(NUM_EPOCH)

    # save model for evaluation
    torch.save(model.state_dict(), MODLE_PATH)


if __name__ == '__main__':
    train_glove_model()