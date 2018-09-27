from collections import defaultdict
import spacy


class SpacyTokenizer:
    """ Tool for tokenize powered by spacy module
    """

    def __init__(self, lang: str, disable=['parser', 'tagger', 'ner']):
        """ Initialize the language type for token

        Args:
            lang (str): language type for tokenizer
        """
        self._nlp = spacy.load(lang, disable=disable)

    def tokenize(self, text: str) -> list:
        # we don't need new line as token
        lines = text.splitlines()

        doc = [[token.text for token
                in self._nlp.tokenizer(text.strip())] for text in lines]

        return doc


class Dictionary:
    """ Tool to build word2idx and doc2idx

    Args:
        doc {list}: list of documents contains words
    """

    def __init__(self, doc=None):

        self.vocab_size = 0
        self.word2idx = defaultdict(int)

        self.update(doc)

    def update(self, doc: list):
        """ Update word2idx information by doc

        Args:
            doc (list): list of words
        """

        if doc is None:
            return

        vocab_size, word2idx = self.vocab_size, self.word2idx

        # count word occurrance and vocab size
        tokens = set()
        for line in doc:
            tokens.update(line)

        for token in tokens:
            if token not in word2idx:
                word2idx[token] = vocab_size
                vocab_size += 1

        self.vocab_size = vocab_size

    def corpus(self, doc: list) -> list:
        """ Convert text of documents to idx of documents

        Args:
            doc (list): text of documents

        Returns:
            list: idx of documents
        """

        word2idx = self.word2idx
        corpus = [[word2idx[word] for word in line if word in word2idx]
                  for line in doc]
        return corpus


if __name__ == '__main__':
    tokenizer = SpacyTokenizer('en_core_web_sm')
    text = "This is an apple. \n This is a tea."
    doc = tokenizer.tokenize(text)
    print(doc)
