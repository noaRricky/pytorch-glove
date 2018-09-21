from collections import defaultdict
import spacy


class SpacyTokenizer:

    def __init__(self, lang: str):
        self._nlp = spacy.load(lang)

    def tokenize(self, text: str):
        # we don't need new line as token
        lines = text.splitlines()

        doc = [
            [token.text for token in self._nlp(text.strip())] for text in lines]

        return doc


class Dictionary:

    def __init__(self, doc=None):

        self.vocab_size = 0
        self.word2idx = defaultdict(int)

        self.update(doc)

    def update(self, doc: list):

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

        word2idx = self.word2idx
        corpus = [[word2idx[word] for word in line if word in word2idx]
                  for line in doc]
        return corpus


if __name__ == '__main__':
    tokenizer = SpacyTokenizer('en_core_web_sm')
    text = "This is an apple. \n This is a tea."
    doc = tokenizer.tokenize(text)
    dictionary = Dictionary()
    dictionary.update(doc)
    corpus = dictionary.corpus(doc)
    print(corpus)
