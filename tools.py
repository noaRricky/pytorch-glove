from collections import Counter, defaultdict
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

    def __init__(self, min_occurrances=1):

        self.vocab_size = 0
        self.word2idx = defaultdict(int)
        self.min_occurrances = min_occurrances

    def update(self, doc: list):

        vocab_size, word2idx, min_occurrances = self.vocab_size, self.word2idx, self.min_occurrances

        # count word occurrance and vocab size
        words_counts = Counter()
        for tokens in doc:
            words_counts.update(tokens)

        word_len = len(words_counts)
        # filter word occurrances <= min_occurrances and build word2idx
        filter_vocabs = [word for word, count in words_counts.most_common(
            word_len) if count >= min_occurrances]
        for word in filter_vocabs:
            if word not in word2idx:
                word2idx[word] = vocab_size
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
