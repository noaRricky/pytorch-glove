from collections import Counter
import spacy


class SpacyDictionary:

    def __init__(self, lang: str, min_occurrances=1):

        self.vocab_size = 0
        self.word2idx = {}
        self.min_occurraceces = min_occurrances

        self._nlp = spacy.load(lang)

    def tokenize(self, text: str) -> list:

        min_occurrances = self.min_occurraceces

        # we don't need new line as tokens.
        lines = text.splitlines()

        token_lines = [
            [token.text for token in self._nlp(text.strip())] for text in lines]

        # count word occurrance and vocab size
        words_counts = Counter()
        for tokens in token_lines:
            words_counts.update(tokens)
        self.vocab_size = vocab_size = len(words_counts)

        # filter word occurrances <= min_occurrances and build word2idx
        filter_vocabs = [word for word, count in words_counts.most_common(
            vocab_size) if count >= min_occurrances]
        word2idx = {word: idx for idx, word in enumerate(filter_vocabs)}

        # build corpus
        corpus = [[word2idx[word] for word in tokens if word in word2idx]
                  for tokens in token_lines]
        return corpus


if __name__ == '__main__':
    dictionary = SpacyDictionary('en_core_web_sm')
    string = "When Sebastian Thrun started working on self-driving cars at started \n apple"
    corpus = dictionary.tokenize(string)
    print(corpus)
    print(dictionary.vocab_size)
