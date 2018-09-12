from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.init as init


class GloVe(nn.Module):

    def __init__(self, embedding_size, context_size, vocab_size=100000, min_occurrances=1, x_max=100, alpha=3 / 4):
        super(GloVe, self).__init__()

        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        if isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError(
                "'context_size' should be an int or a tuple of two ints")
        self.vocab_size = vocab_size
        self.min_occurrances = min_occurrances
        self.alpha = alpha
        self.x_max = x_max

        self.__focal_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.__context_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.__focal_bias = nn.Parameter(torch.Tensor(vocab_size))
        self.__context_bias = nn.Parameter(torch.Tensor(vocab_size))
        self.__coocurrence_matrix = None

        for params in self.parameters():
            init.uniform_(params, a=-1, b=1)

    def fit(self, corpus):
        left_size, right_size = self.left_context, self.right_context
        vocab_size, min_occurrances = self.vocab_size, self.min_occurrances

        # get co-occurence count matrix
        word_counts = Counter()
        cooccurence_counts = defaultdict(float)
        for region in corpus:
            word_counts.update(region)
            for left_context, word, right_context in _context_windows(region, left_size, right_size):
                for i, context_word in enumerate(left_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(right_context):
                    cooccurence_counts[(word, context_word)] += 1 / (i + 1)
        if len(cooccurence_counts) == 0:
            raise ValueError(
                "No coccurrences in corpus, Did you try to reuse a generator?")

        # get words bag information
        self.__words = [word for word, count in word_counts.most_common(vocab_size)
                        if count >= min_occurrances]
        self.__word_to_id = {word: i for i, word in enumerate(self.__words)}
        self.__coocurrence_matrix = {
            (self.__word_to_id[words[0]], self.__word_to_id[words]): count
            for words, count in cooccurence_counts.items()
            if words[0] in self.__word_to_id and words[1] in self.__word_to_id
        }


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    """Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.

    Args:
        region (str): the sentence for extracting the token base on the contet
        start_index (int): index for start step of window
        end_index (int): index for the end step of window
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0): min(end_index, last_index) + 1]
    return selected_tokens


if __name__ == '__main__':
    embedding_size = 100
    context_size = 4
    glove = GloVe(embedding_size, context_size)
