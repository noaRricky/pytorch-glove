from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class GloVeModel(nn.Module):
    """Implement GloVe model with Pytorch
    """

    def __init__(self, embedding_size, context_size, vocab_size, min_occurrance=1, x_max=100, alpha=3 / 4):
        super(GloVeModel, self).__init__()

        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        if isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError(
                "'context_size' should be an int or a tuple of two ints")
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.min_occurrance = min_occurrance
        self.x_max = x_max

        self._focal_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)
        self._context_embeddings = nn.Embedding(
            vocab_size, embedding_size).type(torch.float64)
        self._focal_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        self._context_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        self._glove_dataset = None

        for params in self.parameters():
            init.uniform_(params, a=-1, b=1)

    def fit(self, corpus):
        """get dictionary word list and co-occruence matrix from corpus

        Args:
            corpus (list): contain word id list

        Raises:
            ValueError: when count zero cocurrences will raise the problems
        """

        left_size, right_size = self.left_context, self.right_context
        vocab_size, min_occurrance = self.vocab_size, self.min_occurrance

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
        tokens = [word for word, count in
                  word_counts.most_common(vocab_size) if count >= min_occurrance]
        coocurrence_matrix = [(words[0], words[1], count)
                              for words, count in cooccurence_counts.items()
                              if words[0] in tokens and words[1] in tokens]
        self._glove_dataset = GloVeDataSet(coocurrence_matrix)

    def train(self, num_epoch, device, batch_size=512, learning_rate=0.05, loop_interval=10):
        """Training GloVe model

        Args:
            num_epoch (int): number of epoch
            device (str): cpu or gpu
            batch_size (int, optional): Defaults to 512.
            learning_rate (float, optional): Defaults to 0.05. learning rate for Adam optimizer
            batch_interval (int, optional): Defaults to 100. interval time to show average loss

        Raises:
            NotFitToCorpusError: if the model is not fit by corpus, the error will be raise
        """

        if self._glove_dataset is None:
            raise NotFitToCorpusError(
                "Please fit model with corpus before training")

        # basic training setting
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        glove_dataloader = DataLoader(self._glove_dataset, batch_size)
        total_loss = 0

        for epoch in range(num_epoch):
            for idx, batch in enumerate(glove_dataloader):
                optimizer.zero_grad()

                i_s, j_s, counts = batch
                i_s = i_s.to(device)
                j_s = j_s.to(device)
                counts = counts.to(device)
                loss = self._loss(i_s, j_s, counts)

                total_loss += loss.item()
                if idx % loop_interval == 0:
                    avg_loss = total_loss / loop_interval
                    print("epoch: {}, current step: {}, average loss: {}".format(
                        epoch, idx, avg_loss))
                    total_loss = 0

                loss.backward()
                optimizer.step()

        print("finish glove vector training")

    def get_coocurrance_matrix(self):
        """ Return co-occurance matrix for saving

        Returns:
            list: list itam (word_idx1, word_idx2, cooccurances)
        """

        return self._glove_dataset._coocurrence_matrix

    def embedding_for_tensor(self, tokens):
        if not torch.is_tensor(tokens):
            raise ValueError("the tokens must be pytorch tensor object")

        return self._focal_embeddings(tokens) + self._context_embeddings(tokens)

    def _loss(self, focal_input, context_input, coocurrence_count):
        x_max, alpha = self.x_max, self.alpha

        focal_embed = self._focal_embeddings(focal_input)
        context_embed = self._context_embeddings(context_input)
        focal_bias = self._focal_biases(focal_input)
        context_bias = self._context_biases(context_input)

        # count weight factor
        weight_factor = torch.pow(coocurrence_count / x_max, alpha)
        weight_factor[weight_factor > 1] = 1

        embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        log_cooccurrences = torch.log(coocurrence_count)

        distance_expr = (embedding_products + focal_bias +
                         context_bias + log_cooccurrences) ** 2

        single_losses = weight_factor * distance_expr
        mean_loss = torch.mean(single_losses)
        return mean_loss


class GloVeDataSet(Dataset):

    def __init__(self, coocurrence_matrix):
        self._coocurrence_matrix = coocurrence_matrix

    def __getitem__(self, index):
        return self._coocurrence_matrix[index]

    def __len__(self):
        return len(self._coocurrence_matrix)


class NotTrainedError(Exception):
    pass


class NotFitToCorpusError(Exception):
    pass


def _context_windows(region, left_size, right_size):
    """generate left_context, word, right_context tuples for each region

    Args:
        region (str): a sentence
        left_size (int): left windows size
        right_size (int): right windows size
    """

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
        region (str): the sentence for extracting the token base on the context
        start_index (int): index for start step of window
        end_index (int): index for the end step of window
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):
                             min(end_index, last_index) + 1]
    return selected_tokens
