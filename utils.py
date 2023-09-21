import torch
from om.ont import get_n
from rdflib.term import Literal


def metrics(correct, tries, total):
    precision = 0 if tries == 0 else correct / tries
    recall = 0 if total == 0 else correct / total
    fm = 2 * (precision * recall) / (1 if precision + recall == 0 else precision + recall)
    return precision, recall, fm


def get_name(entity, graph):
    if type(entity) is str:
        entity = Literal(entity)
    name = get_n(entity, graph)

    if name.startswith('//'):
        name = entity.split('http://yago-knowledge.org/resource/')[-1]

    return name


def pad_encode(sentences, word_map):
    """
    Encodes a list of sentences into a padded tensor of integer values using a word mapping.

    Example:
        >>> word_map = {
        ...     'I': 1,
        ...     'love': 2,
        ...     'coding': 3,
        ...     'Python': 4,
        ...     'great': 5,
        ...     'fun': 6,
        ...     'is': 7,
        ... }
        >>> sentences = ["I love coding Python", "Python is great", "Coding is fun"]
        >>> encoded_sentences = pad_encode(sentences, word_map)
        >>> print(encoded_sentences)
        tensor([[1, 2, 3, 4],
                [4, 7, 5, 0],
                [3, 7, 6, 0]])

    :param sentences: A list of input sentences to be encoded into tensors.
    :param word_map: A dictionary mapping words to their corresponding integer representations.
    :return: A tensor containing the padded and encoded sentences, where each sentence is represented
        as a list of integers. The tensor has dimensions (num_sentences, max_sentence_length), where
        num_sentences is the number of input sentences, and max_sentence_length is the length of the longest
        sentence in terms of the number of words.
    """
    sentence_list = []
    max_len = -1
    for sentence in sentences:
        sentence = list(map(lambda word: word_map[word], sentence.split()))
        if len(sentence) > max_len:
            max_len = len(sentence)
        sentence_list.append(sentence)

    padded_sentences = []
    for sentence in sentence_list:
        padded_sentences.append(sentence + [0] * (max_len - len(sentence)))

    return torch.LongTensor(padded_sentences)


def emb_average(sentence_ids, model):
    """
    Calculates the average word embedding for a list of sentences using a given model.

    :param sentence_ids: (list of torch.Tensor): A list of tensors representing sentences with word embeddings.
    :param model: (torch.nn.Module): A neural network model that can compute embeddings for input sentences.
    :return: A tensor representing the average word embedding for each input sentence.
    """
    unsqueezed_sentence = torch.cat(list(map(lambda embedding: embedding.unsqueeze(0), sentence_ids)))
    embedding_sum = model(unsqueezed_sentence).sum(dim=1)
    non_zero_embeddings = torch.sum((unsqueezed_sentence != 0).float(), dim=1).unsqueeze(1)
    non_zero_embeddings[non_zero_embeddings == 0] = 1
    return embedding_sum / non_zero_embeddings


def calc_acc(predicted, correct):
    """
    Calculates the accuracy of a model's predictions.
    :param predicted: A list of predicted labels.
    :param correct:  A list of correct labels.
    :return: The accuracy of the model's predictions.
    """
    acc = (torch.LongTensor(predicted) == correct).float().sum() / correct.shape[0]
    return acc.item()
