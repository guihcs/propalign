import torch
from om.ont import get_n
from rdflib.term import Literal


def metrics(correct, tries, total):
    precision = 0 if tries == 0 else correct / tries
    recall = 0 if total == 0 else correct / total
    fm = 2 * (precision * recall) / (1 if precision + recall == 0 else precision + recall)
    return precision, recall, fm


def gn(e, g):
    if type(e) is str:
        e = Literal(e)
    ns = get_n(e, g)

    if ns.startswith('//'):
        ns = e.split('http://yago-knowledge.org/resource/')[-1]

    return ns



def pad_encode(s, wm):
    l1 = []
    max_len = -1
    for q in s:
        w = list(map(lambda q: wm[q], q.split()))
        if len(w) > max_len:
            max_len = len(w)
        l1.append(w)

    nl1 = []
    for w in l1:
        nl1.append(w + [0] * (max_len - len(w)))

    return torch.LongTensor(nl1)


def emb_average(ids, emb):
    xe = torch.cat(list(map(lambda q: q.unsqueeze(0), ids)))
    xem = emb(xe).sum(dim=1)
    cf = torch.sum((xe != 0).float(), dim=1).unsqueeze(1)
    cf[cf == 0] = 1
    return xem / cf


def calc_acc(pred, cty):
    acc = (torch.LongTensor(pred) == cty).float().sum() / cty.shape[0]
    return acc.item()