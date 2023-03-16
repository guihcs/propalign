import torch
import struct
import os


def _read_key(f):
    key = [f.read(1)]

    while key[-1] != b' ':
        key.append(f.read(1))

    data = b''.join(key[:-1])
    return data.decode('utf-8')


def _to_vector(d):
    v = []

    for i in range(0, 800, 4):
        v.append(struct.unpack('<f', d[i:i + 4]))

    return torch.Tensor(v).squeeze(1)


class Finbank:

    def __init__(self, ep):
        super(Finbank, self).__init__()
        self.ep = ep

        if not os.path.exists('./fin_cache'):
            print('Embeddings cache not found. Building...')
            os.mkdir('./fin_cache')

            with open('./fin_cache/fbk.txt', 'w') as fbk:
                with open('./fin_cache/fbe.bin', 'wb') as fbe:

                    with open(self.ep, 'rb') as f:

                        line_count, emb_size = f.readline().decode('utf-8').split()
                        line_count, emb_size = int(line_count), int(emb_size)

                        for _ in range(line_count):

                            try:
                                key = _read_key(f)
                                emb = f.read(801)
                                fbk.write(key + '\n')
                                fbe.write(emb)

                            except UnicodeDecodeError:
                                f.read(801)

        with open('./fin_cache/fbk.txt', 'r') as f:
            names = f.readlines()
            self.wm = {w[:-1]: i for i, w in enumerate(names)}

        self.embs = dict()

    def sim(self, w1, w2):
        return torch.cosine_similarity(self.get_emb(w1).unsqueeze(0), self.get_emb(w2).unsqueeze(0)).item()

    def get_emb(self, w1):
        if w1 not in self.wm:
            return torch.zeros(200)

        if self.wm[w1] not in self.embs:
            with open('./fin_cache/fbe.bin', 'rb') as f:
                f.seek(self.wm[w1] * 801)
                data = f.read(801)
                self.embs[self.wm[w1]] = _to_vector(data)
        return self.embs[self.wm[w1]]


