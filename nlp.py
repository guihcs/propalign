import nltk


def get_core_concept(e1):
    t1 = nltk.pos_tag(e1)
    v1 = []
    sn = False
    for t in t1:
        if 'V' in t[1] and len(t[0]) > 4:

            v1.append(t[0])
            break

        if 'N' in t[1] or 'J' in t[1] and not sn:
            if 'IN' in t[1]:
                sn = True
            else:
                v1.append(t[0])

    return v1


def get_core_tagged(e1):
    t1 = nltk.pos_tag(e1)
    v1 = []
    sn = False
    for t in t1:
        if 'V' in t[1] and len(t[0]) > 4:

            v1.append(t)
            break

        if 'N' in t[1] or 'J' in t[1] and not sn:
            if 'IN' in t[1]:
                sn = True
            else:
                v1.append(t)

    return v1


def filter_jj(words):
    tags = nltk.pos_tag(words)
    return list(map(lambda x: x[0], filter(lambda x: x[1][0] == 'N', tags)))