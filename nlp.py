import nltk

nltk.download('averaged_perceptron_tagger')


def get_core_concept(entity):
    """
    Get the core concept of an entity. The core concept is the first verb with length > 4 or the first noun with its
    adjectives.
    :param entity: RDFLib entity
    :return: list of words
    """
    tags = nltk.pos_tag(entity)
    core_concept = []
    no_name = False
    for (word, tag) in tags:
        if 'V' in tag and len(word) > 4:
            core_concept.append(word)
            break

        if 'N' in tag or 'J' in tag and not no_name:
            if 'IN' in tag:
                no_name = True
            else:
                core_concept.append(word)

    return core_concept


def filter_adjectives(words):
    """
    Filter adjectives from a list of words.
    :param words: list of words
    :return: list of words without adjectives
    """
    tags = nltk.pos_tag(words)
    return list(map(lambda word: word[0], filter(lambda word: word[1][0] == 'N', tags)))
