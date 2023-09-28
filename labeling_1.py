#source: https://towardsdatascience.com/clustering-sentence-embeddings-to-identify-intents-in-short-text-48d22d3bf02e
from imports import *

def most_common(lst, n=None):
    """
    Given a list, return the n most common elements as a list of tuples,
    where each tuple is (element, count).
    If n is None, return all elements.
    """
    return Counter(lst).most_common(n)


nlp = spacy.load('en_core_web_trf')


def extract_labels(category_docs):
    """
    Extract labels from documents in the same cluster by concatenating
    most common verbs, ojects, and nouns
    """

    verbs = []
    dobjs = []
    nouns = []
    adjs = []

    verb = ''
    dobj = ''
    noun1 = ''
    noun2 = ''

    # for each document, append verbs, dobs, nouns, and adjectives to
    # running lists for whole cluster
    for i in range(len(category_docs)):
        doc = nlp(category_docs[i])
        for token in doc:
            if token.is_stop == False:
                if token.dep_ == 'ROOT':
                    verbs.append(token.text.lower())

                elif token.dep_ == 'dobj':
                    dobjs.append(token.lemma_.lower())

                elif token.pos_ == 'NOUN':
                    nouns.append(token.lemma_.lower())

                elif token.pos_ == 'ADJ':
                    adjs.append(token.lemma_.lower())

    # take most common words of each form
    if len(verbs) > 0:
        verb = most_common(verbs, 1)[0][0]

    if len(dobjs) > 0:
        dobj = most_common(dobjs, 1)[0][0]

    if len(nouns) > 0:
        noun1 = most_common(nouns, 1)[0][0]

    if len(set(nouns)) > 1:
        noun2 = most_common(nouns, 2)[1][0]

    # concatenate the most common verb-dobj-noun1-noun2 (if they exist)
    label_words = [verb, dobj]

    for word in [noun1, noun2]:
        if word not in label_words:
            label_words.append(word)

    if '' in label_words:
        label_words.remove('')

    label = '_'.join(label_words)

    return label