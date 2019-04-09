import sys
from senseclust.consts import CLUS_LANG
from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.similarities import SoftCosineSimilarity
from finntk.emb.fasttext import vecs
from sqlalchemy.sql import select
from senseclust.queries import joined
from wikiparse.tables import headword, word_sense
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import numpy as np
from senseclust.utils import graph_clust, group_clust


def softcos(defns):
    dictionary = Dictionary(defns.values())
    bow_corpus = [dictionary.doc2bow(document) for document in defns.values()]

    similarity_matrix = vecs.get_en().similarity_matrix(dictionary)
    index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)
    affinities = np.zeros((len(defns), len(defns)))

    for row, similarities in enumerate(index):
        for col, similarity in similarities:
            affinities[col, row] = similarity
    clust_labels = graph_clust(affinities)
    return group_clust(list(defns.keys()), clust_labels)


def gloss_graph(lemma_name, include_wiktionary=False, session=None):
    defns = {}
    # Add wiktionary senses
    wiktionary_senses = session.execute(select([
        word_sense.c.sense_id,
        word_sense.c.etymology_index,
        word_sense.c.sense,
        word_sense.c.extra,
    ]).select_from(joined).where(
        (headword.c.name == lemma_name) &
        (word_sense.c.pos == "Noun")
    )).fetchall()
    for row in wiktionary_senses:
        tokens = word_tokenize(row["sense"])
        if not tokens:
            sys.stderr.write(f"Empty defn: {row['sense_id']} '{row['sense']}'\n")
            continue
        defns[row["sense_id"]] = tokens
    # Add WordNet senses
    wordnet_senses = wordnet.lemmas(lemma_name, lang=CLUS_LANG)
    for lemma in wordnet_senses:
        defns[wordnet.ss2of(lemma.synset())] = word_tokenize(lemma.synset().definition())
    clus = softcos(defns)
    return clus
