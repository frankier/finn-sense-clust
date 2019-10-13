from sklearn.cluster import affinity_propagation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sqlalchemy.sql import select
from senseclust.consts import CLUS_LANG
from senseclust.queries import joined, lemma_where
from wikiparse.tables import word_sense
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import sys
import re


SYNSET_RE = re.compile(r"[0-9]{8}-[anv]")


def group_by(it):
    clus = {}
    for k, v in it:
        clus.setdefault(k, []).append(v)
    return clus


def mat_of_sets(sets):
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    return vectorizer.fit_transform(sets)


def self_xtab(li):
    for idx, e1 in enumerate(li):
        for e2 in li[:idx]:
            yield e1, e2


def graph_clust(affinities, return_centers=False):
    def default():
        if return_centers:
            return list(range(n)), list(range(n))
        else:
            return list(range(n))
    (n, n) = affinities.shape
    if n == 1:
        return default()
    mask = np.ones(affinities.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    if np.all(affinities[mask].flat == affinities[mask].flat[0]):
        if affinities[mask].flat[0] == 0:
            return default()
        else:
            result = [0] * n
            if return_centers:
                return result, []
            else:
                return result
    for damping in [0.5, 0.7, 0.9]:
        centers, labels = affinity_propagation(affinities, damping=damping)
        if not np.any(np.isnan(labels)):
            if return_centers:
                return labels, centers
            else:
                return labels
    return default()


def group_clust(labels, clust_labels):
    return group_by(((clust_num, labels[idx]) for idx, clust_num in enumerate(clust_labels)))


def split_line(line):
    frame_id, lemma_id = line.strip().split(",", 1)
    lemma, frame_no = frame_id.split(".", 1)
    return lemma, frame_no, lemma_id


def get_defns(lemma_name, pos, include_wiktionary=False, session=None, skip_empty=True):
    defns = {}
    # Add wiktionary senses
    wiktionary_senses = session.execute(select([
        word_sense.c.sense_id,
        word_sense.c.etymology_index,
        word_sense.c.sense,
        word_sense.c.extra,
    ]).select_from(joined).where(
        lemma_where(lemma_name, pos)
    )).fetchall()
    for row in wiktionary_senses:
        tokens = word_tokenize(row["sense"])
        if skip_empty and not tokens:
            sys.stderr.write(f"Empty defn: {row['sense_id']} '{row['sense']}'\n")
            continue
        defns[row["sense_id"]] = tokens
    # Add WordNet senses
    wordnet_senses = wordnet.lemmas(lemma_name, lang=CLUS_LANG)
    for lemma in wordnet_senses:
        defns[wordnet.ss2of(lemma.synset())] = word_tokenize(lemma.synset().definition())
    return defns


def unclusterable_default(keys, return_centers=False):
    labels = list(range(len(keys)))
    clus = group_clust(keys, labels)
    if return_centers:
        return clus, keys
    else:
        return clus


def is_wn_ref(ref):
    return SYNSET_RE.match(ref)
