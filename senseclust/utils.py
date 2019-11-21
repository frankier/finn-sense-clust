from sklearn.cluster import affinity_propagation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist, squareform
import numpy as np
from finntk.wordnet.utils import pre_id_to_post
from senseclust.consts import CLUS_LANG
from senseclust.wordnet import get_lemma_objs, WORDNETS
from senseclust.queries import wiktionary_query
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


def graph_clust_grouped(affinities, labels, return_centers=False):
    if return_centers:
        clust_labels, centers = graph_clust(affinities, return_centers=True)
        return group_clust(labels, clust_labels), [labels[c] for c in centers]
    else:
        clust_labels = graph_clust(affinities)
        return group_clust(labels, clust_labels)


def split_line(line):
    frame_id, lemma_id = line.strip().split(",", 1)
    lemma, frame_no = frame_id.split(".", 1)
    return lemma, frame_no, lemma_id


def cos_affinities(mat):
    dists = pdist(mat, metric='cosine')
    # XXX: Should this sometimes be clipped to above 0
    sim = squareform(1 - dists)
    np.fill_diagonal(sim, 1)
    return sim


def get_wiktionary(session, lemma_name, pos):
    return session.execute(wiktionary_query(lemma_name, pos)).fetchall()


def get_defns(
    lemma_name,
    pos,
    include_wiktionary=True,
    session=None,
    include_wordnet=True,
    skip_empty=True,
    tokenize=True
):
    defns = {}
    # Add wiktionary senses
    if include_wiktionary:
        assert session is not None
        for row in get_wiktionary(session, lemma_name, pos):
            tokens = row["sense"].strip()
            if skip_empty and not tokens:
                sys.stderr.write(f"Empty defn: {row['sense_id']} '{row['sense']}'\n")
                continue
            if tokenize:
                tokens = word_tokenize(tokens)
            defns[row["sense_id"]] = tokens
    # Add WordNet senses
    if include_wordnet:
        for synset_id, lemma_objs in get_lemma_objs(lemma_name, WORDNETS, pos).items():
            assert len(lemma_objs) >= 1
            tokens = lemma_objs[0][1].synset().definition().strip()
            if skip_empty and not tokens:
                sys.stderr.write(f"Empty defn: {lemma_name}.{pos}: {synset_id}'\n")
                continue
            if tokenize:
                tokens = word_tokenize(tokens)
            defns[pre_id_to_post(synset_id)] = tokens
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


def mk_dictionary_bow_corpus(docs):
    from gensim.corpora import Dictionary
    dictionary = Dictionary()
    bow_corpus = []
    for document in docs:
        bow_corpus.append(dictionary.doc2bow(document, allow_update=True))
    return dictionary, bow_corpus
