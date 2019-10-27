from .base import SenseClusExp
from expcomb.utils import mk_nick
from senseclust.utils import graph_clust_grouped, get_defns, cos_affinities, unclusterable_default
from wikiparse.utils.db import get_session

from sentence_transformers import SentenceTransformer

import logging
logging.basicConfig(level=logging.INFO)

_model = None


def get_models():
    global _model
    if _model is None:
        _model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
    return _model


def defns_to_berts(defns):
    model = get_models()
    sentences = list(defns.values())
    return model.encode(sentences)


def bert_clus(defns, return_centers=False):
    keys = list(defns.keys())
    if len(defns) <= 1:
        return unclusterable_default(keys, return_centers=return_centers)
    layers = defns_to_berts(defns)
    affinities = cos_affinities(layers)
    return graph_clust_grouped(affinities, keys, return_centers)


def bert_graph(lemma_name, pos, return_centers=False):
    session = get_session()
    defns = get_defns(lemma_name, pos, include_wiktionary=True, session=session, tokenize=False)
    return bert_clus(defns)


class Bert(SenseClusExp):
    returns_centers = True

    def __init__(self):
        self.clus_func = bert_graph
        super().__init__(
            ("Bert",),
            mk_nick("bert"),
            "Bert",
            None,
            {},
        )
