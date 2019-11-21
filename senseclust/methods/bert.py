from .base import SenseClusExp
from expcomb.utils import mk_nick
from senseclust.utils import graph_clust_grouped, get_defns, cos_affinities, unclusterable_default
from senseclust.res import get_sent_trans
from wikiparse.utils.db import get_session

import logging
logging.basicConfig(level=logging.INFO)


def defns_to_berts(defns):
    model = get_sent_trans()
    sentences = list(defns.values())
    return model.encode(sentences)


def bert_affinities(defns):
    layers = defns_to_berts(defns)
    return cos_affinities(layers)


def bert_clus(defns, return_centers=False):
    keys = list(defns.keys())
    if len(defns) <= 1:
        return unclusterable_default(keys, return_centers=return_centers)
    return graph_clust_grouped(bert_affinities(defns), keys, return_centers)


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
