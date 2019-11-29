import os
from .base import SenseClusExp
from expcomb.utils import mk_nick
from senseclust.utils import (
    graph_clust_grouped, cos_affinities_none, unclusterable_default,
    get_wiktionary_defns, get_wordnet_defns
)
from senseclust.res import get_sent_trans
from senseclust.pre_embedded_glosses import get_pre_embed_wn, SENSE_SEP

import logging
logging.basicConfig(level=logging.INFO)


def encode_non_empty(defns):
    model = get_sent_trans()
    embedded = [None] * len(defns)
    idxs = []
    non_empty = []
    for idx, defn in enumerate(defns):
        if not defn:
            continue
        idxs.append(idx)
        non_empty.append(defn)
    for idx, vec in zip(idxs, model.encode(non_empty)):
        embedded[idx] = vec
    return embedded


def get_defns_layers(lemma_name, pos, skip_empty=True):
    pre_embed_wn = get_pre_embed_wn("PRE_EMBED_WN")
    pre_embed_wiki = get_pre_embed_wn("PRE_EMBED_WIKI")
    wiki_defns = {}
    wn_defns = {}
    layers = []
    for k, v in get_wiktionary_defns(lemma_name, pos,
                                     skip_empty=False, tokenize=False):
        wiki_defns[k] = v
        if pre_embed_wiki:
            layers.append(pre_embed_wiki.get_vec(lemma_name + SENSE_SEP + k))
    if not pre_embed_wiki and pre_embed_wn:
        layers.extend(encode_non_empty(wiki_defns.values()))
    for k, v in get_wordnet_defns(lemma_name, pos,
                                  skip_empty=False, tokenize=False):
        wn_defns[k] = v
        if pre_embed_wn:
            layers.append(pre_embed_wn.get_vec(k))
    if not pre_embed_wn:
        defns_values = []
        if not pre_embed_wiki:
            defns_values.extend(wiki_defns.values())
        defns_values.extend(wn_defns.values())
        layers.extend(encode_non_empty(defns_values))
    return {**wiki_defns, **wn_defns}, layers


def bert_clus(keys, layers, return_centers=False):
    if len(keys) <= 1:
        return unclusterable_default(keys, return_centers=return_centers)
    return graph_clust_grouped(cos_affinities_none(layers), keys, return_centers)


def bert_graph(lemma_name, pos, return_centers=False):
    defns, layers = get_defns_layers(lemma_name, pos)
    return bert_clus(list(defns.keys()), layers, return_centers)


class Bert(SenseClusExp):
    returns_centers = True

    def __init__(self):
        self.clus_func = bert_graph
        super().__init__(
            ("SentBert",),
            mk_nick("bert"),
            "SentBert",
            None,
            {},
        )
