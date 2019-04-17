from senseclust.methods.label import label_graph, Label
from senseclust.methods.vec_clust import vec_clust_autoextend_graph, Vec
from senseclust.methods.gloss import gloss_graph, Gloss
from senseclust.methods.baseline import Baseline
from senseclust.methods.base import ExpGroup, WiktionaryExpGroup


METHODS = {
    "label-graph": label_graph,
    "vec-clust-autoextend-graph": vec_clust_autoextend_graph,
    "gloss-graph": gloss_graph,
}

SUPPORTS_WIKTIONARY = {"gloss-graph"}


EXPERIMENTS = [
    WiktionaryExpGroup([Baseline("1"), Baseline("n")]),
    ExpGroup([Label()]),
    ExpGroup([Vec()]),
    WiktionaryExpGroup([Gloss()]),
]
