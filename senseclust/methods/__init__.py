from senseclust.methods.label import label_graph
from senseclust.methods.vec_clust import vec_clust_autoextend_graph
from senseclust.methods.gloss import gloss_graph


METHODS = {
    "label-graph": label_graph,
    "vec-clust-autoextend-graph": vec_clust_autoextend_graph,
    "gloss-graph": gloss_graph,
}

SUPPORTS_WIKTIONARY = {"gloss-graph"}
