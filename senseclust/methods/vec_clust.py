from finntk.wordnet.utils import pre_id_to_post
from finntk.emb.autoextend import vecs, get_lemma_id
from senseclust.utils import graph_clust, group_clust, cos_affinities_none
from senseclust.wordnet import get_lemma_objs, WORDNETS
from .base import SenseClusExp
from expcomb.utils import mk_nick


def vec_clust_autoextend_graph(lemma_name, pos):
    fiwn_space = vecs.get_vecs()
    labels = []
    mat = []
    for synset_id, lemma_objs in get_lemma_objs(lemma_name, WORDNETS, pos).items():
        labels.append(pre_id_to_post(synset_id))
        lemma_obj = None
        for wn, l in lemma_objs:
            if wn == "qf2":
                lemma_obj = l
        if lemma_obj is None:
            mat.append(None)
            continue
        lemma_id = get_lemma_id(lemma_obj)
        try:
            vec = fiwn_space[lemma_id]
        except KeyError:
            mat.append(None)
            continue
        mat.append(vec)
    affinities = cos_affinities_none(mat)
    clust_labels = graph_clust(affinities)
    return group_clust(labels, clust_labels)


class Vec(SenseClusExp):
    def __init__(self):
        self.clus_func = vec_clust_autoextend_graph
        super().__init__(
            ("SenseVec",),
            mk_nick("sense-vec"),
            "Sensevec",
            None,
            {},
        )
