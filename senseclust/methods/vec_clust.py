from finntk.wordnet.utils import pre_id_to_post
from finntk.emb.autoextend import vecs, get_lemma_id
from senseclust.utils.clust import graph_clust_grouped, cos_affinities_none, unclusterable_default
from senseclust.wordnet import get_lemma_objs, WORDNETS
from .base import SenseClusExp
from expcomb.utils import mk_nick


def vec_clust_autoextend_graph(lemma_name, pos, return_centers=False):
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
    if not labels:
        return unclusterable_default(labels, return_centers=return_centers)
    affinities = cos_affinities_none(mat)
    return graph_clust_grouped(affinities, labels, return_centers)


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
