from .base import SenseClusExp
from expcomb.utils import mk_nick
from senseclust.utils import get_defns, unclusterable_default, graph_clust_grouped
from senseclust.methods.base import WiktionaryExpGroup
from wikiparse.utils.db import get_session
from .bert import bert_affinities
from .label import get_sense_sets, mat_of_sets
from .ety import ety
from nltk.corpus import wordnet
from scipy.spatial.distance import pdist
import numpy as np
from itertools import islice
from functools import partial


def comb_graph(lemma_name, pos, return_centers=False, do_label=False, do_ety_same=False, do_ety_diff=False):
    # Obtain and give ids to all defns which might be needed
    session = get_session()
    defns = get_defns(lemma_name, pos, include_wiktionary=True, session=session, tokenize=False, skip_empty=False)

    keys = list(defns.keys())
    if len(defns) <= 1:
        return unclusterable_default(keys, return_centers=return_centers)

    defn_ids = {}
    for idx, defn_key in enumerate(defns.keys()):
        defn_ids[defn_key] = idx

    # Start with *BERT*
    # XXX: what about clipping the BERT affinities
    affinities = bert_affinities(defns)

    # Overwrite with *Label* when affinity is larger
    if do_label:
        synsets, lemma_sets = get_sense_sets(lemma_name, pos)
        mat = mat_of_sets(lemma_sets)
        sims = 1 - pdist(mat.todense(), metric='russellrao')
        if len(sims):
            sims = sims / np.amax(sims)

            def adjust_idx(idx):
                return defn_ids[wordnet.ss2of(synsets[idx])]

            i = 0
            j = 1
            for sim in sims:
                ai = adjust_idx(i)
                aj = adjust_idx(j)
                if sim > affinities[ai, aj]:
                    affinities[ai, aj] = affinities[aj, ai] = sim
                j += 1
                if j >= len(synsets):
                    i += 1
                    j = i + 1
                    assert i < len(synsets)

    # Take Ety into account by setting links within the same ety to 1 and outside to 0
    if do_ety_same or do_ety_diff:
        ety_groups = ety(lemma_name, pos)

    if do_ety_same:
        # Set same ety links to 1
        for ety_idx, sense_ids in ety_groups.items():
            for s1_idx, s1 in enumerate(sense_ids):
                for s2 in sense_ids[s1_idx + 1:]:
                    sid1 = defn_ids[s1]
                    sid2 = defn_ids[s2]
                    affinities[sid1, sid2] = affinities[sid2, sid1] = 1

    if do_ety_diff:
        # Set different ety links to 0
        for ety_groups_idx, (ety_idx, sense_ids) in enumerate(ety_groups.items()):
            for ety_groups_idx2, (ety_idx2, sense_ids2) in enumerate(islice(ety_groups.items(), ety_groups_idx)):
                for s1 in sense_ids:
                    for s2 in sense_ids2:
                        sid1 = defn_ids[s1]
                        sid2 = defn_ids[s2]
                        affinities[sid1, sid2] = affinities[sid2, sid1] = 0

    # Cluster
    return graph_clust_grouped(affinities, keys, return_centers)


class Comb(SenseClusExp):
    returns_centers = True

    def __init__(self, do_label=False, do_ety_same=False, do_ety_diff=False):
        self.clus_func = partial(
            comb_graph,
            do_label=do_label,
            do_ety_same=do_ety_same,
            do_ety_diff=do_ety_diff
        )
        ety_both = do_ety_same and do_ety_diff
        if ety_both:
            ety_nick = "etyboth"
            ety_disp = "+EtyBoth"
        elif do_ety_same:
            ety_nick = "etysame"
            ety_disp = "+EtySame"
        elif do_ety_diff:
            ety_nick = "etydiff"
            ety_disp = "+EtyDiff"
        else:
            ety_nick = None
            ety_disp = ""
        super().__init__(
            ("Comb",),
            mk_nick("comb", do_label and "label" or None, ety_nick),
            "Bert" + (do_label and "+Label" or "") + ety_disp,
            None,
            {"do_label": do_label, "do_ety_same": do_ety_same, "do_ety_diff": do_ety_diff},
        )


class CombGroup(WiktionaryExpGroup):
    def __init__(self):
        variations = []
        for do_label in [False, True]:
            for do_ety_diff in [False, True]:
                for do_ety_same in [False, True]:
                    if not do_label and not do_ety_diff and not do_ety_same:
                        continue
                    variations.append(Comb(do_label, do_ety_same, do_ety_diff))
        return super().__init__(variations)
