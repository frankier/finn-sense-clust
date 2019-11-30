from .base import SenseClusExp
from expcomb.utils import mk_nick
from senseclust.exceptions import NoSuchLemmaException
from senseclust.utils import unclusterable_default, graph_clust_grouped, cos_affinities_none, get_defns
from senseclust.methods.base import BothExpGroup
from .bert import get_defns_layers
from .label import get_sense_sets, mat_of_sets
from .ety import ety
from .wmd import wmd_affinities, wmdistance_partial, wmdistance
from scipy.spatial.distance import pdist
import numpy as np
from itertools import islice
from functools import partial


def comb_graph(lemma_name, pos, return_centers=False, do_bert=False, do_label=False, do_wmdsyn=False, do_wmdpartsyn=False, do_ety_same=False, do_ety_diff=False, do_ety_exemp=False):
    # Obtain and give ids to all defns which might be needed
    defns, layers = get_defns_layers(lemma_name, pos, skip_empty=False)

    keys = list(defns.keys())
    num_defns = len(defns)
    if num_defns <= 1:
        return unclusterable_default(keys, return_centers=return_centers)

    defn_ids = {}
    for idx, defn_key in enumerate(defns.keys()):
        defn_ids[defn_key] = idx

    # Start with *BERT*
    if do_bert:
        affinities = cos_affinities_none(layers)
    else:
        affinities = np.eye(num_defns)

    # Overwrite with *Label* when affinity is larger
    if do_label:
        try:
            labels, lemma_sets = get_sense_sets(lemma_name, pos)
        except NoSuchLemmaException:
            pass
        else:
            mat = mat_of_sets(lemma_sets)
            sims = 1 - pdist(mat.todense(), metric='russellrao')
            if len(sims):
                sims = sims / np.amax(sims)

                def adjust_idx(idx):
                    return defn_ids[labels[idx]]

                i = 0
                j = 1
                for sim in sims:
                    ai = adjust_idx(i)
                    aj = adjust_idx(j)
                    if sim > affinities[ai, aj]:
                        affinities[ai, aj] = affinities[aj, ai] = sim
                    j += 1
                    if j >= len(labels):
                        i += 1
                        j = i + 1
                        assert i < len(labels)

    # Overwrite with WmdSyn or WmdPartSyn
    if do_wmdsyn or do_wmdpartsyn:
        wmd_defns = get_defns(lemma_name, pos, lower=True, include_enss=True)
        np.maximum(affinities, wmd_affinities(wmdistance_partial if do_wmdpartsyn else wmdistance, wmd_defns), out=affinities)

    # Take Ety into account by setting links within the same ety to 1 and outside to 0
    if do_ety_same or do_ety_diff or do_ety_exemp:
        ety_groups = ety(lemma_name, pos)

    if do_ety_same and len(ety_groups) > 1:
        # Set same ety links to 1
        for ety_idx, sense_ids in ety_groups.items():
            for s1_idx, s1 in enumerate(sense_ids):
                for s2 in sense_ids[s1_idx + 1:]:
                    sid1 = defn_ids[s1]
                    sid2 = defn_ids[s2]
                    affinities[sid1, sid2] = affinities[sid2, sid1] = 1

    if do_ety_diff and len(ety_groups) > 1:
        # Set different ety links to 0
        for ety_groups_idx, (ety_idx, sense_ids) in enumerate(ety_groups.items()):
            for ety_groups_idx2, (ety_idx2, sense_ids2) in enumerate(islice(ety_groups.items(), ety_groups_idx)):
                for s1 in sense_ids:
                    for s2 in sense_ids2:
                        sid1 = defn_ids[s1]
                        sid2 = defn_ids[s2]
                        affinities[sid1, sid2] = affinities[sid2, sid1] = 0

    # Set up ety exemplars
    preference = np.zeros(num_defns)
    if do_ety_exemp and len(ety_groups) > 1:
        med = np.median(affinities)
        for ety_idx, sense_ids in ety_groups.items():
            preference[defn_ids[sense_ids[0]]] = med

    # Cluster
    return graph_clust_grouped(affinities, keys, return_centers, preference)


class Comb(SenseClusExp):
    returns_centers = True

    def __init__(self, do_bert=False, do_label=False, do_wmdsyn=False, do_wmdpartsyn=False, do_ety=False, do_ety_exemp=False):
        self.clus_func = partial(
            comb_graph,
            do_bert=do_bert,
            do_label=do_label,
            do_wmdsyn=do_wmdsyn,
            do_wmdpartsyn=do_wmdpartsyn,
            do_ety_diff=do_ety,
            do_ety_exemp=do_ety_exemp,
        )
        disp = ""
        nick_bits = []

        def add(opt, nick, d=None):
            nonlocal disp, nick_bits
            nick_bits.append(nick)
            if not d:
                d = nick.upper()
            if disp:
                disp += "+"
            disp += d

        add(do_bert, "bert", "SBert")
        add(do_label, "lbl")
        add(do_wmdsyn, "ws")
        add(do_wmdpartsyn, "wps")
        add(do_ety, "ety")
        add(do_ety_exemp, "ex")
        super().__init__(
            ("Comb",),
            mk_nick("comb", *nick_bits),
            disp,
            None,
            {
                "do_bert": do_bert,
                "do_label": do_label,
                "do_wmdsyn": do_wmdsyn,
                "do_wmdpartsyn": do_wmdpartsyn,
                "do_ety": do_ety,
                "do_ety_exemp": do_ety_exemp
            },
        )


class CombGroup(BothExpGroup):
    def __init__(self):
        variations = []
        for do_bert in [False, True]:
            for do_wmdsyn in [False, True]:
                for do_wmdpartsyn in [False, True]:
                    if do_wmdsyn and do_wmdpartsyn:
                        # ~(do_wmdsyn & do_wmdpartsyn)
                        continue
                    if not do_bert and not do_wmdsyn and not do_wmdpartsyn:
                        continue
                    for do_label in [False, True]:
                        for do_ety in [False, True]:
                            for do_ety_exemp in [False, True]:
                                if do_ety_exemp and not do_ety:
                                    # do_ety_exemp => do_ety
                                    continue
                                if (do_bert + do_label + do_wmdsyn + do_wmdpartsyn + do_ety + do_ety_exemp) == 1:
                                    continue
                                variations.append(Comb(do_bert, do_label, do_wmdsyn, do_wmdpartsyn, do_ety, do_ety_exemp))
        return super().__init__(variations)
