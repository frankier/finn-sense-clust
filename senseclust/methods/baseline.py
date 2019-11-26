from senseclust.utils import group_clust, get_defns
from .base import SenseClusExp
from expcomb.utils import mk_nick


def mk_baseline(mode="1"):
    def baseline(lemma_name, pos):
        defns = get_defns(lemma_name, pos)
        keys = list(defns.keys())
        labels = [0] * len(keys) if mode == "1" else range(len(keys))
        return group_clust(keys, labels)
    return baseline


class Baseline(SenseClusExp):
    def __init__(self, mode="1"):
        self.clus_func = mk_baseline(mode)
        super().__init__(
            ("Baseline",),
            mk_nick("baseline", mode),
            "One cluster" if mode == "1" else "N clusters",
            None,
            {"mode": mode},
        )
