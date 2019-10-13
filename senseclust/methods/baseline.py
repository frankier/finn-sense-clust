from senseclust.utils import group_clust, get_defns
from .base import SenseClusExp
from expcomb.utils import mk_nick
from wikiparse.utils.db import get_session


def mk_baseline(mode="1"):
    def baseline(lemma_name, pos):
        session = get_session()
        defns = get_defns(
            lemma_name,
            pos,
            include_wiktionary=True,
            session=session
        )
        keys = list(defns.keys())
        labels = [1] * len(keys) if mode == "1" else range(1, len(keys) + 1)
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
