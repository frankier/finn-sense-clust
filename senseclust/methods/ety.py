from .base import SenseClusExp
from senseclust.utils import get_wiktionary, get_session
from expcomb.utils import mk_nick


def ety(lemma_name, pos):
    session = get_session()
    result = {}
    for row in get_wiktionary(session, lemma_name, pos):
        result.setdefault(row["etymology_index"], []).append(row["sense_id"])
    return result


class Ety(SenseClusExp):
    def __init__(self):
        self.clus_func = ety
        super().__init__(
            ("Ety",),
            mk_nick("ety"),
            "Ety",
            None,
            {},
        )
