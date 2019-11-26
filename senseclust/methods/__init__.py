from senseclust.methods.label import Label
from senseclust.methods.vec_clust import Vec
from senseclust.methods.scos import SoftCos
from senseclust.methods.baseline import Baseline
from senseclust.methods.base import BothExpGroup, WordnetOnlyExpGroup, WiktionaryOnlyExpGroup
from senseclust.methods.bert import Bert
from senseclust.methods.wmd import Wmd
from senseclust.methods.ety import Ety
from senseclust.methods.comb import CombGroup


EXPERIMENTS = [
    BothExpGroup([Baseline("1"), Baseline("n")]),
    WordnetOnlyExpGroup([Label()]),
    WordnetOnlyExpGroup([Vec()]),
    BothExpGroup([SoftCos(False), SoftCos(False)]),
    BothExpGroup([Bert()]),
    BothExpGroup([Wmd(False, False), Wmd(False, True), Wmd(True, False), Wmd(True, True)]),
    WiktionaryOnlyExpGroup([Ety()]),
    CombGroup(),
]
