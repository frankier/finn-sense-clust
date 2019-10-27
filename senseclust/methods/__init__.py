from senseclust.methods.label import Label
from senseclust.methods.vec_clust import Vec
from senseclust.methods.scos import SoftCos
from senseclust.methods.baseline import Baseline
from senseclust.methods.base import ExpGroup, WiktionaryExpGroup
from senseclust.methods.bert import Bert
from senseclust.methods.wmd import Wmd
from senseclust.methods.ety import Ety


EXPERIMENTS = [
    WiktionaryExpGroup([Baseline("1"), Baseline("n")]),
    ExpGroup([Label()]),
    ExpGroup([Vec()]),
    WiktionaryExpGroup([SoftCos(False), SoftCos(True)]),
    WiktionaryExpGroup([Bert()]),
    WiktionaryExpGroup([Wmd(False), Wmd(True)]),
    WiktionaryExpGroup([Ety()]),
]
