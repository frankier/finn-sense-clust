from senseclust.methods.label import Label
from senseclust.methods.vec_clust import Vec
from senseclust.methods.gloss import Gloss
from senseclust.methods.baseline import Baseline
from senseclust.methods.base import ExpGroup, WiktionaryExpGroup


EXPERIMENTS = [
    WiktionaryExpGroup([Baseline("1"), Baseline("n")]),
    ExpGroup([Label()]),
    ExpGroup([Vec()]),
    WiktionaryExpGroup([Gloss()]),
]
