from finntk.wordnet.utils import fi2en_post, en2fi_post
from finntk.wordnet.reader import fiwn
from nltk.corpus import wordnet
from stiff.wordnet.fin import Wordnet as WordnetFin
from stiff.wordnet.utils import synset_key_lemmas
from finntk.wordnet.reader import fiwn_encnt


def get_lemma_names(ssof, wns):
    wns = list(wns)
    lemmas = []
    if "qf2" in wns:
        fi_ssof = en2fi_post(ssof)
        ss = fiwn.of2ss(fi_ssof)
        lemmas.extend(ss.lemmas())
        wns.remove("qf2")
    for wnref in wns:
        ss = wordnet.of2ss(ssof)
        lemmas.extend(ss.lemmas(lang=wnref))
    return {l.name() for l in lemmas}


def lemmas(lemma_name, wn, pos=None):
    if wn == "qf2":
        return fiwn_encnt.lemmas(lemma_name, pos=pos)
    else:
        return wordnet.lemmas(lemma_name, pos=pos, lang=wn)


def get_lemma_objs(lemma_name, wns, pos=None):
    wn_lemmas = {}
    for wn in wns:
        for lemma in lemmas(lemma_name, wn, pos=pos):
            wn_lemmas.setdefault(wn, []).append(lemma)
    return synset_key_lemmas(wn_lemmas, WordnetFin)


WORDNETS = ['fin', 'qf2', 'qwf']
