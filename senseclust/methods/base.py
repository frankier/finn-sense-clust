import os
import sys
from dataclasses import dataclass

from expcomb.models import Exp, ExpGroup as ExpGroupBase
from senseclust.exceptions import NoSuchLemmaException
from senseclust.eval import eval, UnguessedException
from click.utils import LazyFile
from os.path import exists


@dataclass(frozen=True)
class ExpPathInfo:
    corpus: str
    guess: str
    gold: str

    def get_paths(self, iden, exp):
        return self.corpus, self.guess, None, self.gold


class SenseClusExp(Exp):
    returns_centers = False

    def run(self, words_fn, guess_fn, exemplars=False):
        add_exemplars = self.returns_centers and exemplars
        with open(words_fn) as inf, LazyFile(guess_fn, "w") as outf:
            try:
                for line in inf:
                    lemma_name, pos = line.strip().rsplit(",", 1)
                    try:
                        if add_exemplars:
                            clus_obj, centers = self.clus_lemma(lemma_name, pos, True)
                        else:
                            clus_obj = self.clus_lemma(lemma_name, pos)
                            centers = []
                    except NoSuchLemmaException:
                        print(f"No such lemma: {lemma_name}", file=sys.stderr)
                    else:
                        for k, v in sorted(clus_obj.items()):
                            num = k + 1
                            for ss in v:
                                if add_exemplars:
                                    exemplar = "1" if ss in centers else "0"
                                    print(f"{lemma_name}.{num:02},{ss},{exemplar}", file=outf)
                                else:
                                    print(f"{lemma_name}.{num:02},{ss}", file=outf)
            except Exception:
                # This is probably a partial guess: delete it to avoid getting
                # incorrect results
                outf.close()
                if exists(guess_fn):
                    os.unlink(guess_fn)
                raise

    def calc_score(self, gold, guess_path):
        try:
            return eval(open(gold), open(guess_path), False)
        except UnguessedException as exc:
            exc.gold_fn = gold
            exc.guess_fn = guess_path
            raise

    def clus_lemma(self, *args, **kwargs):
        return self.clus_func(*args, **kwargs)


class ExpGroup(ExpGroupBase):
    supports_wiktionary = False
    supports_wordnet = False
    group_attrs = ("supports_wiktionary", "supports_wordnet")


class WiktionaryOnlyExpGroup(ExpGroup):
    supports_wiktionary = True


class WordnetOnlyExpGroup(ExpGroup):
    supports_wordnet = True


class BothExpGroup(ExpGroup):
    supports_wiktionary = True
    supports_wordnet = True
