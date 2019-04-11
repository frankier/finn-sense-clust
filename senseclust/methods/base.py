import sys
from os.path import join as pjoin
from dataclasses import dataclass

from expcomb.models import Exp
from senseclust.exceptions import NoSuchLemmaException
from senseclust.eval import eval
from wikiparse.utils.db import get_session


@dataclass(frozen=True)
class ExpPathInfo:
    corpus: str
    guess: str
    gold: str

    def get_paths(self, iden, exp):
        guess_path = pjoin(self.guess, iden)
        print("get_paths", self.corpus, guess_path, None, self.gold)
        return self.corpus, guess_path, None, self.gold


class SenseClusExp(Exp):
    supports_wiktionary = False

    def run(self, words_fn, guess_fn):
        session = None
        with open(words_fn) as inf, open(guess_fn, "w") as outf:
            for lemma_name in inf:
                lemma_name = lemma_name.strip()
                try:
                    if self.supports_wiktionary:
                        if not session:
                            session = get_session()
                        clus_obj = self.clus_lemma(
                            lemma_name,
                            include_wiktionary=True,
                            session=session
                        )
                    else:
                        clus_obj = self.clus_lemma(lemma_name)
                except NoSuchLemmaException:
                    print(f"No such lemma: {lemma_name}", file=sys.stderr)
                else:
                    for k, v in sorted(clus_obj.items()):
                        num = k + 1
                        for ss in v:
                            print(f"{lemma_name}.{num:02},{ss}", file=outf)

    def calc_score(self, gold, guess_path):
        return eval(open(gold), open(guess_path), False)
