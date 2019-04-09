import sys
import click
from nltk.corpus import wordnet
from senseclust.exceptions import NoSuchLemmaException
from senseclust.methods import METHODS, SUPPORTS_WIKTIONARY
from wikiparse.utils.db import get_session


@click.group()
def senseclust():
    pass


@senseclust.command("run")
@click.argument("method", type=click.Choice(METHODS.keys()))
@click.argument("lemmas", type=click.File('r'))
@click.argument("db", required=False)
def run(method, lemmas, db=None):
    session = None
    for lemma_name in lemmas:
        lemma_name = lemma_name.strip()
        try:
            if method in SUPPORTS_WIKTIONARY:
                if not session:
                    session = get_session(db)
                clus_obj = METHODS[method](lemma_name, include_wiktionary=True, session=session)
            else:
                clus_obj = METHODS[method](lemma_name)
        except NoSuchLemmaException:
            print(f"No such lemma: {lemma_name}", file=sys.stderr)
        else:
            for k, v in sorted(clus_obj.items()):
                num = k + 1
                for ss in v:
                    print(f"{lemma_name}.{num:02},{ss}")


if __name__ == "__main__":
    senseclust()
