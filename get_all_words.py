import click
from stiff.wordnet.fin import Wordnet as WordnetFin
from sqlalchemy.sql import select
from wikiparse.tables import headword, word_sense
from wikiparse.utils.db import get_session
from senseclust.wordnet import get_lemma_objs
from senseclust.queries import joined
from finntk.data.wordnet_pos import POS_MAP


def norm(lemma):
    return lemma.replace(" ", "_").lower()


@click.command()
@click.argument("mode", type=click.Choice(["all", "wn", "wiki"]),
                required=False)
def main(mode="all"):
    if mode != "wiki":
        for lemma, wns in WordnetFin.lemma_names().items():
            lemma_obj = next(iter(get_lemma_objs(lemma, wns).values()))[0][1]
            pos = lemma_obj.synset().pos()
            if pos == "s":
                pos = "a"
            lemma = norm(lemma)
            print(f"{lemma},{pos}")

    if mode != "wn":
        session = get_session()

        for pos in POS_MAP.keys():
            for row in session.execute(select([
                headword.c.name
            ]).select_from(joined).where(
                word_sense.c.pos.in_(POS_MAP[pos]) &
                word_sense.c.inflection_of_id.is_(None)
            ).distinct()).fetchall():
                lemma = norm(row[0])
                print(f"{lemma},{pos}")


if __name__ == "__main__":
    main()
