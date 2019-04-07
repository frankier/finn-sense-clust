from pprint import pprint
import click
from senseclust.tables import freqs
from senseclust.queries import joined
from wikiparse.utils.db import get_session
from wikiparse.tables import headword, word_sense
from sqlalchemy.sql import distinct, select
from sqlalchemy.sql.functions import count


@click.command()
@click.argument("db")
@click.argument("limit", required=False, type=int)
@click.option("--verbose/--no-verbose")
def main(db: str, limit=50, verbose=False):
    query = select([
            headword.c.name,
            freqs.c.freq,
        ]).select_from(joined).where(
            word_sense.c.etymology_index.isnot(None) &
            (word_sense.c.pos == "Noun") &
            word_sense.c.inflection_of_id.is_(None)
        ).group_by(
            headword.c.id
        ).having(
            count(
                distinct(word_sense.c.etymology_index)
            ) > 1
        ).order_by(freqs.c.freq.desc()).limit(limit)
    session = get_session(db)
    candidates = session.execute(query).fetchall()
    for word, freq in candidates:
        print(word + ".Noun", "#", freq)
    if verbose:
        print("\n")
        for word, _ in candidates:
            print("#", word)
            pprint(session.execute(select([
                word_sense.c.sense_id,
                word_sense.c.sense,
            ]).select_from(joined).where(
                headword.c.name == word
            )).fetchall())


if __name__ == "__main__":
    main()
