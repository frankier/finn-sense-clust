from os.path import join as pjoin
import click
from wikiparse.utils.db import get_session
from wikiparse.tables import headword, word_sense
from sqlalchemy.sql import select
from senseclust.queries import joined
from senseclust.wordnet import get_lemma_objs, WORDNETS
from stiff.writers import annotation_comment
from finntk.wordnet.utils import pre_id_to_post


@click.command()
@click.argument("db")
@click.argument("words", type=click.File('r'))
@click.argument("out_dir")
def main(db, words, out_dir):
    session = get_session(db)
    for word in words:
        word_pos = word.split("#")[0].strip()
        word, pos = word_pos.split(".")
        assert pos == "Noun"
        with open(pjoin(out_dir, word_pos), "w") as outf:
            # Get Wiktionary results
            results = session.execute(select([
                word_sense.c.sense_id,
                word_sense.c.etymology_index,
                word_sense.c.sense,
                word_sense.c.extra,
            ]).select_from(joined).where(
                (headword.c.name == word) &
                (word_sense.c.pos == "Noun")
            ).order_by(word_sense.c.etymology_index)).fetchall()
            prev_ety = None
            for row in results:
                if prev_ety is not None and row["etymology_index"] != prev_ety:
                    outf.write("\n")
                outf.write("{} # {}\n".format(row["sense_id"], row["extra"]["raw_defn"].strip().replace("\n", " --- ")))
                prev_ety = row["etymology_index"]

            # Get WordNet results
            for synset_id, lemma_objs in get_lemma_objs(word, "n", WORDNETS).items():
                wordnets = {wn for wn, _ in lemma_objs}
                outf.write("\n")
                outf.write("{} # [{}] {}\n".format(pre_id_to_post(synset_id), ", ".join(wordnets), annotation_comment(lemma_objs)))


if __name__ == "__main__":
    main()
