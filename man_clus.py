from pprint import pprint
import click
from senseclust.queries import joined
from wikiparse.tables import headword, word_sense
from sqlalchemy.sql import distinct, select
from sqlalchemy.sql.functions import count
from os.path import join as pjoin
from senseclust.wordnet import get_lemma_objs, WORDNETS
from stiff.writers import annotation_comment
from finntk.wordnet.utils import pre_id_to_post
from wikiparse.utils.db import get_session, insert
import wordfreq
from senseclust.tables import metadata, freqs
from os.path import basename
import re


SYNSET_RE = re.compile(r"[0-9]{8}-[anv]")


@click.group()
def man_clus():
    pass


@man_clus.command()
@click.argument("db")
@click.argument("words", type=click.File('r'))
@click.argument("out_dir")
def gen(db, words, out_dir):
    """
    Generate unclustered words in OUT_DIR from word list WORDS
    """
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


@man_clus.command()
@click.argument("db")
def add_freq_data(db: str):
    """
    Add table of frequencies to DB
    """
    session = get_session(db)
    metadata.create_all(session().get_bind().engine)
    with click.progressbar(wordfreq.get_frequency_dict("fi").items(), label="Inserting frequencies") as name_freqs:
        for name, freq in name_freqs:
            insert(session, freqs, name=name, freq=freq)
    session.commit()


@man_clus.command()
@click.argument("infs", nargs=-1)
@click.argument("out", type=click.File('w'))
def compile(infs, out):
    """
    Compile manually clustered words in files INFS to OUT as a gold csv ready
    for use by eval
    """
    out.write("manann,ref\n")
    for inf in infs:
        word_pos = basename(inf)
        word = word_pos.split(".")[0]
        idx = 1
        with open(inf) as f:
            for line in f:
                if not line.strip():
                    idx += 1
                else:
                    ref = line.split("#")[0].strip()
                    out.write(f"{word}.{idx:02d},{ref}\n")


@man_clus.command()
@click.argument("inf", type=click.File('r'))
@click.argument("outf", type=click.File('w'))
def filter_wn(inf, outf):
    """
    Filter a gold CSV to remove non-WordNet rows
    """
    assert inf.readline().strip() == "manann,ref"
    outf.write("manann,ref\n")
    for line in inf:
        manann, ref = line.strip().split(",")
        if not SYNSET_RE.match(ref):
            continue
        outf.write(line)


@man_clus.command()
@click.argument("db")
@click.argument("limit", required=False, type=int)
@click.option("--verbose/--no-verbose")
def pick_words(db: str, limit=50, verbose=False):
    """
    Pick etymologically ambigious nouns for creating manual clustering.
    """
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
    man_clus()
