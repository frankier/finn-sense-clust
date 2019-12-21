from sqlalchemy.sql import select
from senseclust.tables import freqs
from wikiparse.tables import headword, word_sense
from finntk.data.wordnet_pos import POS_MAP

joined = (
    headword.join(
        word_sense,
        word_sense.c.headword_id == headword.c.id
    )
)

joined_freq = joined.join(
    freqs,
    headword.c.name == freqs.c.name
)


def lemma_where(lemma_name, pos):
    return (
        (headword.c.name == lemma_name) &
        word_sense.c.pos.in_(POS_MAP[pos]) &
        word_sense.c.inflection_of_id.is_(None))


def wiktionary_query_all():
    return select([
        headword.c.name,
        word_sense.c.sense_id,
        word_sense.c.etymology_index,
        word_sense.c.sense,
        word_sense.c.extra,
    ]).select_from(joined)


def wiktionary_query(lemma_name, pos):
    return select([
        word_sense.c.sense_id,
        word_sense.c.etymology_index,
        word_sense.c.sense,
        word_sense.c.extra,
    ]).select_from(joined).where(
        lemma_where(lemma_name, pos)
    )
