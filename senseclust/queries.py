from senseclust.tables import freqs
from wikiparse.tables import headword, word_sense
from .consts import POS_MAP

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
