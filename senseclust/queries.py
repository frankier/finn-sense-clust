from senseclust.tables import freqs
from wikiparse.tables import headword, word_sense

joined = (
    headword.join(
        word_sense,
        word_sense.c.headword_id == headword.c.id
    ).join(
        freqs,
        headword.c.name == freqs.c.name
    )
)
