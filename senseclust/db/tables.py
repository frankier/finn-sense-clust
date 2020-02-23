import enum
from sqlalchemy import Boolean, Column, Enum, Integer, String, MetaData, Table

metadata = MetaData()


class Source(enum.Enum):
    wordnet = 0
    wiktionary = 1


cluster = Table(
    "cluster",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("lemma", String),
    Column("clus", Integer),
    Column("source", Enum(Source)),
    Column("sense", String),
    Column("exemp", Boolean),
)
