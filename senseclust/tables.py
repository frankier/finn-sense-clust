from sqlalchemy import Column, String, MetaData, Table, Float

metadata = MetaData()


freqs = Table(
    "freqs",
    metadata,
    Column("name", String, unique=True),
    Column("freq", Float),
)
