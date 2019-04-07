import click
from wikiparse.utils.db import get_session, insert
import wordfreq
from senseclust.tables import metadata, freqs


@click.command()
@click.argument("db")
def add_freq_data(db: str):
    session = get_session(db)
    metadata.create_all(session().get_bind().engine)
    with click.progressbar(wordfreq.get_frequency_dict("fi").items(), label="Inserting frequencies") as name_freqs:
        for name, freq in name_freqs:
            insert(session, freqs, name=name, freq=freq)
    session.commit()


if __name__ == "__main__":
    add_freq_data()
