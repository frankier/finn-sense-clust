import click
from senseclust.utils.clust import is_wn_ref
from .tables import cluster, metadata, Source
from wikiparse.utils.db import batch_commit, get_session
from wikiparse.utils.std import IterDirOrTar


def add_cmds(group):
    @group.command()
    @click.argument('indir', type=click.Path())
    def import_clust(indir):
        session = get_session()
        metadata.create_all(session().get_bind().engine)

        def clus_batch(line):
            line = line.decode('utf-8')
            lemma_clus, sense, exemp = line.strip().rsplit(",", 2)
            lemma, clus = lemma_clus.rsplit(".", 1)

            session.execute(cluster.insert().values(
                lemma=lemma,
                clus=int(clus),
                source=Source.wordnet if is_wn_ref(sense) else Source.wiktionary,
                sense=sense,
                exemp=exemp == "1",
            ))
        with click.progressbar(IterDirOrTar(indir), label="Inserting clusts") as clusts_chunks:
            lines = (
                line
                for _, clusts_chunk in clusts_chunks
                for line in clusts_chunk)
            batch_commit(session, lines, clus_batch)

    @group.command()
    def trunc_clust():
        print("Dropping", cluster.name)
        session = get_session()
        session.execute(f"TRUNCATE {cluster.name} RESTART IDENTITY CASCADE;")
        session.commit()
