import click
from vecstorenn import VecStorage
from senseclust.res import get_sent_trans, get_batch_size
from senseclust.queries import wiktionary_query_all
from senseclust.pre_embedded_glosses import SENT_BERT_SIZE, wiktionary_sense_id
from wikiparse.utils.db import get_session
from nltk.corpus import wordnet
from itertools import islice
import logging
import os


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def wordnet_batcher():
    all_synsets = wordnet.all_synsets()
    batch_size = get_batch_size()
    while 1:
        batch = islice(all_synsets, batch_size)
        ids = []
        defns = []
        for synset in batch:
            ids.append(wordnet.ss2of(synset))
            defns.append(synset.definition())
        if not ids:
            return
        yield ids, defns


def wiktionary_batcher():
    session = get_session()
    rows = session.execute(wiktionary_query_all()).fetchall()
    batch_size = get_batch_size()
    ids = []
    defns = []
    for row in rows:
        defn = row["sense"].strip()
        if not defn:
            continue
        ids.append(wiktionary_sense_id(row))
        defns.append(defn)
        if len(ids) >= batch_size:
            yield ids, defns
            ids = []
            defns = []


@click.command()
@click.argument("mode", type=click.Choice(["wn", "wiki"]))
@click.argument("out-path", type=click.Path())
def pre_embed_glosses(mode: str, out_path: str):
    model = get_sent_trans()
    if mode == "wn":
        batcher = wordnet_batcher()
    else:
        batcher = wiktionary_batcher()
    with VecStorage(out_path, SENT_BERT_SIZE, "wi") as storage:
        for ids, defns in batcher:
            logger.info("ids: %s", ids)
            logger.info("defns: %s", defns)
            encoded = model.encode(defns)
            for id, vec in zip(ids, encoded):
                storage.add_vec(id, vec)


if __name__ == "__main__":
    pre_embed_glosses()
