import click
from vecstorenn import VecStorage
from senseclust.res import get_sent_trans, get_batch_size
from nltk.corpus import wordnet
from itertools import islice
import logging
import os


SENT_BERT_SIZE = 1024
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


@click.command()
@click.argument("out-path", type=click.Path())
def pre_embed_glosses(out_path: str):
    model = get_sent_trans()
    batch_size = get_batch_size()
    all_synsets = wordnet.all_synsets()
    with VecStorage(out_path, SENT_BERT_SIZE, "wi") as storage:
        while 1:
            batch = islice(all_synsets, batch_size)
            ids = []
            defns = []
            for synset in batch:
                ids.append(wordnet.ss2of(synset))
                defns.append(synset.definition())
            if not len(ids):
                break
            logger.info("ids: %s", ids)
            logger.info("defns: %s", defns)
            encoded = model.encode(defns)
            for id, vec in zip(ids, encoded):
                storage.add_vec(id, vec)


if __name__ == "__main__":
    pre_embed_glosses()
