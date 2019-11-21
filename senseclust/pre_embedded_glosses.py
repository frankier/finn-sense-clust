import os
from vecstorenn import VecStorage


SENSE_SEP = "!S&!"
SENT_BERT_SIZE = 1024


def wiktionary_sense_id(row):
    return row["name"] + SENSE_SEP + row["sense_id"]


vec_storages = {}


def get_pre_embed_wn(env):
    if env in vec_storages:
        return vec_storages[env]
    pre_embed = os.environ.get(env)
    if not pre_embed:
        return
    vec_storages[env] = VecStorage(pre_embed, SENT_BERT_SIZE, "ri")
    return vec_storages[env]
