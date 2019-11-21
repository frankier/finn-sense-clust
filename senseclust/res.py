import os


_sent_trans = None


def get_sent_trans():
    from sentence_transformers import SentenceTransformer
    global _sent_trans
    if _sent_trans is None:
        _sent_trans = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
    return _sent_trans


def get_batch_size():
    return int(os.environ.get("BATCH_SIZE", "32"))
