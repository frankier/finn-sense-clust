import csv
import sys
from finntk.wordnet.utils import fi2en_post
import fileinput
from nltk.corpus import wordnet
import click
import csv
import re
from nltk.corpus import wordnet
from more_itertools import peekable


PRED_MAT = 'data/PredicateMatrix.v1.3/PredicateMatrix.v1.3.txt'
PROPBANK_DEFNS = 'data/Finnish_PropBank/gen_lemmas/pb-defs.tsv'
MODEL_RE = re.compile(r".*\(tags: model:([^, \)]+)\).*")

csvout_arg = click.argument("csvout", type=click.File('w'))
pb_defns_arg = click.option('--pb-defns', default=PROPBANK_DEFNS,
                            help='Path to Finnish PropBank pb-defs.tsv file',
                            type=click.File('r'))


def get_writer(csvout):
    csvout = csv.writer(csvout)
    csvout.writerow(['pb', 'wn'])
    return csvout


def get_reader(propbank):
    return csv.DictReader(propbank, delimiter='\t')


def get_eng_pb_wn_map(matrix):
    mapping = {}
    matrix = get_reader(matrix)
    for row in matrix:
        if reject_non_english and row['1_ID_LANG'] != 'id:eng':
            continue
        if row['11_WN_SENSE'] != 'wn:NULL':
            pb = row['16_PB_ROLESET'].split(':', 1)[1]
            wn = row['11_WN_SENSE'].split(':', 1)[1]
            mapping.setdefault(pb, set()).add(wn)
    return mapping


@click.group()
def link():
    pass


@link.command(short_help="Write out lemmas in synset")
@click.argument("csvin", type=click.File('r'))
def dump(csvin):
    """
    Write out the lemmas in each synset from a CSVIN (frame, synset) relation.
    """
    next(csvin)
    for line in csvin:
        frame, ssof = line.strip().split(",", 1)
        ss = wordnet.of2ss(ssof)
        print(frame, " ".join((l.name() for l in ss.lemmas(lang="fin"))))


@link.command(short_help="Get (frame, synset) rel from FinnPropBank")
@pb_defns_arg 
@csvout_arg
def extract_synset_rel(pb_defns, csvout):
    """
    Dumps a (frame, synset) relation to CSVOUT using a the synset_id column from FinnPropBank.
    """
    propbank = get_reader(pb_defns)
    csvout = get_writer(csvout)
    not_found = 0
    found = 0
    for row in propbank:
        pb_finn = "{}.{:0>2}".format(row['base'], row['number'])
        if row['synset_id'] in ("NULL", "666", ""):
            continue
        for raw_synset_id in row['synset_id'].split(","):
            stripped_synset_id = raw_synset_id.strip()
            if not stripped_synset_id:
                continue
            fi_synset_id = stripped_synset_id + "-v"
            try:
                en_synset_id = fi2en_post(fi_synset_id)
            except KeyError:
                print(f"Not found {fi_synset_id} (while processing {pb_finn})", file=sys.stderr)
                not_found += 1
            else:
                csvout.writerow((pb_finn, en_synset_id))
                found += 1
    print(f"Found: {found}; Not found: {not_found}", file=sys.stderr)


@link.command(short_help="Join FinnPropBank with PredicateMatrix to get (frame, synset) rel")
@click.option('--pred-matrix', default=PRED_MAT,
              help='Path to PredicateMatrix.v?.?.txt TSV file',
              type=click.File('r'))
@pb_defns_arg 
@click.option('--reject-non-english/--accept-non-english', default=False,
              help='Accept or reject non-English based mappings')
@click.option('--use-model/--use-link-original', default=True,
              help='Link via the model tag or via the link_original field')
@click.option('--synset/--english-lemmas', default=True,
              help='Map to synset IDs rather than English lemmas')
@csvout_arg
def join_synset(pred_matrix, pb_defns, csvout, reject_non_english, use_model, synset):
    """
    Dumps a (frame, synset) relation to CSVOUT by joining predicate matrix with FinnPropBank.
    """
    # Load mapping from English PropBank senses to English WordNet senses
    mapping = get_eng_pb_wn_map(matrix)

    # Join with mapping from Finnish to English PropBank
    propbank = get_reader(pb_defns)
    csvout = get_writer(csvout)
    for row in propbank:
        pb_finn = "{}.{:0>2}".format(row['base'], row['number'])
        if use_model:
            match = MODEL_RE.match(row['note'])
            if match:
                pb = match.group(1)
            else:
                pb = None
        else:
            pb = row['link_original']
        if pb == 'none.01':
            pb = None
        if pb is not None and pb in mapping:
            for wn in mapping[pb]:
                if synset:
                    csvout.writerow((pb_finn, wordnet.ss2of(wordnet.lemma_from_key(wn + "::").synset())))
                else:
                    csvout.writerow((pb_finn, wn))


if __name__ == "__main__":
    link()
