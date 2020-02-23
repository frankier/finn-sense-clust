import click
import csv
from senseclust.wordnet import WORDNETS

PROPBANK_DEFNS = 'data/Finnish_PropBank/gen_lemmas/pb-defs.tsv'
PRED_MAT = 'data/PredicateMatrix.v1.3/PredicateMatrix.v1.3.txt'

csvin_arg = click.argument("csvin", type=click.File('r'))
csvout_arg = click.argument("csvout", type=click.File('w'))
pb_defns_arg = click.option('--pb-defns', default=PROPBANK_DEFNS,
                            help='Path to Finnish PropBank pb-defs.tsv file',
                            type=click.File('r'))
wns_arg = click.option('--wn', type=click.Choice(WORDNETS),
                       default=['fin'], multiple=True,
                       help='Which WordNet (multiple allowed) to use: OMW FiWN, '
                       'FiWN2 or OMW FiWN wikitionary based extensions')
predmat_arg = click.option('--pred-matrix', default=PRED_MAT,
                           help='Path to PredicateMatrix.v?.?.txt TSV file',
                           type=click.File('r'))


def get_writer(csvout):
    csvout = csv.writer(csvout)
    csvout.writerow(['pb', 'wn'])
    return csvout


def get_reader(propbank):
    return csv.DictReader(propbank, delimiter='\t')


def get_eng_pb_wn_pairs(matrix, reject_non_english):
    matrix = get_reader(matrix)
    for row in matrix:
        if reject_non_english and row['1_ID_LANG'] != 'id:eng':
            continue
        if row['11_WN_SENSE'] != 'wn:NULL':
            pb = row['16_PB_ROLESET'].split(':', 1)[1]
            wn = row['11_WN_SENSE'].split(':', 1)[1]
            yield pb, wn


def get_eng_pb_wn_map(matrix, reject_non_english):
    mapping = {}
    for pb, wn in get_eng_pb_wn_pairs(matrix, reject_non_english):
        mapping.setdefault(pb, set()).add(wn)
    return mapping


def lemma_id_to_synset_id(lemma_id):
    from nltk.corpus import wordnet
    return wordnet.ss2of(wordnet.lemma_from_key(lemma_id + "::").synset())
