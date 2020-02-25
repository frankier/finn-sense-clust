from itertools import groupby

from expcomb.table.spec import (
    SumTableSpec,
    CatValGroup,
    LookupGroupDisplay,
    UnlabelledMeasure,
    DimGroups,
    SumDimGroups,
    SortedColsSpec,
    SelectingMeasure,
    box_highlight,
)
from expcomb.filter import SimpleFilter, OrFilter

VECS = LookupGroupDisplay(
    CatValGroup(
        "opts,vec", ["fasttext", "numberbatch", "word2vec", "double", "triple"]
    ),
    {
        "fasttext": "fastText",
        "numberbatch": "Numberbatch",
        "word2vec": "Word2Vec",
        "double": "Concat 2",
        "triple": "Concat 3",
    },
)


MEANS = LookupGroupDisplay(
    CatValGroup(
        "opts,mean",
        [
            "pre_sif_mean",
            # "sif_mean",
            "normalized_mean",
            # "unnormalized_mean",
            "catp3_mean",
            "catp4_mean",
        ],
    ),
    {
        "normalized_mean": "AWE",
        "catp3_mean": "CATP3",
        "catp4_mean": "CATP4",
        "pre_sif_mean": "pre-SIF",
    },
)


FILTER = OrFilter(
    SimpleFilter("SoftCos"),
    SimpleFilter("Wmd"),
    SimpleFilter("SentBert"),
    SimpleFilter("Label"),
    SimpleFilter("SenseVec"),
    SimpleFilter("Comb"),
)

WIKI_FILTER = OrFilter(
    SimpleFilter("SoftCos"),
    SimpleFilter("Wmd"),
    SimpleFilter("SentBert"),
    SimpleFilter("Ety"),
    SimpleFilter("Comb"),
)

ALL_FILTER = OrFilter(
    SimpleFilter("SoftCos"),
    SimpleFilter("Wmd"),
    SimpleFilter("SentBert"),
    SimpleFilter("Label"),
    SimpleFilter("SenseVec"),
    SimpleFilter("Comb"),
    SimpleFilter("Ety"),
)

BOTH_FILTER = OrFilter(
    SimpleFilter("SoftCos"),
    SimpleFilter("Wmd"),
    SimpleFilter("SentBert"),
    SimpleFilter("Comb"),
)

PATH_MAP = {
    "frame-synset-union2.filtered2.csv": "joined",
    "synset-rel.filtered2.csv": "synset",
    "joined-link.filtered2.csv": "link",
    "joined-model.filtered2.csv": "model",
    "synth-clus.csv": "synth",
    "manclus.csv": "man",
    "manclus.wn.csv": "man-wn",
    "manclus.wiki.csv": "man-wiki",
    "manclus.link.csv": "man-link",
}


def fmt(x):
    return "{:.1f}".format(x * 100)


def cnt_fmt(x):
    return ",".join((str(x[k]) for k in ["tp", "fp", "fn", "tn"]))


def real_ultimate_over_sorted_groups(docs):
    def key_doc(doc):
        res = []
        if doc["path"] == ["Comb"]:
            res.extend([True, not doc["opts"]["do_bert"], doc["opts"]["do_wmdsyn"], doc["opts"]["do_wmdpartsyn"], doc["opts"]["do_label"], doc["opts"]["do_ety"], doc["opts"]["do_ety_exemp"]])
        else:
            res.append(False)
        res.append(doc["disp"])
        return res

    return ((k[-1], v) for k, v in groupby(sorted(docs, key=key_doc), key=key_doc))


def safe_float(tpl):
    try:
        return float(tpl[0])
    except ValueError:
        return float("-inf")


TABLES = [
    (
        "wordnet_table",
        SumTableSpec(
            SumDimGroups(),
            DimGroups([
                LookupGroupDisplay(
                    CatValGroup("gold_base", [
                        "frame-synset-union2.filtered2.csv",
                        "synset-rel.filtered2.csv",
                        "joined-link.filtered2.csv",
                        "joined-model.filtered2.csv",
                        "manclus.wn.csv",
                    ]), PATH_MAP
                ),
            ]),
            UnlabelledMeasure("o,macc"),
            fmt,
        ),
        FILTER,
    ),
    (
        "wiktionary_table",
        SumTableSpec(
            SumDimGroups(),
            DimGroups([
                LookupGroupDisplay(
                    CatValGroup("gold_base", [
                        "manclus.csv",
                        "manclus.wiki.csv",
                        "manclus.link.csv",
                    ]), PATH_MAP
                ),
            ]),
            UnlabelledMeasure("o,macc"),
            fmt,
        ),
        WIKI_FILTER,
    ),
    (
        "everything_table",
        SumTableSpec(
            SumDimGroups(),
            DimGroups([
                LookupGroupDisplay(
                    CatValGroup("gold_base", [
                        "frame-synset-union2.filtered2.csv",
                        "synset-rel.filtered2.csv",
                        "joined-link.filtered2.csv",
                        "joined-model.filtered2.csv",
                        "manclus.csv",
                        "manclus.wn.csv",
                        "manclus.wiki.csv",
                        "manclus.link.csv",
                    ]), PATH_MAP
                ),
            ]),
            UnlabelledMeasure("o,macc"),
            fmt,
        ),
        ALL_FILTER,
    ),
    (
        "everything_f1",
        SumTableSpec(
            SumDimGroups(),
            DimGroups([
                LookupGroupDisplay(
                    CatValGroup("gold_base", [
                        "frame-synset-union2.filtered2.csv",
                        "synset-rel.filtered2.csv",
                        "joined-link.filtered2.csv",
                        "joined-model.filtered2.csv",
                        "synth-clus.csv",
                        "manclus.csv",
                        "manclus.wn.csv",
                        "manclus.wiki.csv",
                        "manclus.link.csv",
                    ]), PATH_MAP
                ),
            ]),
            UnlabelledMeasure("pr,f1"),
            fmt,
        ),
        ALL_FILTER,
    ),
    (
        "everything_rand",
        SumTableSpec(
            SumDimGroups(),
            DimGroups([
                LookupGroupDisplay(
                    CatValGroup("gold_base", [
                        "frame-synset-union2.filtered2.csv",
                        "synset-rel.filtered2.csv",
                        "joined-link.filtered2.csv",
                        "joined-model.filtered2.csv",
                        "synth-clus.csv",
                        "manclus.csv",
                        "manclus.wn.csv",
                        "manclus.wiki.csv",
                        "manclus.link.csv",
                    ]), PATH_MAP
                ),
            ]),
            UnlabelledMeasure("o,rand"),
            fmt,
        ),
        ALL_FILTER,
    ),
    (
        "everything_mat",
        SumTableSpec(
            SumDimGroups(),
            DimGroups([
                LookupGroupDisplay(
                    CatValGroup("gold_base", [
                        "frame-synset-union2.filtered2.csv",
                        "synset-rel.filtered2.csv",
                        "joined-link.filtered2.csv",
                        "joined-model.filtered2.csv",
                        "synth-clus.csv",
                        "manclus.csv",
                        "manclus.wn.csv",
                        "manclus.wiki.csv",
                        "manclus.link.csv",
                    ]), PATH_MAP
                ),
            ]),
            UnlabelledMeasure("cnt"),
            cnt_fmt,
        ),
        ALL_FILTER,
    ),
    (
        "over_table",
        SortedColsSpec(
            SumDimGroups(),
            DimGroups([
                LookupGroupDisplay(
                    CatValGroup("gold_base", [
                        "frame-synset-union2.filtered2.csv",
                        "synth-clus.csv",
                        "manclus.csv",
                    ]), PATH_MAP
                ),
            ]),
            SelectingMeasure(
                (SimpleFilter(gold_base="synth-clus.csv"), UnlabelledMeasure("o,rand")),
                UnlabelledMeasure("o,macc"),
            ),
            fmt,
            lambda col: col.sort(
                reverse=True,
                key=safe_float,
            ),
        ),
        BOTH_FILTER,
    ),
    (
        "real_ultimate_over",
        SumTableSpec(
            SumDimGroups(real_ultimate_over_sorted_groups),
            DimGroups([
                LookupGroupDisplay(
                    CatValGroup("gold_base", [
                        "frame-synset-union2.filtered2.csv",
                        "synset-rel.filtered2.csv",
                        "joined-link.filtered2.csv",
                        "joined-model.filtered2.csv",
                        "synth-clus.csv",
                        "manclus.csv",
                        "manclus.wn.csv",
                        "manclus.wiki.csv",
                        "manclus.link.csv",
                    ]), PATH_MAP
                ),
            ]),
            SelectingMeasure(
                (SimpleFilter(gold_base="synth-clus.csv"), UnlabelledMeasure("o,rand")),
                UnlabelledMeasure("o,macc"),
            ),
            fmt,
            box_highlight,
        ),
        ALL_FILTER,
    ),
]
