from expcomb.table.spec import (
    SumTableSpec,
    CatValGroup,
    LookupGroupDisplay,
    UnlabelledMeasure,
    MeasuresSplit,
    DimGroups,
    SumDimGroups,
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


TABLES = [
    (
        "wordnet_table",
        SumTableSpec(
            SumDimGroups(two_levels=False),
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
            SumDimGroups(two_levels=False),
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
            SumDimGroups(two_levels=False),
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
            SumDimGroups(two_levels=False),
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
            SumDimGroups(two_levels=False),
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
            SumDimGroups(two_levels=False),
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
]
