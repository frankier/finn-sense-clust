from expcomb.table.spec import (
    SumTableSpec,
    CatValGroup,
    LookupGroupDisplay,
    UnlabelledMeasure,
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
    SimpleFilter("Bert"),
    SimpleFilter("Label"),
    SimpleFilter("SenseVec"),
    SimpleFilter("Comb"),
)

WIKI_FILTER = OrFilter(
    SimpleFilter("SoftCos"),
    SimpleFilter("Wmd"),
    SimpleFilter("Bert"),
    SimpleFilter("Ety"),
    SimpleFilter("Comb"),
)

ALL_FILTER = OrFilter(
    SimpleFilter("SoftCos"),
    SimpleFilter("Wmd"),
    SimpleFilter("Bert"),
    SimpleFilter("Label"),
    SimpleFilter("SenseVec"),
    SimpleFilter("Comb"),
    SimpleFilter("Ety"),
)

PATH_MAP = {
    "eval/frame-synset-union2.csv": "joined",
    "eval/synset-rel.filtered.csv": "synset",
    "eval/joined-link.filtered.csv": "link",
    "eval/joined-model.filtered.csv": "model",
    "eval/synth-clus.csv": "synth",
    "eval/manclus.csv": "man",
    "eval/manclus.wn.csv": "man-wn",
    "eval/manclus.wiki.csv": "man-wiki",
    "eval/manclus.link.csv": "man-link",
}


def fmt(x):
    return "{:.1f}".format(x * 100)


TABLES = [
    (
        "wordnet_table",
        SumTableSpec(
            SumDimGroups(two_levels=False),
            DimGroups([
                LookupGroupDisplay(
                    CatValGroup("gold", [
                        "eval/frame-synset-union2.csv",
                        "eval/synset-rel.filtered.csv",
                        "eval/joined-link.filtered.csv",
                        "eval/joined-model.filtered.csv",
                        "eval/manclus.wn.csv",
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
                    CatValGroup("gold", [
                        "eval/manclus.csv",
                        "eval/manclus.wiki.csv",
                        "eval/manclus.link.csv",
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
                    CatValGroup("gold", [
                        "eval/frame-synset-union2.csv",
                        "eval/synset-rel.filtered.csv",
                        "eval/joined-link.filtered.csv",
                        "eval/joined-model.filtered.csv",
                        "eval/manclus.csv",
                        "eval/manclus.wn.csv",
                        "eval/manclus.wiki.csv",
                        "eval/manclus.link.csv",
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
                    CatValGroup("gold", [
                        "eval/frame-synset-union2.csv",
                        "eval/synset-rel.filtered.csv",
                        "eval/joined-link.filtered.csv",
                        "eval/joined-model.filtered.csv",
                        "eval/manclus.csv",
                        "eval/manclus.wn.csv",
                        "eval/manclus.wiki.csv",
                        "eval/manclus.link.csv",
                    ]), PATH_MAP
                ),
            ]),
            UnlabelledMeasure("pr,f1"),
            fmt,
        ),
        ALL_FILTER ,
    ),
]
