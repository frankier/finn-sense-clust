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
    SimpleFilter("Gloss"),
    SimpleFilter("Label"),
    SimpleFilter("Sense vector"),
)

WIKI_FILTER = OrFilter(
    SimpleFilter("Gloss"),
)


def fmt(x):
    return "{:.3f}".format(x)


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
                        "eval/synth-clus.csv",
                        "eval/manclus.wn.csv",
                    ]), {
                        "eval/frame-synset-union2.csv": "auto",
                        "eval/synset-rel.filtered.csv": "synset-rel",
                        "eval/joined-link.filtered.csv": "joined-link",
                        "eval/joined-model.filtered.csv": "joined-model",
                        "eval/synth-clus.csv": "synth",
                        "eval/manclus.wn.csv": "man-wn",
                    }
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
                    ]), {
                        "eval/manclus.csv": "man",
                        "eval/manclus.wiki.csv": "man-wiki",
                        "eval/manclus.link.csv": "man-link",
                    }
                ),
            ]),
            UnlabelledMeasure("o,macc"),
            fmt,
        ),
        WIKI_FILTER,
    )
]
