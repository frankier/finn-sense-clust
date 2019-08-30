from expcomb.table.spec import (
    SumTableSpec,
    CatValGroup,
    LookupGroupDisplay,
    UnlabelledMeasure,
    MeasuresSplit,
)


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


TABLES = [
    (
        "wordnet_table",
        SumTableSpec(
            [
                LookupGroupDisplay(
                    CatValGroup("gold", [
                        "eval/frame-synset-union2.csv",
                        "eval/synth-clus.csv",
                        "eval/manclus.wn.csv"
                    ]),
                ),
            ],
            UnlabelledMeasure("o,macc"),
        ),
    ),
    (
        "wiktionary_table",
        SumTableSpec(
            [
                LookupGroupDisplay(
                    CatValGroup("gold", [
                        "eval/manclus.csv",
                        "eval/manclus.wiki.csv",
                        "eval/manclus.link.csv"
                    ]),
                ),
            ],
            UnlabelledMeasure("o,macc"),
        ),
    )
]
