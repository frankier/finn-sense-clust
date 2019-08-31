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
                        "eval/frame-synset-union2.filtered.csv",
                        "eval/synset-rel.filtered.csv",
                        "eval/joined-link.filtered.csv",
                        "eval/joined-model.filtered.csv",
                        "eval/synth-clus.filtered.csv",
                        "eval/manclus.wn.csv",
                    ]), {
                        "eval/frame-synset-union2.filtered.csv": "auto",
                        "eval/synset-rel.filtered.csv": "synset-rel",
                        "eval/joined-link.filtered.csv": "joined-link",
                        "eval/joined-model.filtered.csv": "joined-model",
                        "eval/synth-clus.filtered.csv": "synth",
                        "eval/manclus.wn.csv": "man-wn",
                    }
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
                        "eval/manclus.link.csv",
                    ]), {
                        "eval/manclus.csv": "man",
                        "eval/manclus.wiki.csv": "man-wiki",
                        "eval/manclus.link.csv": "man-link",
                    }
                ),
            ],
            UnlabelledMeasure("o,macc"),
        ),
    )
]
