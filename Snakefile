## Eval
from os.path import join as pjoin
from expcomb.filter import SimpleFilter, parse_filter
from expc import SnakeMake

FILTER = config.setdefault("FILTER", "")
EVAL = config.setdefault("EVAL", "eval")
WORDS = config.setdefault("WORDS", "words")
WORK = config.setdefault("WORK", "work")
GUESS = config.setdefault("GUESS", pjoin(WORK, "guess"))
RESULTS = config.setdefault("RESULTS", pjoin(WORK, "results"))

SEED = "42"
ITERS = "100000"

WN_ONLY_EVAL = {
    "all-words": ["synth-clus", "frame-synset-union2.filtered2", "synset-rel.filtered2", "joined-link.filtered2", "joined-model.filtered2"],
    "man-words": ["manclus.wn"],
}
WIKI_ONLY_EVAL = {
    "man-words": [
        "manclus.wiki",
    ]
}
BOTH_EVAL = {
    "man-words": [
        "manclus",
        "manclus.link",
    ]
}
MULTI_EVAL = {"synth-clus", "manclus.link"}

def all_results():
    filter = parse_filter(FILTER)
    def eval_paths(nick, eval_dict):
        for corpus, evals in eval_dict.items():
            for eval in evals:
                yield f"{RESULTS}/{corpus}--{eval}--{nick}.db"
    for nick in SnakeMake.intersect_nicks(filter, supports_wordnet=True):
        yield from eval_paths(nick, WN_ONLY_EVAL)
    for nick in SnakeMake.intersect_nicks(filter, supports_wiktionary=True):
        yield from eval_paths(nick, WIKI_ONLY_EVAL)
    for nick in SnakeMake.intersect_nicks(filter, supports_wordnet=True, supports_wiktionary=True):
        yield from eval_paths(nick, BOTH_EVAL)


rule all:
    input: list(all_results())

rule test:
    input: WORDS + "/{corpus}"
    output: GUESS + "/{corpus}.{nick}"
    wildcard_constraints:
        corpus=r"[^\.]+"
    shell:
        "python expc.py --filter 'nick={wildcards.nick}' test " + WORDS + "/{wildcards.corpus} {output}"

rule eval:
    input: GUESS + "/{corpus}.{nick}"
    output: RESULTS + "/{corpus}--{eval}--{nick}.db"
    run:
        multi = "--multi" if wildcards.eval in MULTI_EVAL else "--single"
        shell("python expc.py --filter 'nick={wildcards.nick}' eval " + multi + " {output} " + WORDS + "/{wildcards.corpus} " + GUESS + " " + EVAL + "/{wildcards.eval}.csv")

# Final output

rule run_gloss:
    input: WORDS + "/really-all-words-split/{seg}"
    output: WORK + "/output/{seg}.csv"
    shell:
        "python expc.py --filter 'nick=gloss' test --exemplars {input} {output}"

SEGS = glob_wildcards(WORDS + "/really-all-words-split/{seg}")[0]

rule gloss_all:
    input: expand(WORK + "/output/{seg}.csv", seg=SEGS)

# Bootstrapping
BOOTSTRAPS = [
    ("conc-words", "frame-synset-union2.filtered2", "macc"),
    ("synth-words", "synth-clus", "rand"),
    ("man-words", "manclus", "macc"),
]

BS_CORPUS_MAP = {
    "conc-words": "all-words",
    "synth-words": "all-words",
    "man-words": "man-words",
}

def cmp_inputs(wildcards):
    filter = parse_filter(FILTER)
    nicks = SnakeMake.intersect_nicks(filter, supports_wordnet=True, supports_wiktionary=True)
    return expand(WORK + "/bootstrap/resamples/{nick}/" + wildcards.corpus + "/" + wildcards.eval + "/" + wildcards.measure + ".pkl", nick=nicks)


rule bootstrap:
    input: [WORK + f"/bootstrap/cld/{corpus}/{eval}/{measure}.db" for corpus, eval, measure in BOOTSTRAPS]

rule create_schedule:
    input: WORDS + "/{corpus}"
    output: WORK + "/bootstrap/schedules/{corpus}.pkl"
    shell:
        "python expc.py sigtest create-schedule" +
        " --seed " + SEED +
        " --iters " + ITERS +
        " {input} {output}"

rule resample:
    input:
        eval=EVAL + "/{eval}.csv",
        guess=lambda wc: f"{GUESS}/{BS_CORPUS_MAP[wc.corpus]}.{wc.nick}",
        result=lambda wc: f"{RESULTS}/{BS_CORPUS_MAP[wc.corpus]}--{wc.eval}--{wc.nick}.db",
        schedule=WORK + "/bootstrap/schedules/{corpus}.pkl",
    output: WORK + "/bootstrap/resamples/{nick,[^/]+}/{corpus,[^/]+}/{eval,[^/]+}/{measure,[^/]+}.pkl",
    run:
        multi = "--multi" if wildcards.eval in MULTI_EVAL else "--single"
        shell(
            "python expc.py sigtest resample " + multi +
	    " {output} {input.eval} {input.guess} {input.result} {input.schedule} o,{wildcards.measure}"
        )

rule compare:
    input: cmp_inputs
    output: WORK + "/bootstrap/cmp/{corpus,[^/]+}/{eval,[^/]+}/{measure,[^/]+}.db"
    shell:
        "python expc.py sigtest compare-resampled {input} {output}"

rule cld:
    input: WORK + "/bootstrap/cmp/{corpus,[^/]+}/{eval,[^/]+}/{measure,[^/]+}.db"
    output: WORK + "/bootstrap/cld/{corpus,[^/]+}/{eval,[^/]+}/{measure,[^/]+}.db"
    shell:
        "python expc.py sigtest cld {input} {output}"
