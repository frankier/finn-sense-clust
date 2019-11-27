## Eval
from expcomb.filter import SimpleFilter, parse_filter
from expc import SnakeMake

def merge(r1, r2):
    return {
        k: r1.get(k, []) + r2.get(k, []) for k in {*r1.keys(), *r2.keys()}
    }

FILTER = config.setdefault("FILTER", "")
EVAL = config.setdefault("EVAL", "eval")
WORDS = config.setdefault("WORDS", "words")
SEED = "42"
ITERS = "100000"

WORK = "work"
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
ALL_EVAL = merge(merge(WN_ONLY_EVAL, WIKI_ONLY_EVAL), BOTH_EVAL)
MULTI_EVAL = {"synth-clus", "manclus.link"}

def all_results():
    filter = parse_filter(FILTER)
    def eval_paths(nick, eval_dict):
        for corpus, evals in eval_dict.items():
            for eval in evals:
                yield f"{WORK}/results/{corpus}--{eval}--{nick}.db"
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
    output: WORK + "/guess/{corpus}.{nick}"
    wildcard_constraints:
        corpus=r"[^\.]+"
    shell:
        "python expc.py --filter 'nick={wildcards.nick}' test " + WORDS + "/{wildcards.corpus} {output}"

rule eval:
    input: WORK + "/guess/{corpus}.{nick}"
    output: WORK + "/results/{corpus}--{eval}--{nick}.db"
    run:
        multi = "--multi" if wildcards.eval in MULTI_EVAL else "--single"
        shell("python expc.py --filter 'nick={wildcards.nick}' eval " + multi + " {output} " + WORDS + "/{wildcards.corpus} " + WORK + "/guess " + EVAL + "/{wildcards.eval}.csv")

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

def cmp_inputs(wildcards):
    if wildcards.corpus in WIKI_EXTRA_EVAL:
        nicks = SnakeMake.get_nicks(SimpleFilter(supports_wiktionary=True))
    else:
        nicks = SnakeMake.get_nicks()
    return expand(WORK + "/bootstrap/resamples/{nick}/" + wildcards.corpus + "/" + wildcards.eval + ".pkl", nick=nicks)


rule bootstrap:
    input: [WORK + f"/bootstrap/cld/{corpus}/{eval}.db" for corpus, evals in ALL_EVAL.items() for eval in evals]

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
        guess=WORK + "/guess/{corpus}.{nick}",
        result=WORK + "/results/{corpus}--{eval}--{nick}.db",
        schedule=WORK + "/bootstrap/schedules/{corpus}.pkl",
    output: WORK + "/bootstrap/resamples/{nick,[^/]+}/{corpus,[^/]+}/{eval,[^/]+}.pkl",
    run:
        multi = "--multi" if wildcards.eval in MULTI_EVAL else "--single"
        shell(
            "python expc.py sigtest resample " + multi +
	    " {output} {input.eval} {input.guess} {input.result} {input.schedule} o,macc"
        )

rule compare:
    input: cmp_inputs
    output: WORK + "/bootstrap/cmp/{corpus,[^/]+}/{eval,[^/]+}.db"
    shell:
        "python expc.py sigtest compare-resampled {input} {output}"

rule cld:
    input: WORK + "/bootstrap/cmp/{corpus,[^/]+}/{eval,[^/]+}.db"
    output: WORK + "/bootstrap/cld/{corpus,[^/]+}/{eval,[^/]+}.db"
    shell:
        "python expc.py sigtest cld {input} {output}"
