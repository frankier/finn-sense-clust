from expc import SnakeMake

WORK = "work"
WN_ONLY_EVAL = {
    "all-words": ["synth-clus", "frame-synset-union2"],
    "man-words": ["manclus_wn"],
}
WIKI_EXTRA_EVAL = {
    "man-words": [
        "manclus",
        "manclus_wiki",
        "manclus_link",
    ]
}
MULTI_EVAL = {"synth-clus", "manclus_link"}

def all_results():
    def eval_paths(nick, eval_dict):
        for corpus, evals in eval_dict.items():
            for eval in evals:
                yield f"{WORK}/results/{corpus}--{eval}--{nick}.db"
    for nick in SnakeMake.get_nicks():
        yield from eval_paths(nick, WN_ONLY_EVAL)
    for nick in SnakeMake.get_nicks(opt_dict={"supports_wiktionary": True}):
        yield from eval_paths(nick, WIKI_EXTRA_EVAL)

rule all:
    input: list(all_results())

rule test:
    input: "words/{corpus}"
    output: WORK + "/guess/{corpus}.{nick}"
    wildcard_constraints:
        corpus=r"[^\.]+"
    shell:
        "python expc.py --filter 'nick={wildcards.nick}' test words/{wildcards.corpus} " + WORK + "/guess/ eval/manclus.csv"

rule eval:
    input: WORK + "/guess/{corpus}.{nick}"
    output: WORK + "/results/{corpus}--{eval}--{nick}.db"
    run:
        clean_eval = wildcards.eval.replace("_", ".")
	multi = "--multi" if wildcards.eval in MULTI_EVAL else "--single"
        shell("python expc.py --filter 'nick={wildcards.nick}' eval " + multi + " {output} words/{wildcards.corpus} " + WORK + "/guess/ eval/" + clean_eval + ".csv")
