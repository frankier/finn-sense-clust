.SECONDARY:
.EXPORT_ALL_VARIABLES:

print-%:
	@echo $*=$($*)

SHELL := /bin/bash

all: all-words all-eval

all-words: words/man-words words/all-words-auto words/really-all-words-split/.done

all-eval: eval/frame-synset-union2.filtered2.csv eval/synset-rel.filtered2.csv eval/joined-link.filtered2.csv eval/joined-model.filtered2.csv eval/synth-from-conc.csv eval/synth-from-predmat.csv eval/manclus.csv eval/manclus.wn.csv eval/manclus.wiki.csv eval/manclus.link.csv

eval/:
	mkdir eval
words/:
	mkdir words

results/:
	mkdir results

# Propbank stuff
eval/synset-rel.csv: data/Finnish_PropBank/gen_lemmas/pb-defs.tsv | eval/
	python link.py extract-synset-rel eval/synset-rel.csv

eval/joined-link.csv: data/Finnish_PropBank/gen_lemmas/pb-defs.tsv data/PredicateMatrix.v1.3/PredicateMatrix.v1.3.txt | eval/
	python link.py join-synset --use-link-original eval/joined-link.csv

eval/joined-model.csv: data/Finnish_PropBank/gen_lemmas/pb-defs.tsv data/PredicateMatrix.v1.3/PredicateMatrix.v1.3.txt | eval/
	python link.py join-synset eval/joined-model.csv

eval/%.filtered1.csv: eval/%.csv
	python link.py filter-repeats $< $@

eval/%.filtered2.csv: eval/%.filtered1.csv
	python link.py filter-clus --wn fin --wn qf2 --wn qwf $< $@

eval/frame-synset-union1.csv: eval/synset-rel.filtered1.csv eval/joined-link.filtered1.csv
	python link.py priority-union $^ $@

eval/frame-synset-union2.csv: eval/frame-synset-union1.filtered1.csv eval/joined-model.filtered1.csv
	python link.py priority-union $^ $@

eval/synth-from-conc.csv: eval/frame-synset-union2.csv
	python synth.py from-conc --contradictions path --wn fin --wn qf2 --wn qwf eval/frame-synset-union2.csv eval/synth-from-conc.csv

eval/synth-from-predmat.csv: eval/frame-synset-union2.csv
	python synth.py from-predmat --contradictions path --wn fin --wn qf2 --wn qwf eval/frame-synset-union2.csv eval/synth-from-predmat.csv

words/%.awords: eval/%.csv | words/
	python link.py get-words --pos v $< > $@

words/all-words-auto: words/frame-synset-union2.filtered2.awords words/synset-rel.filtered2.awords words/joined-link.filtered2.awords words/joined-model.filtered2.awords words/synth-from-predmat.awords
	LC_ALL=C sort -u $^ > $@

words/really-all-words:
	python get_all_words.py | sort -u - > words/really-all-words

words/really-all-words-split/.done: words/really-all-words
	mkdir -p words/really-all-words-split
	split -l100 -a4 words/really-all-words words/really-all-words-split/
	touch $@

# Manclus stuff

eval/manclus.csv: manclus/201904/*.Noun manclus/20191202/*.Noun | eval/
	python man_clus.py compile manclus/201904/*.Noun manclus/20191202/*.Noun eval/manclus.csv

eval/manclus.wn.csv: eval/manclus.csv
	python man_clus.py filter --filter wn $< $@

eval/manclus.wiki.csv: eval/manclus.csv
	python man_clus.py filter --filter wiki $< $@

eval/manclus.link.csv: eval/manclus.csv
	python man_clus.py filter --filter link $< $@

words/man-words: eval/manclus.csv | words/
	python link.py get-words --pos n $< > $@

# Experiments ==> Moved to Snakefile

# Clean

.PHONY: clean-eval-words
clean-eval-words:
	rm -r eval words || true

.PHONY: clean-results
clean-results:
	rm -r results || true

.PHONY: rerun-results
rerun-results: clean-results results/label-graph.csv results/vec-clust-autoextend-graph.csv

.PHONY: clean
clean: clean-eval-words clean-results
	rm eval.txt || true
