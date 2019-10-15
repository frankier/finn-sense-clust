SHELL := /bin/bash

all: all-words all-eval

all-words: words/man-words words/all-words words/really-all-words-split/.done

all-eval: eval/frame-synset-union2.csv eval/synth-clus.csv eval/manclus.csv eval/manclus.wn.csv eval/manclus.wiki.csv eval/manclus.link.csv

eval/:
	mkdir eval
words/:
	mkdir words

results/:
	mkdir results

# Propbank stuff
eval/synset-rel.csv: data/Finnish_PropBank/gen_lemmas/pb-defs.tsv | eval/
	poetry run python link.py extract-synset-rel eval/synset-rel.csv

eval/joined-link.csv: data/Finnish_PropBank/gen_lemmas/pb-defs.tsv data/PredicateMatrix.v1.3/PredicateMatrix.v1.3.txt | eval/
	poetry run python link.py join-synset --use-link-original eval/joined-link.csv

eval/joined-model.csv: data/Finnish_PropBank/gen_lemmas/pb-defs.tsv data/PredicateMatrix.v1.3/PredicateMatrix.v1.3.txt | eval/
	poetry run python link.py join-synset eval/joined-model.csv

eval/synset-rel.filtered.csv: eval/synset-rel.csv 
	poetry run python link.py filter-repeats eval/synset-rel.csv eval/synset-rel.filtered.csv

eval/joined-link.filtered.csv: eval/joined-link.csv 
	poetry run python link.py filter-repeats eval/joined-link.csv eval/joined-link.filtered.csv

eval/joined-model.filtered.csv: eval/joined-model.csv 
	poetry run python link.py filter-repeats eval/joined-model.csv eval/joined-model.filtered.csv

eval/frame-synset-union1.csv: eval/synset-rel.filtered.csv eval/joined-link.filtered.csv 
	poetry run python link.py priority-union eval/synset-rel.filtered.csv eval/joined-link.filtered.csv eval/frame-synset-union1.csv

eval/frame-synset-union2.csv: eval/frame-synset-union1.csv eval/joined-model.filtered.csv 
	poetry run python link.py priority-union eval/frame-synset-union1.csv eval/joined-model.filtered.csv eval/frame-synset-union2.csv

eval/synth-clus.csv: eval/frame-synset-union2.csv
	poetry run python link.py synth-clus --wn fin --wn qf2 --wn qwf eval/frame-synset-union2.csv eval/synth-clus.csv

words/conc-words: eval/frame-synset-union2.csv | words/
	poetry run python link.py get-words --pos v --wn fin --wn qf2 --wn qwf --filter smap2 eval/frame-synset-union2.csv > words/conc-words

words/synth-words: eval/synth-clus.csv | words/
	poetry run python link.py get-words --pos v --wn fin --wn qf2 --wn qwf --filter smap2 --multi-group eval/synth-clus.csv > words/synth-words

words/all-words: words/conc-words words/synth-words
	LC_ALL=C sort -u words/conc-words words/synth-words > words/all-words

words/really-all-words:
	poetry run python get_all_words.py | sort -u - > words/really-all-words

words/really-all-words-split/.done: words/really-all-words
	mkdir -p words/really-all-words-split
	split -l100 -a4 words/really-all-words words/really-all-words-split/
	touch $@

# Manclus stuff

eval/manclus.csv: manclus/*.Noun | eval/
	poetry run python man_clus.py compile manclus/* eval/manclus.csv

eval/manclus.wn.csv: eval/manclus.csv
	poetry run python man_clus.py filter --filter wn $< $@

eval/manclus.wiki.csv: eval/manclus.csv
	poetry run python man_clus.py filter --filter wiki $< $@

eval/manclus.link.csv: eval/manclus.csv
	poetry run python man_clus.py filter --filter link $< $@

words/man-words: eval/manclus.csv | words/
	poetry run python link.py get-words --pos n --filter none eval/manclus.csv > words/man-words

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
