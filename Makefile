SHELL := /bin/bash

all: eval/frame-synset-union2.csv eval/synth-clus.csv words/all-words eval.txt

eval/:
	mkdir eval

words/:
	mkdir words

results/:
	mkdir results

eval/synset-rel.csv: data/Finnish_PropBank/gen_lemmas/pb-defs.tsv | eval/
	pipenv run python link.py extract-synset-rel eval/synset-rel.csv

eval/joined-link.csv: data/Finnish_PropBank/gen_lemmas/pb-defs.tsv data/PredicateMatrix.v1.3/PredicateMatrix.v1.3.txt | eval/
	pipenv run python link.py join-synset --use-link-original eval/joined-link.csv

eval/joined-model.csv: data/Finnish_PropBank/gen_lemmas/pb-defs.tsv data/PredicateMatrix.v1.3/PredicateMatrix.v1.3.txt | eval/
	pipenv run python link.py join-synset eval/joined-model.csv

eval/synset-rel.filtered.csv: eval/synset-rel.csv 
	pipenv run python link.py filter-repeats eval/synset-rel.csv eval/synset-rel.filtered.csv

eval/joined-link.filtered.csv: eval/joined-link.csv 
	pipenv run python link.py filter-repeats eval/joined-link.csv eval/joined-link.filtered.csv

eval/joined-model.filtered.csv: eval/joined-model.csv 
	pipenv run python link.py filter-repeats eval/joined-model.csv eval/joined-model.filtered.csv

eval/frame-synset-union1.csv: eval/synset-rel.filtered.csv eval/joined-link.filtered.csv 
	pipenv run python link.py priority-union eval/synset-rel.filtered.csv eval/joined-link.filtered.csv eval/frame-synset-union1.csv

eval/frame-synset-union2.csv: eval/frame-synset-union1.csv eval/joined-model.filtered.csv 
	pipenv run python link.py priority-union eval/frame-synset-union1.csv eval/joined-model.filtered.csv eval/frame-synset-union2.csv

eval/synth-clus.csv: eval/frame-synset-union2.csv
	pipenv run python link.py synth-clus --wn fin --wn qf2 --wn qwf eval/frame-synset-union2.csv eval/synth-clus.csv

words/conc-words: eval/frame-synset-union2.csv | words/
	pipenv run python link.py get-words --wn fin --wn qf2 --wn qwf --filter smap2 eval/frame-synset-union2.csv > words/conc-words

words/synth-words: eval/synth-clus.csv | words/
	pipenv run python link.py get-words --wn fin --wn qf2 --wn qwf --filter smap2 --multi-group eval/synth-clus.csv > words/synth-words

words/all-words: words/conc-words words/synth-words
	LC_ALL=C sort -u words/conc-words words/synth-words > words/all-words

results/label-graph.csv: results/ words/all-words
	pipenv run python clus.py run label-graph words/all-words > results/label-graph.csv

results/vec-clust-autoextend-graph.csv: results/ words/all-words
	pipenv run python clus.py run vec-clust-autoextend-graph words/all-words > results/vec-clust-autoextend-graph.csv

eval.txt: results/label-graph.csv results/vec-clust-autoextend-graph.csv
	bash eval.sh > eval.txt

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
