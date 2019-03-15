python get_words.py $1 > words
pipenv run python clus.py run-graph-lang words > gold.csv
pipenv run python eval.py $1 gold.csv
