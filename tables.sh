echo "wordnet table"
poetry run python expc.py sum-table --groups "gold:eval/frame-synset-union2.csv,eval/synth-clus.csv,eval/manclus.wn.csv" work/results/*.db "o,macc"
echo "wiktionary table"
poetry run python expc.py sum-table --groups "gold:eval/manclus.csv,eval/manclus.wiki.csv,eval/manclus.link.csv" work/results/*.db "o,macc"
