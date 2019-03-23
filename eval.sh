for test in results/*
do
  echo "Concrete gold evaluation of $test"
  pipenv run python eval.py eval/frame-synset-union2.csv $test
  echo "Synthetic gold evaluation of $test"
  pipenv run python eval.py --multi-group eval/synth-clus.csv $test
done
