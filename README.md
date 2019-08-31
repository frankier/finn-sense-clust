# finn-sense-clust

This repo contains some experiments for clustering FinnWordNet lemmas.

Install::

    $ ./install.sh

Make the evaluation materials::

    $ DATABASE_URL=sqlite:///path/to/db.db make

Run the experiments::

    $ DATABASE_URL=sqlite:///path/to/db.db snakemake

Print the tables::

    $ poetry run python expc.py all-tables work

Run it for all words::

    $ DATABASE_URL=sqlite:///path/to/db.db poetry run snakemake -j4 gloss_all
