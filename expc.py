import click
from expcomb.cmd import mk_expcomb
from senseclust.methods.base import ExpPathInfo
from senseclust.methods import EXPERIMENTS


expc = mk_expcomb(EXPERIMENTS)


@expc.mk_test
@click.argument("test_corpus", type=click.Path())
@click.argument("guess_dir", type=click.Path())
@click.argument("gold", type=click.Path())
def test(
    test_corpus,
    guess_dir,
    gold,
):
    return ExpPathInfo(test_corpus, guess_dir, gold)


if __name__ == "__main__":
    expc()
