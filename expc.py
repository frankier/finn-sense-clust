import click
from expcomb.cmd import mk_expcomb, TinyDBParam
from senseclust.methods.base import ExpPathInfo
from senseclust.methods import EXPERIMENTS
from senseclust.eval import eval as eval_func


def mk_eval(multi):
    def eval(gold, test):
        with open(gold) as gold_f, open(test) as test_f:
            return eval_func(gold_f, test_f, multi)
    return eval


expc, SnakeMake = mk_expcomb(EXPERIMENTS, eval)


@expc.group_apply_cmd
@click.argument("test_corpus", type=click.Path())
@click.argument("guess", type=click.Path())
@click.option("--exemplars/--no-exemplar")
def test(
    experiments,
    test_corpus,
    guess,
    exemplars=False,
):
    path_info = ExpPathInfo(test_corpus, guess, None)
    for exp_group in experiments:
        if exemplars:
            exp_group.run_all(path_info, exemplars=True)
        else:
            exp_group.run_all(path_info)


@expc.exp_apply_cmd
@click.argument("db", type=TinyDBParam())
@click.argument("corpus")
@click.argument("guess_dir", type=click.Path())
@click.argument("gold", type=click.Path())
@click.option("--multi/--single")
def eval(exp, db, corpus, guess_dir, gold, multi=False):
    from expcomb.score import calc_exp_score, proc_score
    measures = calc_exp_score(exp, corpus, gold, guess_dir, mk_eval(multi))
    proc_score(exp, db, measures, gold)


if __name__ == "__main__":
    expc()
