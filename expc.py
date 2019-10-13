import click
from expcomb.cmd import mk_expcomb
from expcomb.utils import TinyDBParam
from expcomb.sigtest.bootstrap import Bootstrapper, mk_resample, mk_compare_resampled, simple_create_schedule
from senseclust.methods.base import ExpPathInfo
from senseclust.methods import EXPERIMENTS
from senseclust.eval import eval as eval_func, gen_gold_groupings, pre_cnt_lines, eval_resampled
from senseclust.evaltables import TABLES
from senseclust.groupings import gen_groupings


def mk_get_score(multi):
    def get_score(gold, test):
        return eval_func(gold, test, multi)

    return get_score


def create_sample_maps(multi, gold, guess):
    line_map = []
    gold_map = {}
    for lemma, groups in gen_gold_groupings(gold, multi):
        line_map.append(lemma)
        gold_map[lemma] = groups
    guess_map = {}
    for lemma, groups in gen_groupings(guess):
        guess_map[lemma] = groups
    return line_map, guess_map, gold_map


class SenseClustBootstrapper(Bootstrapper):
    def __init__(self, multi):
        self.multi = multi

    def score_one(self, gold, guess):
        return eval_func(open(gold), open(guess), self.multi)

    def create_score_dist(self, gold, guess, schedule):
        dist = []
        cnt_map = pre_cnt_lines(open(gold), open(guess), self.multi)
        for resample in schedule:
            dist.append(eval_resampled(resample, cnt_map))
        return dist


@mk_resample
@click.argument("outf", type=click.File("wb"))
@click.argument("gold", type=click.Path())
@click.argument("guess", type=click.Path())
@click.argument("result", type=TinyDBParam())
@click.argument("schedule", type=click.File("rb"))
@click.option("--multi/--single")
def resample(outf, gold, guess, result, schedule, multi):
    return SenseClustBootstrapper(multi), outf, gold, guess, result, schedule, None


@mk_compare_resampled
@click.argument("inf", type=click.File("rb"))
@click.argument("outf", type=click.File("wb"))
@click.option("--multi/--single")
def compare_resampled(inf, outf, multi):
    return SenseClustBootstrapper(multi), inf, outf


single_bootstrapper = SenseClustBootstrapper(False)
simple_create_schedule(single_bootstrapper)


def mk_eval(multi):
    def eval(gold, test):
        with open(gold) as gold_f, open(test) as test_f:
            return eval_func(gold_f, test_f, multi)
    return eval


expc, SnakeMake = mk_expcomb(EXPERIMENTS, eval, tables=TABLES)


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
    proc_score(exp, db, measures, guess_dir, gold)


if __name__ == "__main__":
    expc()
