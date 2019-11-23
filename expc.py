import click
from os.path import join as pjoin
from expcomb.cmd import mk_expcomb
from expcomb.table.utils import pick_str
from expcomb.utils import TinyDBParam, mk_iden
from expcomb.sigtest.bootstrap import Bootstrapper, mk_resample, mk_compare_resampled, simple_create_schedule, simple_compare_resampled
from senseclust.methods.base import ExpPathInfo
from senseclust.methods import EXPERIMENTS
from senseclust.eval import eval as eval_func, gen_gold_groupings, pre_cnt_lines, eval_resampled, UnguessedException
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
    def __init__(self, multi, measure=None):
        self.multi = multi
        self.measure = measure

    def score_one(self, gold, guess):
        result = eval_func(open(gold), open(guess), self.multi)
        if self.measure is not None:
            result = pick_str(result, self.measure)
        return result

    def create_score_dist(self, gold, guess, schedule):
        dist = []
        cnt_map = pre_cnt_lines(open(gold), open(guess), self.multi)
        for resample in schedule:
            result = eval_resampled(resample, cnt_map)
            if self.measure is not None:
                result = pick_str(result, self.measure)
            dist.append(result)
        return dist


@mk_resample
@click.argument("outf", type=click.File("wb"))
@click.argument("gold", type=click.Path())
@click.argument("guess", type=click.Path())
@click.argument("result", type=TinyDBParam())
@click.argument("schedule", type=click.File("rb"))
@click.argument("measure", required=False)
@click.option("--multi/--single")
def resample(outf, gold, guess, result, schedule, measure=None, multi=False):
    return SenseClustBootstrapper(multi, measure), outf, gold, guess, result, schedule, None


simple_compare_resampled()
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
    from expcomb.score import proc_score
    iden = mk_iden(corpus, exp)
    guess_path = pjoin(guess_dir, iden)
    try:
        measures = mk_eval(multi)(gold, guess_path)
    except UnguessedException as exc:
        exc.gold_fn = gold
        exc.guess_fn = guess_path
        raise
    proc_score(exp, db, measures, guess_dir, gold)


if __name__ == "__main__":
    expc()
