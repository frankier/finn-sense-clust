import click
from senseclust.eval import eval as eval_func


@click.command()
@click.argument("gold", type=click.File('r'))
@click.argument("test", type=click.File('r'))
@click.option('--multi-group/--single-group')
def eval(gold, test, multi_group):
    measures = eval_func(gold, test, multi_group)
    lemmas = measures["lemmas"]
    cnt = measures["cnt"]
    p = measures["pr"]["p"]
    r = measures["pr"]["r"]
    f1 = measures["pr"]["f1"]
    print(f"lemmas: {lemmas}")
    print(f"tp: {cnt['tp']}  fp: {cnt['fp']}  fn: {cnt['fn']}  tn: {cnt['tn']}")
    print(f"p: {p*100:.2f}%  r: {r*100:.2f}%  f1: {f1*100:.2f}%")


if __name__ == "__main__":
    eval()
