import click
import re


SYNSET_RE = re.compile(r"[0-9]{8}-[anv]")


@click.command()
@click.argument("inf", type=click.File('r'))
@click.argument("outf", type=click.File('w'))
def main(inf, outf):
    assert inf.readline().strip() == "manann,ref"
    outf.write("manann,ref\n")
    for line in inf:
        manann, ref = line.strip().split(",")
        if not SYNSET_RE.match(ref):
            continue
        outf.write(line)


if __name__ == "__main__":
    main()
