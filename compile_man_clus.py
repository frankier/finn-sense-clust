import click
from os.path import basename


@click.command()
@click.argument("infs", nargs=-1)
@click.argument("out", type=click.File('w'))
def main(infs, out):
    out.write("manann,ref\n")
    for inf in infs:
        word_pos = basename(inf)
        word = word_pos.split(".")[0]
        idx = 1
        with open(inf) as f:
            for line in f:
                if not line.strip():
                    idx += 1
                else:
                    ref = line.split("#")[0].strip()
                    out.write(f"{word}.{idx:02d},{ref}\n")


if __name__ == "__main__":
    main()
