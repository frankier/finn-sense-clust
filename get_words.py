import fileinput
from groupings import gen_groupings


def main():
    inf = fileinput.input()
    next(inf)
    for lemma, clusts in gen_groupings(inf):
        print(lemma)


if __name__ == "__main__":
    main()
