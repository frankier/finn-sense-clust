import fileinput
from senseclust.groupings import gen_groupings


def main():
    inf = fileinput.input()
    next(inf)
    for lemma, groupings in gen_groupings(inf):
        if len(groupings) <= 1:
            continue
        print(lemma)


if __name__ == "__main__":
    main()
