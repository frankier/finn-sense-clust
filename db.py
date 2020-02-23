import click

from senseclust.db.cmd import add_cmds


@click.group()
def db():
    pass


add_cmds(db)

if __name__ == "__main__":
    db()
