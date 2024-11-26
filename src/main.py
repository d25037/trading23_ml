from typer import Typer

import analyzer
import database
import fetcher
import sandbox
import train

app = Typer()
app.add_typer(fetcher.app, name="fetch")
app.add_typer(analyzer.app, name="analyze")
app.add_typer(train.app, name="train")
app.add_typer(database.app, name="database")
app.add_typer(sandbox.app, name="sandbox")


if __name__ == "__main__":
    app()
