from typer import Typer

import analyzer
import database
import fetcher
import sandbox

app = Typer()
app.add_typer(fetcher.app, name="fetcher")
app.add_typer(analyzer.app, name="analyzer")
app.add_typer(database.app, name="database")
app.add_typer(sandbox.app, name="sandbox")


if __name__ == "__main__":
    app()
