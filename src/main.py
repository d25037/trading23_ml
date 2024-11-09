from typer import Typer

import analyzer
import database
import fetcher

app = Typer()
app.add_typer(fetcher.app, name="fetcher")
app.add_typer(analyzer.app, name="analyzer")


@app.command()
def main():
    # analyzer.dataset_reader()
    analyzer.analyzer("day5")


if __name__ == "__main__":
    app()
