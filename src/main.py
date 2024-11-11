import random

from typer import Typer

import analyzer
import fetcher

app = Typer()
app.add_typer(fetcher.app, name="fetcher")
app.add_typer(analyzer.app, name="analyzer")


@app.command()
def test():
    k = 1.36841
    l = -5.6551
    print(round(k, 2))
    print(round(l, 2))

    numbers = random.sample(range(20, 1000), 100)
    print(sorted(numbers))
    print(len(numbers))


@app.command()
def run(label: str):
    # analyzer.dataset_reader()
    analyzer.analyzer(label)


if __name__ == "__main__":
    app()
