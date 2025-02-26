from os import makedirs

from loguru import logger
from typer import Option, Typer

import analyzer
import database
import fetcher
import sandbox
import train
import weekly_candidates
import with_ai

app = Typer()
app.add_typer(fetcher.app, name="fetch")
app.add_typer(analyzer.app, name="analyze")
app.add_typer(train.app, name="train")
app.add_typer(database.app, name="database")
app.add_typer(sandbox.app, name="sandbox")
app.add_typer(with_ai.app, name="with-ai")
app.add_typer(weekly_candidates.app, name="weekly-candidates")


@app.callback()
def main(
    verbose: bool = Option(False, "--verbose", "-v", help="Enable debug logging"),
    log_level: str = Option(
        "INFO", "--log-level", help="Set log level (e.g. DEBUG, INFO, WARNING)"
    ),
):
    # verboseがTrueならDEBUG、それ以外はユーザー指定のレベル
    level = "DEBUG" if verbose else log_level.upper()
    logger.remove()  # 既存の設定を削除
    log_dir = "./logs"
    makedirs(log_dir, exist_ok=True)
    logger.add(f"{log_dir}/app.log", rotation="10 MB", level=level)

    logger.info("app start.")


if __name__ == "__main__":
    app()
