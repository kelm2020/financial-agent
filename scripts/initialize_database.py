from alembic import command
from alembic.config import Config
from langgraph.checkpoint.postgres import PostgresSaver

from config.settings import get_settings


def main() -> None:
    settings = get_settings()
    command.upgrade(Config("alembic.ini"), "head")
    with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
        checkpointer.setup()


if __name__ == "__main__":  # pragma: no cover
    main()
