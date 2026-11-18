from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "instance" / "churn.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def add_column_if_missing(connection: sqlite3.Connection, table_name: str, column_sql: str) -> None:
    column_name = column_sql.split()[0]
    if not column_exists(connection, table_name, column_name):
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")
        print(f"Added {table_name}.{column_name}")


def backup_database() -> None:
    if DB_PATH.exists():
        backup_path = DB_PATH.with_name("churn.sqlite.pre-v3.bak")
        shutil.copy2(DB_PATH, backup_path)
        print(f"Backed up database to {backup_path}")


def migrate() -> None:
    backup_database()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    connection.commit()
    connection.close()
    print(f"Migration v3 complete for {DB_PATH}")


if __name__ == "__main__":
    migrate()
