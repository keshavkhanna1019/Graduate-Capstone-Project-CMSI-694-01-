import sqlite3

DB_PATH = "face.db"


def get_connection():
    return sqlite3.connect(DB_PATH)
