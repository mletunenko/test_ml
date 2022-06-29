import sqlite3 as sql
from datetime import datetime
import time
import configparser


create_table_a = """
CREATE TABLE IF NOT EXISTS "a" (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	a INTEGER
);"""

create_table_i = """
CREATE TABLE IF NOT EXISTS "i" (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	"//i/i/i" CHARACTER(20),
	"i+i" CHARACTER(20)
);"""

create_table_bd = """
CREATE TABLE IF NOT EXISTS "bd" (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	b INTEGER,
	d REAL,
	time TEXT
);"""


def create_tables(db_name):
    con = sql.connect(db_name)
    cur = con.cursor()
    cur.execute(create_table_a)
    cur.execute(create_table_i)
    cur.execute(create_table_bd)
    con.commit()
    con.close()

def export_to_sqlite_a(db_name, raw_data):
    con = sql.connect(db_name)
    cur = con.cursor()
    a = list(raw_data['A'])
    a = list(map(lambda x: (x,), a))
    cur.executemany('INSERT INTO "a" (a) VALUES (?);', a)
    con.commit()
    con.close()


def export_to_sqlite_i(db_name, raw_data):
    con = sql.connect(db_name)
    cur = con.cursor()
    a = list(raw_data['I'])
    a = list(map(lambda x: (f'//{x}/{x}/{x}', f'{x}+{x}'), a))
    cur.executemany('INSERT INTO "i" ("//i/i/i","i+i" ) VALUES (?, ?);', a)
    con.commit()
    con.close()

def export_to_sqlite_bd(db_name, raw_data):
    con = sql.connect(db_name)
    cur = con.cursor()
    b = list(raw_data['B'])
    d = list(raw_data['D'])
    for i in range(0, 400):
        if i % 100 == 0:
            time.sleep(60)
        cur.execute('INSERT INTO "bd" (b, d, time) VALUES (?, ?, ?);', (b[i], d[i], str(datetime.now())))

    con.commit()
    con.close()
def read_congig_db_name():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config['Database']['db_name']


def proccess_data(raw_data):
    db_name = read_congig_db_name()
    create_tables(db_name)
    export_to_sqlite_a(db_name, raw_data)
    export_to_sqlite_i(db_name, raw_data)
    export_to_sqlite_bd(db_name, raw_data)

