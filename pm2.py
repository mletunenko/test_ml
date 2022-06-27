from config import DB_NAME
import sqlite3 as sql

LAST_ROW_TIMESTAMP_QUERY = """
    SELECT time FROM bd ORDER BY id DESC LIMIT 1;
    """

LAST_I_ROW_QUERY = """
    SELECT  ("//i/i/i" ||'-'|| "i+i")
    FROM i ORDER BY id DESC LIMIT 1;
    """


def get_first_command_query(timestamp):
    return f"""
    SELECT (a.id ||"-"|| b ||"-"|| a )
    FROM a
    LEFT JOIN bd on a.id = bd.id
    WHERE datetime(time) >= datetime('{timestamp}', '-30 seconds')
    ORDER BY d;
    """


def get_last_timestamp():
    con = sql.connect(DB_NAME)
    with con:
        cur = con.cursor()
        last_row_timestamp = cur.execute(LAST_ROW_TIMESTAMP_QUERY).fetchone()[0]
        first_command_query = get_first_command_query(last_row_timestamp)
        last_timestamp = cur.execute(first_command_query).fetchall()
        return last_timestamp


def get_last_i_row():
    con = sql.connect(DB_NAME)
    with con:
        cur = con.cursor()
        last_i_row = cur.execute(LAST_I_ROW_QUERY).fetchone()
        return last_i_row


if __name__ == '__main__':
    wait = True
    while wait:
        command = int(input('Введите команду -> '))
        if command == 1:
            last_timestamp = get_last_timestamp()
            for i in last_timestamp:
                print(i[0])
        elif command == 2:
            last_i_row = get_last_i_row()
            print(last_i_row[0])
        elif command == 9:
            wait = False
