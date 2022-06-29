import sqlite3 as sql
from data_proccess import read_congig_db_name

LAST_ROW_TIMESTAMP_QUERY = """
    SELECT time FROM bd ORDER BY id DESC LIMIT 1;
    """

LAST_I_ROW_QUERY = """
    SELECT  ("//i/i/i" ||'-'|| "i+i")
    FROM i ORDER BY id DESC LIMIT 1;
    """


def get_last_timestamp_query(timestamp):
    return f"""
    SELECT (a.id ||"-"|| b ||"-"|| a )
    FROM a
    LEFT JOIN bd on a.id = bd.id
    WHERE datetime(time) >= datetime('{timestamp}', '-30 seconds')
    ORDER BY d;
    """


def get_last_timestamp(db_name):
    with sql.connect(db_name) as con:
        cur = con.cursor()
        last_row_timestamp = cur.execute(LAST_ROW_TIMESTAMP_QUERY).fetchone()[0]
        first_command_query = get_last_timestamp_query(last_row_timestamp)
        last_timestamp = cur.execute(first_command_query).fetchall()
        return last_timestamp


def get_last_i_row(db_name):
    with sql.connect(db_name) as con:
        cur = con.cursor()
        last_i_row = cur.execute(LAST_I_ROW_QUERY).fetchone()
        return last_i_row


def ask_user():
    while True:
        try:
            command = int(input('Введите команду (1, 2, 9) -> '))
        except ValueError:
            print("Доступные команды: 1, 2 или 9")
            continue
        if command not in [1, 2, 9]:
            print("Доступные команды: 1, 2 или 9")
            continue
        return command


if __name__ == '__main__':
    db_name = read_congig_db_name()
    while True:
        command = ask_user()
        if command == 1:
            last_timestamp = get_last_timestamp(db_name)
            for i in last_timestamp:
                print(i[0])
        elif command == 2:
            last_i_row = get_last_i_row(db_name)
            print(last_i_row[0])
        elif command == 9:
            break
