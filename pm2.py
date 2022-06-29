import sqlite3 as sql
from data_proccess import read_congig_db_name
from urllib.request import pathname2url

LAST_ROW_TIMESTAMP_QUERY = """
    SELECT time FROM bd ORDER BY id DESC LIMIT 1;
    """

LAST_I_ROW_QUERY = """
    SELECT  ("//i/i/i" ||'-'|| "i+i")
    FROM i ORDER BY id DESC LIMIT 1;
    """


def get_last_timestamp_query(timestamp):
    """Create qyery text according to timestamp"""
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
        # Try to execute query to database or stop execution of program
        try:
            last_row_timestamp = cur.execute(LAST_ROW_TIMESTAMP_QUERY).fetchone()[0]
        except sql.OperationalError:
            print(f'Таблицы "bd" не существует в базе данных')
            exit(1)
        timestamp_command_query = get_last_timestamp_query(last_row_timestamp)
        # Try to execute query to database or stop execution of program
        try:
            last_timestamp = cur.execute(timestamp_command_query).fetchall()
        except sql.OperationalError:
            print(f'Таблицы "bd" и/или таблицы "а" не существует в базе данных')
            exit(1)
        return last_timestamp


def get_last_i_row(db_name):
    """Execute query to get last record from database"""
    with sql.connect(db_name) as con:
        cur = con.cursor()
        # Try to execute query to database or stop execution of program
        try:
            last_i_row = cur.execute(LAST_I_ROW_QUERY).fetchone()
        except sql.OperationalError:
            print(f'Таблицы "i" не существует в базе данных')
            exit(1)
        return last_i_row


def ask_user():
    """Processing of user's comand, check if the last one is correct or ask user to input correct comand"""
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
    # Get database name from config file
    db_name = read_congig_db_name()
    # Try to connect to database or stop execution of program
    try:
        dburi = 'file:{}?mode=rw'.format(pathname2url(db_name))
        conn = sql.connect(dburi, uri=True)
    except sql.OperationalError:
        print(f'Файл базы данных {db_name} не существует в директории проекта')
        exit(1)
    # Wait for the command from user and execute program
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
