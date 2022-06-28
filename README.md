# Описание репозитория

Настоящий репозиторий содержит в себе проект, выполнный в рамках тестового задания.

Код проекта написан на языке Python 3.8, с использованием библиотек для работы с данными и машинного обучения, зависимости
описаны в requirements.txt.

Проект работает с базой данных sqlite3.

Для использования проекта необходимо:

- Создать локально директорию для нового проекта python с виртуальным окружением
- Скопировать файлы данного репозитория в директорию с проектом
- Перейти в директорию проекта и выполнить команду _pip install -r requirements.txt_ для установки необходимых зависимостей

Репозиторий содержит файл с базой данных, которая содержит уже обработанные данные. Если вам необходимо определить другую базу
данных, можно удалить существующую базу либо изменить имя базы данных, тогда будет создана новая. Имя БД определяется в файле
_config.py_.

Исходные данные хранятся в файле _data.xlsx_.

Проект состоит из двух программных модулей.

## Программный модуль №1

ПМ1 осуществляет следущее:

- Извлекает данные из файла _data.xlsx_ и производит их обработку и подготовку
- Разбивает данные на 4 выборки: обучающие параметры, обучающие целевые переменные, тестовые параметры и тестовые целевые
  переменные
- Проводит обучение 5 моделей машинного обучения, используя при этом автоматический подбор гиперпараметров
- Вычисляет показатели для оценки качества полученных моделей
- Определяет 3 модели, показавшие лучшие результаты по итогу обучения. Отбор производится по средней абсолютной ошибке в
  процентах(MAPE)

- Предсказывает целевую переменную для всего набора данных с помощью 3 лучших моделей, добавляет полученные данные к исходному
  набору и сохраняет полученный результат в файл _test_result.xlsx_
- Создает список словарей, в котором содержатся показатели качества лучших моделей. Сохраняет данные в файл _models.json_
- Сохраняет модели в формате _.plk_
- Создает базу данных и таблицы, если они не существуют и заполняет таблицы полученным в процессе обработки данных дата-сетом

Для запуска программного модуля необходимо выполнить команду _python pm1.py_ из директории с проектом.

## Программный модуль №2

Для запуска программного модуля необходимо выполнить команду _python pm2.py_ из директории с проектом.

После запуска модуль находится в режиме ожидания команды от пользователя.

При вводе команды "1" модуль выводит записи, принадлежащие последней временной метке, в формате ID-B-A, отсортированные по полю
D.

При вводе команды "2" модуль выводит последнюю запись в формате //I/I/I-I+I.

При вводе команды "9" модуль заканчивает работу.

## Запросы SQL

Запросы выполняются в командной строке

Создание базы данных

    sqlite3 db_name.sqlite3

Создание таблицы A

    CREATE TABLE IF NOT EXISTS "a" (
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    a INTEGER);

Создание таблицы I

    CREATE TABLE IF NOT EXISTS "i" (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	"//i/i/i" CHARACTER(20),
	"i+i" CHARACTER(20));

Создание таблицы BD

    CREATE TABLE IF NOT EXISTS "bd" (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	b INTEGER,
	d REAL,
	time TEXT);

