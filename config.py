import psycopg2

HOSTNAME = '192.168.1.204'
USERNAME = 'postgres'
PASSWORD = '123'
DATABASE_NAME = 'data_lake'
PORT = 5432

postgres_connection_string = "postgresql://{DB_USER}:{DB_PASS}@{DB_ADDR}:{PORT}/{DB_NAME}".format(
    DB_USER=USERNAME,
    DB_PASS=PASSWORD,
    DB_ADDR=HOSTNAME,
    PORT=PORT,
    DB_NAME=DATABASE_NAME)


def get_postgres_connection():
    try:
        connection = psycopg2.connect(
            user=USERNAME,
            password=PASSWORD,
            host=HOSTNAME,
            port=PORT,
            database=DATABASE_NAME)
        return connection
    except (Exception, psycopg2.Error) as error:
        message = f"get_postgres_connection {error}"
        return abort(400, message)
