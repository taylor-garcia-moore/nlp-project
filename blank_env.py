host = 'string host name'
user = 'string username'
password = 'string password'

def get_db_url(db):
    database_name = db
    url = f'mysql+pymysql://{user}:{password}@{host}/{database_name}'
    return url
