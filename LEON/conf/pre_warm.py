import psycopg2
import configparser
import time
import os

def get_config_value(section, key, default=None):
    """
    Safely get a value from config file
    """
    config = configparser.ConfigParser()
    # Try reading from current directory first, then from conf directory
    config_path = 'leon.cfg' if os.path.exists('leon.cfg') else 'conf/leon.cfg'
    config.read(config_path)
    
    try:
        return config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        print(f"Warning: {section}.{key} not found in config")
        return default

def read_config(section):    
    config = configparser.ConfigParser()
    # Try reading from current directory first, then from conf directory
    config_path = 'leon.cfg' if os.path.exists('leon.cfg') else 'conf/leon.cfg'
    config.read(config_path)
    return config[section]

def prewarm_pg(port):
    conf = read_config('PostgreSQL')
    database = conf['database']
    user = conf['user']
    password = conf['password']
    host = conf['host']
    # port = int(conf['port'])
    print("PostgreSQL Start PreWarming:")
    start = time.time()
    with psycopg2.connect(database=database, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:
            # Create extension if not exists
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm;")
                conn.commit()
            except Exception as e:
                print(f"Warning: Could not create extension: {e}")
                conn.rollback()
    
            # Prewarm main tables
            tables = [
                'aka_name', 'aka_title', 'cast_info', 'char_name',
                'comp_cast_type', 'company_name', 'company_type', 'complete_cast',
                'info_type', 'keyword', 'kind_type', 'link_type',
                'movie_companies', 'movie_info', 'movie_info_idx', 'movie_keyword',
                'movie_link', 'name', 'person_info', 'role_type', 'title'
            ]
            
            for table in tables:
                try:
                    cur.execute(f"SELECT pg_prewarm('{table}', 'buffer', 'main');")
                except Exception as e:
                    print(f"  Warning: Could not prewarm table {table}: {e}")
            
            conn.commit()
            print(f"PostgreSQL Finish PreWarming, total time: {time.time() - start} s")

if __name__ == "__main__":
    # Get ports safely
    ports = []
    
    # Get other_db_port from leon section
    other_db_port_str = get_config_value('leon', 'other_db_port', '[]')
    if other_db_port_str:
        try:
            ports.extend(eval(other_db_port_str))
        except Exception as e:
            print(f"Error parsing other_db_port: {e}")
    
    # Get main PostgreSQL port
    pg_port = get_config_value('PostgreSQL', 'Port')
    if pg_port:
        try:
            ports.append(int(pg_port))
        except ValueError:
            print(f"Invalid port value: {pg_port}")
    
    # If no ports found, use default port 5433
    if not ports:
        print("No ports found in config, using default port 5433")
        ports = [5433]
    
    print(f"Pre-warming databases on ports: {ports}")
    for port in ports:
        print(f"Pre-warming port {port}...")
        prewarm_pg(port)
    