import os,json, logging, asyncio, pyodbc
import time
import datetime
class Repo:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='error.log',
                        filemode='a')

    async def createConnection():
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(BASE_DIR, 'db_settings.json')) as f:
                connection_details = json.load(f)

            # Create a connection string
            conn_str = (
                f"DRIVER={{{connection_details['Driver']}}};"
                f"SERVER={connection_details['Host']};"
                f"DATABASE={connection_details['Name']};"
                f"UID={connection_details['Username']};"
                f"PWD={connection_details['Password']};"
            )

            # Establish a connection to the database
            conn = await asyncio.to_thread(pyodbc.connect, conn_str)
            return conn
        except Exception as ex:
            await asyncio.sleep(15)
            return await Repo.createConnection()

    async def closeConnection(conn):
        try:
            conn.close()
        except Exception as ex:
            raise ex

    async def executeQuery(query):
        timeout_val = 600
        start_time = time.time()
        while True:
            try:
                conn = await Repo.createConnection()
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    records = cursor.fetchall()
                await Repo.closeConnection(conn)
                return records
            except pyodbc.Error as ex:
                print('[', datetime.now(),'] [db-emul] error in execute query', ex)
                try:
                    conn.rollback()

                    # Check if the timeout has been reached
                    if time.time() - start_time >= timeout_val:
                        try:
                            await Repo.closeConnection(conn)
                        except:
                            pass
                        print(ex)
                        return str(ex)
                    else:
                        time.sleep(1)
                except Exception as ex:
                    return str(ex)