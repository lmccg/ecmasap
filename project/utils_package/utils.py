import json
from datetime import datetime
import pytz
# Load configuration from JSON file
with open('config_database.json') as config_file:
    config = json.load(config_file)

DB_USERNAME = config["database"]["db_username"]
DB_PASSWORD = config["database"]["db_password"]
DB_HOST = config["database"]["db_host"]
DB_PORT = config["database"]["db_port"]
DB_NAME = config["database"]["db_name"]
DATABASE_URL = f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def timestamp_with_time_zone():
    with open('./config_settings.json') as f:
        config_settings = json.load(f)
    current_time = datetime.now()
    timezone = config_settings.get('timezone')  # Desired timezone
    formated_timestamp = config_settings.get('datetime_format')
    # Set timezone
    pttz = pytz.timezone(timezone)

    # If current_time is naive (no timezone info), localize it
    if current_time.tzinfo is None:
        current_time = pttz.localize(current_time)
    else:
        # Convert to the specified timezone
        current_time = current_time.astimezone(pttz)
    # Convert to string, with specific format
    formatted_string = current_time.strftime(formated_timestamp)
    # Convert string back to datetime
    current_time = datetime.strptime(formatted_string, formated_timestamp)
    return current_time