import json
from datetime import datetime
import pytz
from ecmasap.project.agents import MainAgent
from peak import Message, Agent
from ..utils_package.repository import Session, Model

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
    with open('config_settings.json') as f:
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


async def initialize_app():
    with open('config_tables_db.json') as f:
        config_tables_db = json.load(f)
    with open('config_settings.json') as f:
        config_settings = json.load(f)
    tables = config_tables_db['tables']
    dataset_types = config_settings.get('possible_data_types')
    dataset_types = list(dataset_types.keys())
    for table in tables:
        for table_name, columns in table.items():
            for column in columns:
                for dataset_type in dataset_types:
                    exist = await check_if_model_exists(column, dataset_type)
                    if not exist:
                        request_type = 'train'
                        data = await get_data_timestamp(config_settings)
                        data["target_table"] = table_name,
                        data['target'] = column
                        data['dataset_type'] = dataset_type
                        utils_agent = Utils(request_type, data, column)
                        utils_agent.setup()


async def get_data_timestamp(config_settings):
    current_time = datetime.now()
    formated_timestamp = config_settings.get('datetime_format')
    formatted_string = current_time.strftime(formated_timestamp)
    parts_timestamps = formatted_string.split(' ')
    end_date = parts_timestamps[0]
    end_time = parts_timestamps[1]
    frequency = config_settings['frequencies_list'][0]
    data = {'end_date': end_date, 'end_time': end_time, 'frequency': frequency}
    return data


async def check_if_model_exists(target, dataset_type):
    session = Session()
    models = session.query(Model.characteristics).filter_by(
        target_name=target).first()
    session.close()

    model_characteristics = models.characteristics
    if model_characteristics:
        if not isinstance(model_characteristics, dict):
            model_characteristics = json.loads(model_characteristics)
        model_characteristic = model_characteristics['dataset_type']
        if model_characteristic == dataset_type:
            return True
        return False
    return False


async def process_request(data, request_type):
    target = data['target']
    utils_agent = Utils(request_type, data, target)
    utils_agent.setup()


class Utils(Agent):
    def __init__(self, request_type, request_data, target):
        super().__init__()
        self.request_type = request_type
        self.request_data = request_data
        self.target = target

    async def trainModels(self, request_type, request_data, target):
        # Load your configuration
        with open('utils_package/config_agents.json') as config_file:
            config = json.load(config_file)

        # Define what type of behavior or request needs to be triggered
        agent_name = config['main_agent']
        jid_domain = config['jid_domain']
        # Assuming you want to simulate the training trigger

        msg = Message(to=f"{agent_name}@{jid_domain}/{agent_name}")
        msg.set_metadata("performative", "request")  # Set the "inform" FIPA performative
        if isinstance(request_data, dict):
            request_data = json.dumps(request_data)
        msg.body = request_type + "|" + target + "|" + request_data
        await self.send(msg)

        # Wait for a response
        response = await self.receive()
        if response:
            if response.get_metadata("performative") == "inform":
                print(f"Response received: {response.body}")
            elif response.get_metadata("performative") == "failure":
                print(f"Request failed: {response.body}")
        else:
            print("No response received within timeout.")

    async def setup(self):
        request_type = self.request_type
        request_data = self.request_data
        target = self.target
        b = self.trainModels(request_type, request_data, target)
        self.add_behaviour(b)
