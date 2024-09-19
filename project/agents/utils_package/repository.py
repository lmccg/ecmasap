import psycopg2
import json
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, LargeBinary, ARRAY, DateTime, JSON, BOOLEAN, \
    func, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker

# Load configuration from JSON file
with open('utils_package/config_database.json') as config_file:
    config = json.load(config_file)

DB_USERNAME = config["database"]["db_username"]
DB_PASSWORD = config["database"]["db_password"]
DB_HOST = config["database"]["db_host"]
DB_PORT = config["database"]["db_port"]
DB_NAME = config["database"]["db_name"]
DATABASE_URL = f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


class Model(Base):
    __tablename__ = 'models'
    model_id = Column(Integer, primary_key=True)
    model_binary = Column(LargeBinary)
    train_data = Column(LargeBinary)
    x_train_data_norm = Column(LargeBinary)
    y_train_data_norm = Column(LargeBinary)
    x_scaler = Column(LargeBinary)
    y_scaler = Column(LargeBinary)
    columns_names = Column(ARRAY(String), nullable=False)
    target_feature = Column(String, nullable=False)
    target_zone = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    ml_model = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    model_params = Column(JSON, nullable=False)
    test_errors = Column(JSON)
    train_errors = Column(JSON, nullable=False)
    notes = Column(JSON)
    dataset_transformations = Column(JSON)
    default_metric = Column(String, nullable=False)
    characteristics = Column(JSON, nullable=False)
    # explainer = Column(LargeBinary)
    # explainer_data = Column(LargeBinary)
    # global_shap_values = Column(LargeBinary)
    # base_values = Column(ARRAY(Float))
    # classes = Column(ARRAY(String))
    # global_explanations = Column(LargeBinary)
    # local_explanations = Column(LargeBinary)
    historic_predictions_model = Column(JSON)
    historic_scores_model = Column(JSON)
    historic_norm_test_data = Column(JSON)
    retrain_counter = Column(Integer, nullable=False)
    flag_training = Column(BOOLEAN, nullable=False)
    models_version = Column(String, nullable=False)
    training_dates = Column(JSON, nullable=False)
    registered_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    model_binary_oid = Column(BigInteger)


class Result(Base):
    __tablename__ = 'results'
    result_id = Column(Integer, primary_key=True)
    model_id = Column(Integer, nullable=False)
    input_data = Column(JSON, nullable=False)
    result_values = Column(JSON, nullable=False)
    execution_time = Column(DateTime, nullable=False)
    chosen_model = Column(BOOLEAN, nullable=False)


def create_database_if_not_exists():
    connection = psycopg2.connect(
        dbname="postgres",
        user=DB_USERNAME,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = connection.cursor()
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_NAME}';")
    exists = cursor.fetchone()
    if not exists:
        cursor.execute(f'CREATE DATABASE {DB_NAME};')
    cursor.close()
    connection.close()


def initialize_database():
    Base.metadata.create_all(engine)


async def start_app():
    # Your logic here
    create_database_if_not_exists()
    initialize_database()


def create_connection():
    connection = psycopg2.connect(
        dbname="postgres",
        user=DB_USERNAME,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return connection


def close_connection(connection):
    connection.close()
