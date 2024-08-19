import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary, ARRAY, DateTime, JSON, BOOLEAN, func
from sqlalchemy.orm import declarative_base, sessionmaker
from utils import DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DATABASE_URL

Base = declarative_base()
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


class Model(Base):
    __tablename__ = 'models'
    model_id = Column(Integer, primary_key=True)
    model_binary = Column(LargeBinary, nullable=False)
    train_data = Column(LargeBinary, nullable=False)
    x_train_data_norm = Column(LargeBinary, nullable=False)
    y_train_data_norm = Column(LargeBinary, nullable=False)
    x_scaler = Column(LargeBinary, nullable=False)
    y_scaler = Column(LargeBinary, nullable=False)
    columns_names = Column(ARRAY(String), nullable=False)
    target_name = Column(String, nullable=False)
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
    # explainer = Column(LargeBinary, nullable=False)
    # explainer_data = Column(LargeBinary, nullable=False)
    # global_shap_values = Column(LargeBinary, nullable=False)
    # base_values = Column(ARRAY(Float), nullable=False)
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

