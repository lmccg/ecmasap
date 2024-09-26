from peak import Agent, Message, CyclicBehaviour
import json
import pickle
import base64
import zlib
from utils_package.repository import Session, Model, Result, start_app, engine,create_connection, close_connection
from utils_package.utils import timestamp_with_time_zone
from sqlalchemy import func, asc
from collections import defaultdict
import asyncio
from datetime import datetime
import pickle
import os


class DBAgent(Agent):
    class ReceiveMsg(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(10)
            if msg:
                # if msg.get_metadata("compressed"):
                print(f"{timestamp_with_time_zone()} DB agent: {msg.sender} sent me a message {msg.body}")
                #
                #     encoded_data = msg.body
                #
                #     # Decode the Base64-encoded string
                #     compressed_data = base64.b64decode(encoded_data)
                #
                #     # Decompress the data
                #     new_msg = zlib.decompress(compressed_data).decode('utf-8')
                # else:
                #     # print(f"DB agent: {msg.sender} sent me a message: '{msg.body}'")
                new_msg = msg.body
                if 'alive' in new_msg:
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = json.dumps(True)
                    await self.send(response_msg)
                elif 'save_model' in new_msg:
                    print(timestamp_with_time_zone(), 'Saving model')
                    data_to_insert_in_database = json.loads(new_msg)
                    data_to_insert_in_database = data_to_insert_in_database['save_model']
                    response_db = await self.save_model_to_db(data_to_insert_in_database)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    if response_db:
                        print(timestamp_with_time_zone(), 'db response', response_db, type(response_db))
                        response_msg.body = response_db
                    else:
                        response_msg.body = "failure"
                    await self.send(response_msg)
                if 'part_binary' in new_msg:
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    parts_of_msg = new_msg.split('|')
                    id_model = parts_of_msg[1]
                    column = parts_of_msg[2]
                    value = parts_of_msg[3]
                    response_db = await self.save_part_of_model_to_db(id_model, column, value)
                    if response_db:
                        print(timestamp_with_time_zone(), response_db)
                        response_msg.body = response_db
                    else:
                        response_msg.body = "failure"
                    await self.send(response_msg)
                # if data_to_insert_in_database['target'] == 'model':
                #     input = data_to_insert_in_database['input_data']
                #     response_db = await self.save_model_to_db(input)
                #     # REPLY BACK
                #     response_msg = msg.make_reply()
                #     response_msg.set_metadata("performative", "inform")
                #     if response_db:
                #         response_msg.body = "success"
                #     else:
                #         response_msg.body = "failure"
                #     await self.send(response_msg)
                elif 'save_result' in new_msg:
                    data_to_insert_in_database = json.loads(new_msg)
                    data_to_insert_in_database = data_to_insert_in_database['save_result']
                    response_db = await self.save_result_to_db(data_to_insert_in_database)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    if response_db:
                        response_msg.body = "success"
                    else:
                        response_msg.body = "failure"
                    await self.send(response_msg)

                elif 'get_regressor_and_scalers' in new_msg:
                    info = json.loads(new_msg)
                    model_id = info['get_regressor_and_scalers']
                    response_db = await self.get_regressor_and_scalers(model_id)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)
                elif 'update_model' in new_msg:
                    info = json.loads(new_msg)
                    model_info = info['update_model']
                    response_db = await self.update_model(model_info)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)

                elif 'update_model_historic' in new_msg:
                    info = json.loads(new_msg)
                    model_info = info['update_model_historic_norm']
                    response_db = await self.update_model_historic(model_info)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)

                elif 'update_model_historics' in new_msg:
                    info = json.loads(new_msg)
                    model_info = info['update_model_historics']
                    response_db = await self.update_model_historics(model_info)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)

                elif 'get_model_historic_norm_and_version' in new_msg:
                    info = json.loads(new_msg)
                    model_info = info['update_model_historic_norm']
                    response_db = await self.get_model_historic_norm_and_version(model_info)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)

                elif 'get_models' in new_msg:
                    response_db = await self.get_models()
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)
                elif 'get_models_to_evaluate' in new_msg:
                    response_db = await self.get_models_to_evaluate()
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)
                elif 'get_data_xai' in new_msg:
                    response_db = await self.get_xai_data()
                    # REPLY BACK
                    response_db = json.dumps(response_db)
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)
                elif 'get_models_to_check_retrain' in new_msg:
                    info = json.loads(new_msg)
                    model_id = info['get_data_check_retrain']
                    response_db = await self.get_data_to_retrain(model_id)
                    response_db = json.dumps(response_db)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)
                elif 'get_model_for' in new_msg:
                    info = json.loads(new_msg)
                    model_info = info['get_model_for']
                    response_db = await self.check_if_model_exists(model_info)
                    response_db = json.dumps(response_db)
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)
                elif 'print_model' in new_msg:
                    response_db = await self.print_model()
                    response_db = json.dumps(response_db)
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)
            else:
                print(timestamp_with_time_zone(), 'waiting for msg')


        def load_object_from_file(self, file_path):
            if 'pkl' in file_path:
                # Open the file and load the object using pickle
                with open(file_path, 'rb') as f:
                    object = pickle.load(f)
            elif 'txt' in file_path:
                # Open the file and load the object using json
                with open(file_path, 'r') as f:
                    object = f.read()
            os.remove(file_path)
            # Return the loaded object
            return object
        # async def handle_compress_msg(self, key, msg):
        #     if key == 'save_model':
        #         print(timestamp_with_time_zone(), "Saving model")
        #         print(timestamp_with_time_zone(), "Received 'I'm here'. Waiting for other messages...")
        #         # Wait for the next two messages
        #         messages_received = 0
        #         while messages_received < 2:
        #             response = await self.receive(timeout=60)  # Adjust timeout as needed
        #             if response:
        #                 print(timestamp_with_time_zone(), response)
        #         new_msg = msg.body
        #         data_to_insert_in_database = json.loads(new_msg)
        #         data_to_insert_in_database = data_to_insert_in_database['save_model']
        #         response_db = await self.save_model_to_db(data_to_insert_in_database)
        #         # REPLY BACK
        #         response_msg = msg.make_reply()
        #         response_msg.set_metadata("performative", "inform")
        #         if response_db:
        #             print(timestamp_with_time_zone(), response_db)
        #             response_msg.body = "success"
        #         else:
        #             response_msg.body = "failure"
        #         await self.send(response_msg)
        def extract_shap_data(self, shap_values):
            if isinstance(shap_values, list):
                return [shap_array.values if hasattr(shap_array, 'values') else shap_array for shap_array in
                        shap_values]
            elif hasattr(shap_values, 'values'):
                return shap_values.values
            return shap_values

        async def save_model_to_db(self, model_info):
            session = None
            try:
                print(timestamp_with_time_zone(), "lets save model")
                session = Session()
                if 'model_id' in model_info:
                    model = session.query(Model).filter_by(model_id=model_info['model_id']).first()
                    model_binary = self.load_object_from_file(model_info.get('model_binary'))
                    # print(type(model_binary))
                    # model_binary = model_binary.encode('latin1')
                    model_binary = pickle.loads(model_binary)
                    model_binary = pickle.dumps(model_binary)
                    model_binary = zlib.compress(model_binary)
                    train_data = pickle.dumps(self.load_object_from_file(model_info.get('train_data')))
                    x_train = pickle.dumps(self.load_object_from_file(model_info.get('x_train_data_norm')))
                    y_train = pickle.dumps(self.load_object_from_file(model_info.get('y_train_data_norm')))
                    x_scaler = pickle.dumps(self.load_object_from_file(model_info.get('x_scaler')))
                    y_scaler = pickle.dumps(self.load_object_from_file(model_info.get('y_scaler')))
                    if model:
                        model.train_data = train_data
                        model.x_train_data_norm = x_train
                        model.y_train_data_norm = y_train
                        model.x_scaler = x_scaler
                        model.y_scaler = y_scaler
                        model.columns_names = model_info.get('columns_names')
                        model.target_feature = model_info.get('target_feature')
                        model.target_zone = model_info.get('target_zone')
                        model.model_name = model_info.get('model_name')
                        model.model_type = model_info.get('model_type')
                        model.model_params = json.dumps(model_info.get('model_params'))
                        model.test_errors = json.dumps(model_info.get('test_errors'))
                        model.train_errors = json.dumps(model_info.get('train_errors'))
                        model.notes = json.dumps(model_info.get('notes'))
                        model.dataset_transformations = json.dumps(model_info.get('dataset_transformations'))
                        model.default_metric = model_info.get('default_metric')
                        model.characteristics = json.dumps(model_info.get('characteristics'))
                        # model.explainer = pickle.dumps(model_info.get('explainer'))
                        # model.explainer_data = pickle.dumps(model_info.get('explainer_data'))
                        # model.global_shap_values = pickle.dumps(self.extract_shap_data(model_info.get('global_shap_values')))
                        # model.base_values = model_info.get('base_values')
                        # model.classes = model_info.get('classes')
                        # model.global_explanations = pickle.dumps(model_info.get('global_explanations', {}))
                        # model.local_explanations = pickle.dumps(model_info.get('local_explanations', {}))
                        if model_info.get('historic_predictions_model') is not None:
                            model.historic_predictions_model = json.dumps(model_info.get('historic_predictions_model'))
                        if model_info.get('historic_scores_model') is not None:
                            model.historic_scores_model = json.dumps(model_info.get('historic_scores_model'))
                        if model_info.get('historic_norm_test_data') is not None:
                            model.historic_norm_test_data = json.dumps(model_info.get('historic_norm_test_data'))
                        model.retrain_counter = model_info.get('retrain_counter')
                        model.flag_training = model_info.get('flag_training')
                        model.models_version = model_info.get('models_version')
                        model.training_dates = json.dumps(model_info.get('training_dates'))
                        model.updated_at = func.now()
                        session.add(model)
                        session.commit()
                        model_id = model.model_id
                    else:
                        print(timestamp_with_time_zone(), "Model dont exists line 255")
                        # model_binary = pickle.dumps(self.load_object_from_file(model_info.get('model_binary')))
                        # train_data = pickle.dumps(self.load_object_from_file(model_info.get('train_data')))
                        # x_train = pickle.dumps(self.load_object_from_file(model_info.get('x_train_data_norm')))
                        # y_train = pickle.dumps(self.load_object_from_file(model_info.get('y_train_data_norm')))
                        # x_scaler = pickle.dumps(self.load_object_from_file(model_info.get('x_scaler')))
                        # y_scaler = pickle.dumps(self.load_object_from_file(model_info.get('y_scaler')))
                        model = Model(
                            # model_binary=model_binary,
                            train_data=train_data,
                            x_train_data_norm=x_train,
                            y_train_data_norm=y_train,
                            x_scaler=x_scaler,
                            y_scaler=y_scaler,
                            columns_names=model_info.get('columns_names'),
                            target_feature=model_info.get('target_feature'),
                            target_zone=model_info.get('target_zone'),
                            model_name=model_info.get('model_name'),
                            model_type=model_info.get('model_type'),
                            ml_model=model_info.get('ml_model'),
                            model_params=json.dumps(model_info.get('model_params')),
                            test_errors=json.dumps(model_info.get('test_errors')),
                            train_errors=json.dumps(model_info.get('train_errors')),
                            notes=json.dumps(model_info.get('notes')),
                            dataset_transformations=json.dumps(model_info.get('dataset_transformations')),
                            default_metric=model_info.get('default_metric'),
                            characteristics=json.dumps(model_info.get('characteristics')),
                            # explainer = pickle.dumps(model_info.get('explainer'))
                            # explainer_data = pickle.dumps(model_info.get('explainer_data'))
                            # global_shap_values = pickle.dumps(self.extract_shap_data(model_info.get('global_shap_values')))
                            # base_values = model_info.get('base_values')
                            # classes = model_info.get('classes')
                            # global_explanations = pickle.dumps(model_info.get('global_explanations', {}))
                            # local_explanations = pickle.dumps(model_info.get('local_explanations', {}))
                            retrain_counter=model_info.get('retrain_counter'),
                            flag_training=model_info.get('flag_training'),
                            models_version=model_info.get('models_version'),
                            training_dates=json.dumps(model_info.get('training_dates')),
                            registered_at=func.now(),
                            updated_at=func.now()
                        )
                        session.add(model)
                        session.commit()
                        model_id = model.model_id
                        if model_info.get('historic_predictions_model') is not None:
                            await self.save_other_entries(model_id, session,'historic_predictions_model',
                                                          json.dumps(model_info.get('historic_predictions_model')))
                        if model_info.get('historic_scores_model') is not None:
                            await self.save_other_entries(model_id, session, 'historic_scores_model', json.dumps(model_info.get('historic_scores_model')))
                        if model_info.get('historic_norm_test_data') is not None:
                            await self.save_other_entries(model_id, session, 'historic_norm_test_data', json.dumps(model_info.get('historic_norm_test_data')))
                        await self.save_large_binaries(model_id, session, model_binary)
                    session.close()
                    return str(model_id)
                else:
                    print(timestamp_with_time_zone(), "Model dont exists line 305")
                    model_binary = self.load_object_from_file(model_info.get('model_binary'))
                    # print(type(model_binary))
                    # model_binary = model_binary.encode('latin1')
                    model_binary = pickle.dumps(model_binary)
                    model_binary = zlib.compress(model_binary)
                    train_data = pickle.dumps(self.load_object_from_file(model_info.get('train_data')))
                    x_train = pickle.dumps(self.load_object_from_file(model_info.get('x_train_data_norm')))
                    y_train = pickle.dumps(self.load_object_from_file(model_info.get('y_train_data_norm')))
                    x_scaler = pickle.dumps(self.load_object_from_file(model_info.get('x_scaler')))
                    y_scaler = pickle.dumps(self.load_object_from_file(model_info.get('y_scaler')))
                    model = Model(
                        # model_binary=model_binary,
                        train_data=train_data,
                        x_train_data_norm=x_train,
                        y_train_data_norm=y_train,
                        x_scaler=x_scaler,
                        y_scaler=y_scaler,
                        columns_names=model_info.get('columns_names'),
                        target_feature=model_info.get('target_feature'),
                        target_zone=model_info.get('target_zone'),
                        model_name=model_info.get('model_name'),
                        model_type=model_info.get('model_type'),
                        ml_model=model_info.get('ml_model'),
                        model_params=json.dumps(model_info.get('model_params')),
                        test_errors=json.dumps(model_info.get('test_errors')),
                        train_errors=json.dumps(model_info.get('train_errors')),
                        notes=json.dumps(model_info.get('notes')),
                        dataset_transformations=json.dumps(model_info.get('dataset_transformations')),
                        default_metric=model_info.get('default_metric'),
                        characteristics=json.dumps(model_info.get('characteristics')),
                        # explainer = pickle.dumps(model_info.get('explainer'))
                        # explainer_data = pickle.dumps(model_info.get('explainer_data'))
                        # global_shap_values = pickle.dumps(self.extract_shap_data(model_info.get('global_shap_values')))
                        # base_values = model_info.get('base_values')
                        # classes = model_info.get('classes')
                        # global_explanations = pickle.dumps(model_info.get('global_explanations', {}))
                        # local_explanations = pickle.dumps(model_info.get('local_explanations', {}))
                        # historic_predictions_model=json.dumps(model_info.get('historic_predictions_model')),
                        # historic_scores_model=json.dumps(model_info.get('historic_scores_model')),
                        # historic_norm_test_data=json.dumps(model_info.get('historic_norm_test_data')),
                        retrain_counter=model_info.get('retrain_counter'),
                        flag_training=model_info.get('flag_training'),
                        models_version=model_info.get('models_version'),
                        training_dates=json.dumps(model_info.get('training_dates')),
                        registered_at=func.now(),
                        updated_at=func.now()
                    )
                    session.add(model)
                    session.commit()
                    model_id = model.model_id
                    if model_info.get('historic_predictions_model') is not None:
                        await self.save_other_entries(model_id, session, 'historic_predictions_model',
                                                      json.dumps(model_info.get('historic_predictions_model')))
                    if model_info.get('historic_scores_model') is not None:
                        await self.save_other_entries(model_id, session, 'historic_scores_model',
                                                      json.dumps(model_info.get('historic_scores_model')))
                    if model_info.get('historic_norm_test_data') is not None:
                        await self.save_other_entries(model_id, session, 'historic_norm_test_data',
                                                      json.dumps(model_info.get('historic_norm_test_data')))
                    await self.save_large_binaries(model_id, session, model_binary)
                    # await self.save_large_binaries(model_id, session, model_binary)
                    session.close()
                    return str(model_id)
            except Exception as e:
                print(timestamp_with_time_zone(), 'exception', e)
                session.rollback()
                return 'Failed'


        # def store_large_object(self, connection, binary_data):
        #     print(timestamp_with_time_zone(), 'store large object')
        #     with connection.cursor() as cursor:
        #         cursor.execute("BEGIN;")  # Start a transaction
        #
        #         # Create a new large object
        #         cursor.execute("SELECT lo_create(0);")
        #         lo_oid = cursor.fetchone()[0]
        #
        #         # Open the large object for writing
        #         lo_fd = connection.lobject(lo_oid, mode="w")
        #
        #         # Write the binary data in chunks (to handle large sizes)
        #         chunk_size = 8192
        #         for i in range(0, len(binary_data), chunk_size):
        #             print(datetime.now(), 'writing...')
        #             lo_fd.write(binary_data[i:i + chunk_size])
        #
        #         # Close the large object
        #         lo_fd.close()
        #
        #         # Commit the transaction
        #         connection.commit()
        #
        #         return lo_oid

        async def save_other_entries(self, model_id, session, entry, value):
            if entry == 'historic_predictions_model':
                session.query(Model).filter(Model.model_id == model_id).update({Model.historic_predictions_model: value})
                session.commit()
            if entry == 'historic_norm_test_data':
                session.query(Model).filter(Model.model_id == model_id).update({Model.historic_norm_test_data: value})
                session.commit()
            if entry == 'historic_scores_model':
                session.query(Model).filter(Model.model_id == model_id).update({Model.historic_scores_model: value})
                session.commit()

        async def save_large_binaries(self, model_id, session, model_binary):
            print(timestamp_with_time_zone(), 'saving large binaries')
            # session.query(Model).filter(Model.model_id == model_id).update({Model.model_binary: model_binary})
            # model.model_binary = model_binary
            # session.add(model)
            # session.commit()
            # try:
            #     print(timestamp_with_time_zone(), 'len:', len(model_binary))
            #     session.query(Model).filter(Model.model_id == model_id).update({Model.model_binary: model_binary})
            #     # connection = session.connection().connection
            #     # lo_oid = self.store_large_object(connection, model_binary)
            #     # session.query(Model).filter(Model.model_id == model_id).update({Model.model_binary: lo_oid})
            #     session.commit()
            # except Exception as e:
            #     print(f"{timestamp_with_time_zone()} Error: {e}")
            #     session.rollback()
            #
            # finally:
            #     session.close()

            # conn = create_connection()
            # cur = conn.cursor()
            # cur.execute("SELECT lo_create(0);")
            # oid = cur.fetchone()[0]
            # large_object = conn.lobject(oid, 'wb')
            # large_object.write(model_binary)
            # large_object.close()
            # cur.execute("UPDATE models SET model_binary_oid = %s WHERE model_id = %s", (oid, model_id))
            # conn.commit()
            # cur.close()
            # close_connection(conn)
            conn = create_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT lo_create(0);")
                oid = cursor.fetchone()[0]
                lo = conn.lobject(oid, 'wb')
                lo.write(model_binary)
                lo.close()
                conn.commit()
                # stmt = (update(models_table).where(models_table.c.model_id == model_id).values(model_binary_oid=oid))
                # conn.execute(stmt)
            close_connection(conn)
            model = session.query(Model).filter(Model.model_id == model_id).first()
            if model:
                model.model_binary_oid = oid
                session.commit()


        async def save_result_to_db(self, result_info):
            session = None
            try:
                session = Session()
                result = Result(
                    model_id=result_info['model_id'],
                    input_data=result_info['input_data'],
                    result_values=result_info['result_values'],
                    execution_time=result_info['execution_time'],
                    chosen_model=result_info['chosen_model']
                )
                session.add(result)
                session.commit()
            except Exception as e:
                session.rollback()
                return 'Failed'
            finally:
                session.close()
                return 'Success'

        async def save_part_of_model_to_db(self, id_model, column, value):
            session = None
            try:
                session = Session()
                model = session.query(Model).filter_by(model_id=id_model).first()
                if model:
                    if column == 'part_1':
                        model.model_part1 = value
                    if column == 'part_2':
                        model.model_part2 = value
                    if column == 'part_3':
                        model.model_part3 = value

                    session.commit()
                    model_id = model.model_id
                    session.close()
                    return model_id
                return 'Failed'
            except Exception as e:
                session.rollback()
                return 'Failed'

        async def get_regressor_and_scalers(self, model_id):
            session = Session()
            model = session.query(Model.model_binary_oid,
                                  Model.x_scaler,
                                  Model.y_scaler,
                                  Model.dataset_transformations).filter_by(model_id=model_id).first()
            session.close()
            model_dict = {}
            if model:
                # todo regressor is the oid
                model_dict = {'regressor': pickle.loads(model.model_binary_oid), 'x_scaler': pickle.loads(model.x_scaler),
                              'y_scaler': pickle.loads(model.y_scaler), 'settings': json.loads(
                        model.dataset_transformations) if model.dataset_transformations else {}}
            return json.dumps(model_dict)

        async def get_model_historic_norm_and_version(self, model_id):
            session = Session()
            model = session.query(Model.historic_norm_test_data, Model.models_version).filter_by(
                model_id=model_id).first()
            session.close()
            model_dict = {}
            if model:
                model_dict = {'historic_norm_test_data': json.loads(
                    model.historic_norm_test_data) if model.historic_norm_test_data else {},
                              'model_version': model.model_version}
            return json.dumps(model_dict)

        async def get_models(self):
            session = Session()
            models = session.query(Model.model_id,
                                   Model.model_name,
                                   Model.target_feature,
                                   Model.target_zone,
                                   Model.ml_model,
                                   Model.model_type,
                                   Model.test_errors,
                                   Model.default_metric,
                                   Model.characteristics).order_by(asc(Model.model_id)).all()
            session.close()

            model_list = [{'model_id': model.model_id,
                           'model_name': model.model_name,
                           'target_feature': model.target_feature,
                           'target_zone': model.target_zone,
                           'ml_model': model.ml_mdodel,
                           'model_type': model.model_type,
                           'default_metric': model.default_metric,
                           "test_errors": model.test_errors,
                           "characteristics": model.characteristics}
                          for model in models]
            return json.dumps(model_list)

        async def get_models_to_evaluate(self):
            #todo add target zone
            session = Session()
            models = session.query(Model.model_id,
                                   Model.model_name,
                                   Model.target_feature,
                                   Model.historic_predictions_model,
                                   Model.historic_scores_model,
                                   Model.train_errors).all()
            session.close()

            model_list = [{'model_id': model.model_id,
                           'model_name': model.model_name,
                           'historic_predictions_model': model.historic_predictions_model,
                           'historic_scores_model': model.historic_scores_model,
                           "train_errors": model.train_errors}
                          for model in models]
            return json.dumps(model_list)

        async def update_model_historic(self, model_info):
            model_id = model_info['model_id']
            session = Session()
            session.query(Model).filter(Model.model_id == model_id).update(
                {Model.historic_norm_test_data: json.dumps(model_info['historic_norm_test_data'])})
            session.commit()

            session.query(Model).filter(Model.model_id == model_id).update(
                {Model.historic_predictions_model: json.dumps(model_info['historic_predictions_model'])})
            session.commit()
            # # Retrieve the colour entry based on the given filters
            # model = (
            #     session.query(Model)
            #     .filter_by(model_id=model_id)
            #     .first()
            # )
            #
            # if not model:
            #     session.close()
            #     return 'Failed'
            #
            #
            # # Update the desired fields
            # model.historic_norm_test_data = json.dumps(model_info['historic_norm_test_data'])
            # model.historic_predictions_model = json.dumps(model_info['historic_predictions_model'])
            # # Commit the changes to the database
            # session.commit()

            session.close()
            return 'Success'

        async def update_model_historics(self, model_info):
            model_id = model_info['model_id']
            session = Session()
            if model_info['historic_predictions_model']:
                session.query(Model).filter(Model.model_id == model_id).update(
                    {Model.historic_predictions_model: json.dumps(model_info['historic_predictions_model'])})
                session.commit()

            if model_info['test_errors']:
                session.query(Model).filter(Model.model_id == model_id).update(
                    {Model.test_errors: json.dumps(model_info['test_errors'])})
                session.commit()

            if model_info['historic_scores_model']:
                session.query(Model).filter(Model.model_id == model_id).update(
                    {Model.historic_scores_model: json.dumps(model_info['historic_scores_model'])})
                session.commit()

            # # Retrieve the colour entry based on the given filters
            # model = (
            #     session.query(Model)
            #     .filter_by(model_id=model_id)
            #     .first()
            # )
            #
            # if not model:
            #     session.close()
            #     return 'Failed'
            # commit = False
            # # Update the desired fields
            # if model_info['historic_predictions_model']:
            #     commit = True
            #     model.historic_predictions_model = json.loads(model_info['historic_predictions_model'])
            # if model_info['test_errors']:
            #     commit = True
            #     model.test_errors = json.loads(model_info['test_errors'])
            # if model_info['historic_scores_model']:
            #     commit = True
            #     model.historic_scores_model = json.loads(model_info['historic_scores_model'])
            # # Commit the changes to the database
            # if commit:
            #     session.commit()

            session.close()
            return 'Success'

        async def update_model(self, model_info):
            model_id = model_info['model_id']
            session = Session()

            # Retrieve the colour entry based on the given filters
            model = (
                session.query(Model)
                .filter_by(model_id=model_id)
                .first()
            )

            if not model:
                session.close()
                return 'Failed'

            # Update the desired fields
            #todo : now is oid
            # model.model_binary = pickle.dumps(model_info['model_binary'])
            model.train_data = pickle.dumps(model_info['train_data'])
            model.x_train_data_norm = pickle.dumps(model_info['x_train_data_norm'])
            model.y_train_data_norm = pickle.dumps(model_info['y_train_data_norm'])
            model.x_scaler = pickle.dumps(model_info['x_scaler'])
            model.y_scaler = pickle.dumps(model_info['y_scaler'])
            model.columns_names = model_info['columns_names']
            model.target_feature = model_info['target_feature']
            model.target_zone = model_info['target_zone']
            model.model_name = model_info['model_name']
            model.model_type = model_info['model_type']
            model.model_params = json.dumps(model_info['model_params'])
            model.train_errors = json.dumps(model_info['train_errors'])
            model.notes = json.dumps(model_info['notes'])
            model.dataset_transformations = json.dumps(model_info['dataset_transformations'])
            model.default_metric = model_info['default_metric']
            model.characteristics = json.dumps(model_info['characteristics'])
            model.retrain_counter = model_info['retrain_counter']
            model.flag_training = model_info['flag_training']
            model.models_version = model_info['models_version']
            model.training_dates = json.dumps(model_info['training_dates'])
            # Commit the changes to the database
            session.commit()

            session.close()
            return 'Success'

        async def get_xai_data(self):
            try:
                session = Session()
                models = session.query(Model).filter_by(flag_training=False).all()
                session.close()
                model_list = [{
                        'model_id': model.model_id,
                        # 'model_binary': pickle.loads(model.model_binary),
                        'train_data': pickle.loads(model.train_data),
                        'columns_names': model.columns_names,
                        'target_feature': model.target_feature,
                        'target_zone': model.target_zone,
                        'model_name': model.model_name,
                        'model_type': model.model_type,
                        'ml_model': model.ml_model,
                        'explainer': pickle.loads(model.explainer),
                        'explainer_data': pickle.loads(model.explainer_data),
                        'global_shap_values': pickle.loads(model.global_shap_values),
                        'base_values': model.base_values,
                        'classes': model.classes,
                        'global_explanations': pickle.loads(
                            model.global_explanations) if model.global_explanations else {},
                        'x_train_data_norm': pickle.loads(model.x_train_data_norm) if model.x_train_data_norm else {},
                        'y_train_data_norm': pickle.loads(model.y_train_data_norm) if model.y_train_data_norm else {},
                        'historic_predictions_model': json.loads(model.historic_predictions_model) if model.historic_predictions_model else {},
                        'historic_norm_test_data': pickle.loads(model.historic_norm_test_data) if model.historic_norm_test_data else {},
                        'local_explanations': pickle.loads(model.local_explanations) if model.local_explanations else {}

                    } for model in models]
                return model_list
            except Exception as e:
                raise RuntimeError(f"Error loading model from database: {str(e)}")

        async def load_model_from_db(self, model_id):
            try:
                session = Session()
                model = session.query(Model).filter_by(model_id=model_id).first()
                session.close()
                if model:
                    return {
                        'model_id': model.model_id,
                        # 'model_binary': pickle.loads(model.model_binary),
                        'train_data': pickle.loads(model.train_data),
                        'columns_names': model.columns_names,
                        'target_feature': model.target_feature,
                        'target_zone':model.target_zone,
                        'model_name': model.model_name,
                        'model_type': model.model_type,
                        'explainer': pickle.loads(model.explainer),
                        'explainer_data': pickle.loads(model.explainer_data),
                        'global_shap_values': pickle.loads(model.global_shap_values),
                        'base_values': model.base_values,
                        'classes': model.classes,
                        'global_explanations': pickle.loads(
                            model.global_explanations) if model.global_explanations else {},
                        'local_explanations': pickle.loads(model.local_explanations) if model.local_explanations else {}
                    }
                return None
            except Exception as e:
                raise RuntimeError(f"Error loading model from database: {str(e)}")

        async def get_data_to_retrain(self, model_id):
            try:
                session = Session()
                model = session.query(Model).filter_by(model_id=model_id).first()
                session.close()
                if model:
                    return {
                        'model_id': model.model_id,
                        'historic_predictions_model': json.loads(model.historic_predictions_model) if model.historic_predictions_model else {},
                        'target_feature': model.target_feature,
                        'target_zone': model.target_zone,
                        'ml_model': model.ml_model,
                        'characteristics': json.loads(model.characteristics) if model.characteristics else {},
                        'historic_scores_model': json.loads(
                            model.historic_scores_model) if model.historic_scores_model else {},
                        'registered_at': model.registered_at,
                        'training_dates': json.loads(model.training_dates) if model.training_dates else {},
                    }
                return {}
            except Exception as e:
                raise RuntimeError(f"Error loading model from database: {str(e)}")

        async def check_if_model_exists(self, model_info):
            session = Session()
            target = model_info['target']
            target_zone = model_info['target_table']
            models = session.query(Model.characteristics).filter_by(target_feature=target, target_zone=target_zone).all()
            session.close()
            for model in models:
                for model_characteristics in model:
                    if model_characteristics:
                        exists = True
                        if not isinstance(model_characteristics, dict):
                            model_characteristics = json.loads(model_characteristics)
                        # if model_characteristics.get('target_table') != model_info['target_table']:
                        #     exists = False
                        if model_characteristics.get('dataset_type') != model_info['dataset_type']:
                            exists = False
                        if model_characteristics.get('frequency') != model_info['frequency']:
                            exists = False
                        if exists:
                            return True
            return False

        async def print_model(self):
            msg = 'ok'
            session = Session()
            model = session.query(Model).filter_by(target_name='cons_total', ml_model='RFR').first()
            session.close()
            oid = model.model_binary_oid
            print(timestamp_with_time_zone(),'oid', oid)
            # print(timestamp_with_time_zone(), 'model:', model)
            # try:
            #     model_binary = model.model_binary
            #     decom_data = zlib.decompress(model_binary)
            #     print(type(decom_data), len(decom_data))
            #     regr = pickle.loads(decom_data)
            #     print(regr, len(regr))
            # except Exception as e:
            #     print(e)
            #     decom_data = zlib.decompress(model)
            #     print(type(decom_data), len(decom_data))
            #     regr = pickle.loads(decom_data)
            #     print(regr, len(regr))
            # Connect to the PostgreSQL database
            conn = create_connection()
            cur = conn.cursor()

            #cur.execute("SELECT model_binary_oid FROM models WHERE target_name='cons_zone1'and ml_model='RFR'")
            #oid = cur.fetchone()[0]

            # Open the large object for reading
            large_object = conn.lobject(oid, 'rb')
            compressed_model_data = large_object.read()
            large_object.close()

            # Decompress the binary model data
            model_data = zlib.decompress(compressed_model_data)

            print(timestamp_with_time_zone(), 'got_model_data:', type(model_data), len(model_data))
            try:
                regressor = pickle.loads(model_data)
                if isinstance(regressor, bytes):
                    print(timestamp_with_time_zone(),'try to load bytes')
                    try:
                        regressor = pickle.loads(regressor)
                        msg = str(timestamp_with_time_zone()) + '_ regressor ' + str(regressor)
                        print(timestamp_with_time_zone(),'regressor', regressor)
                    except Exception as e:
                        print(e)
                        pass
                else:
                    print(timestamp_with_time_zone(), 'type is', type(regressor))
            except:
                pass
            cur.close()
            conn.close()
            return msg

        async def load_model_binary(self, id):
            regressor = None
            session = Session()
            model = session.query(Model).filter_by(model_id=id).first()
            session.close()
            oid = model.model_binary_oid
            conn = create_connection()
            cur = conn.cursor()
            # Open the large object for reading
            large_object = conn.lobject(oid, 'rb')
            compressed_model_data = large_object.read()
            large_object.close()
            # Decompress the binary model data
            model_data = zlib.decompress(compressed_model_data)
            try:
                regressor = pickle.loads(model_data)
                regressor = pickle.loads(regressor)
            except:
                pass
            cur.close()
            conn.close()
            return regressor

    async def setup(self):
        await start_app()
        self.chunked_messages = defaultdict(dict)  # Thread-safe chunk storage
        self.add_behaviour(self.ReceiveMsg())

    # async def clean_up_chunks(self):
    #     while True:
    #         await asyncio.sleep(30)  # Clean every 30 seconds
    #         current_time = datetime.now()
    #         for message_id, data in list(self.chunked_messages.items()):
    #             if (current_time - data.get("timestamp", current_time)).total_seconds() > 180:  # Timeout of 3 minutes
    #                 print(f" {timestamp_with_time_zone()} Cleaning up incomplete message {message_id}")
    #                 del self.chunked_messages[message_id]