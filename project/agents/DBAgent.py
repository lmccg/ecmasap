from peak import Agent, Message, PeriodicBehaviour, CyclicBehaviour
import json
import pickle
from ..utils_package.repository import Session, Model, Result
from sqlalchemy import func, asc


class DBAgent(Agent):
    class ReceiveMsg(PeriodicBehaviour):
        async def run(self):
            msg = await self.receive(10)
            if msg:
                print(f"model: {msg.sender} sent me a message: '{msg.body}'")
                if 'save_model' in msg.body:
                    data_to_insert_in_database = json.loads(msg.body)
                    data_to_insert_in_database = data_to_insert_in_database['save_model']
                    response_db = await self.save_model_to_db(data_to_insert_in_database)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    if response_db:
                        response_msg.body = "success"
                    else:
                        response_msg.body = "failure"
                #     await self.send(response_msg)
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
                elif 'save_result' in msg.body:
                    data_to_insert_in_database = json.loads(msg.body)
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

                elif 'get_regressor_and_scalers' in msg.body:
                    info = json.loads(msg.body)
                    model_id = info['get_regressor_and_scalers']
                    response_db = await self.get_regressor_and_scalers(model_id)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)
                elif 'update_model' in msg.body:
                    info = json.loads(msg.body)
                    model_info = info['update_model']
                    response_db = await self.update_model(model_info)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)

                elif 'update_model_historic_norm' in msg.body:
                    info = json.loads(msg.body)
                    model_info = info['update_model_historic_norm']
                    response_db = await self.update_model_historic_norm(model_info)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)

                elif 'update_model_historics' in msg.body:
                    info = json.loads(msg.body)
                    model_info = info['update_model_historics']
                    response_db = await self.update_model_historics(model_info)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)

                elif 'get_model_historic_norm_and_version' in msg.body:
                    info = json.loads(msg.body)
                    model_info = info['update_model_historic_norm']
                    response_db = await self.get_model_historic_norm_and_version(model_info)
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)

                elif 'get_models' in msg.body:
                    response_db = await self.get_models()
                    # REPLY BACK
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = response_db
                    await self.send(response_msg)


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
                session = Session()
                model = session.query(Model).filter_by(model_id=model_info['model_id']).first()
                if model:
                    model.model_binary = pickle.dumps(model_info['model_binary'])
                    model.train_data = pickle.dumps(model_info['train_data'])
                    model.x_train_data_norm = pickle.dumps(model_info['x_train_data_norm'])
                    model.y_train_data_norm = pickle.dumps(model_info['y_train_data_norm'])
                    model.x_scaler = pickle.dumps(model_info['x_scaler'])
                    model.y_scaler = pickle.dumps(model_info['y_scaler'])
                    model.columns_names = model_info['columns_names']
                    model.target_name = model_info['target_name']
                    model.model_name = model_info['model_name']
                    model.model_type = model_info['model_type']
                    model.model_params = json.dumps(model_info['model_params'])
                    model.test_errors = json.dumps(model_info['test_errors'])
                    model.train_errors = json.dumps(model_info['train_errors'])
                    model.notes = json.dumps(model_info['notes'])
                    model.dataset_transformations = json.dumps(model_info['dataset_transformations'])
                    model.default_metric = model_info['default_metric']
                    model.characteristics = json.dumps(model_info['characteristics'])
                    # model.explainer = pickle.dumps(model_info['explainer'])
                    # model.explainer_data = pickle.dumps(model_info['explainer_data'])
                    # model.global_shap_values = pickle.dumps(self.extract_shap_data(model_info['global_shap_values']))
                    # model.base_values = model_info['base_values']
                    # model.classes = model_info['classes']
                    # model.global_explanations = pickle.dumps(model_info.get('global_explanations', {}))
                    # model.local_explanations = pickle.dumps(model_info.get('local_explanations', {}))
                    model.historic_predictions_model = json.dumps(model_info['historic_predictions_model'])
                    model.historic_scores_model = json.dumps(model_info['historic_scores_model'])
                    model.historic_norm_test_data = json.dumps(model_info['historic_norm_test_data'])
                    model.retrain_counter = model_info['retrain_counter']
                    model.flag_training = model_info['flag_training']
                    model.models_version = model_info['models_version']
                    model.training_dates = json.dumps(model_info['training_dates'])
                    model.updated_at = func.now()
                else:
                    model = Model(
                        model_id=model_info['model_id'],
                        model_binary=pickle.dumps(model_info['model_binary']),
                        train_data=pickle.dumps(model_info['train_data']),
                        x_train_data_norm=pickle.dumps(model_info['x_train_data_norm']),
                        y_train_data_norm=pickle.dumps(model_info['y_train_data_norm']),
                        x_scaler=pickle.dumps(model_info['x_scaler']),
                        y_scaler=pickle.dumps(model_info['y_scaler']),
                        columns_names=model_info['columns_names'],
                        target_name=model_info['target_name'],
                        model_name=model_info['model_name'],
                        model_type=model_info['model_type'],
                        model_params=json.dumps(model_info['model_params']),
                        test_errors=json.dumps(model_info['test_errors']),
                        train_errors=json.dumps(model_info['train_errors']),
                        notes=json.dumps(model_info['notes']),
                        dataset_transformations=json.dumps(model_info['dataset_transformations']),
                        default_metric=model_info['default_metric'],
                        characteristics=json.dumps(model_info['characteristics']),
                        # explainer = pickle.dumps(model_info['explainer'])
                        # explainer_data = pickle.dumps(model_info['explainer_data'])
                        # global_shap_values = pickle.dumps(self.extract_shap_data(model_info['global_shap_values']))
                        # base_values = model_info['base_values']
                        # classes = model_info['classes']
                        # global_explanations = pickle.dumps(model_info.get('global_explanations', {}))
                        # local_explanations = pickle.dumps(model_info.get('local_explanations', {}))
                        historic_predictions_model=json.dumps(model_info['historic_predictions_model']),
                        historic_scores_model=json.dumps(model_info['historic_scores_model']),
                        historic_norm_test_data=json.dumps(model_info['historic_norm_test_data']),
                        retrain_counter=model_info['retrain_counter'],
                        flag_training=model_info['flag_training'],
                        models_version=model_info['models_version'],
                        training_dates=json.dumps(model_info['training_dates']),
                        registered_at=func.now(),
                        updated_at=func.now()
                    )
                    session.add(model)
                session.commit()
            except Exception as e:
                session.rollback()
                return 'Failed'
            finally:
                session.close()
                return 'Success'

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


        async def get_regressor_and_scalers(self, model_id):
            session = Session()
            model = session.query(Model.model_binary,
                                  Model.x_scaler,
                                  Model.y_scaler,
                                  Model.dataset_transformations).filter_by(model_id=model_id).first()
            session.close()
            model_dict = {}
            if model:
                model_dict = {'regressor': pickle.loads(model.model_binary), 'x_scaler': pickle.loads(model.x_scaler), 'y_scaler': pickle.loads(model.y_scaler), 'settings': json.loads(model.dataset_transformations) if model.dataset_transformations else {}}
            return json.dumps(model_dict)

        async def get_model_historic_norm_and_version(self, model_id):
            session = Session()
            model = session.query(Model.historic_norm_test_data, Model.models_version).filter_by(model_id=model_id).first()
            session.close()
            model_dict = {}
            if model:
                model_dict = {'historic_norm_test_data': json.loads(model.historic_norm_test_data) if model.historic_norm_test_data else {}, 'model_version': model.model_version}
            return json.dumps(model_dict)

        async def get_models(self):
            session = Session()
            models = session.query(Model.model_id,
                                   Model.model_name,
                                   Model.target_name,
                                   Model.ml_model,
                                   Model.model_type,
                                   Model.test_errors,
                                   Model.default_metric,
                                   Model.characteristics).order_by(asc(Model.model_id)).all()
            session.close()

            model_list = [{'model_id': model.model_id,
                           'model_name': model.model_name,
                           'target_name': model.target_name,
                           'ml_model': model.ml_mdodel,
                           'model_type': model.model_type,
                           'default_metric': model.default_metric,
                           "test_errors": model.test_errors,
                           "characteristics": model.characteristics}
                          for model in models]
            return json.dumps(model_list)

        async def update_model_historic_norm(self, model_info):
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
            model.historic_norm_test_data = json.dumps(model_info['historic_norm_test_data'])
            # Commit the changes to the database
            session.commit()

            session.close()
            return 'Success'

        async def update_model_historics(self, model_info):
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
            model.historic_predictions_model = json.dumps(model_info['historic_predictions_model'])
            model.historic_scores_model = json.dumps(model_info['historic_scores_model'])
            # Commit the changes to the database
            session.commit()

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
            model.model_binary = pickle.dumps(model_info['model_binary'])
            model.train_data = pickle.dumps(model_info['train_data'])
            model.x_train_data_norm = pickle.dumps(model_info['x_train_data_norm'])
            model.y_train_data_norm = pickle.dumps(model_info['y_train_data_norm'])
            model.x_scaler = pickle.dumps(model_info['x_scaler'])
            model.y_scaler = pickle.dumps(model_info['y_scaler'])
            model.columns_names = model_info['columns_names']
            model.target_name = model_info['target_name']
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

        async def load_model_from_db(self, model_id):
            try:
                session = Session()
                model = session.query(Model).filter_by(model_id=model_id).first()
                session.close()
                if model:
                    return {
                        'model_id': model.model_id,
                        'model_binary': pickle.loads(model.model_binary),
                        'train_data': pickle.loads(model.train_data),
                        'columns_names': model.columns_names,
                        'target_name': model.target_name,
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

    async def setup(self):
        self.add_behaviour(self.ReceiveMsg(period=1))
