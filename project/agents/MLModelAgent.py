from peak import Agent, Message, PeriodicBehaviour, CyclicBehaviour
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, KFold, cross_val_score, cross_val_predict, \
    cross_validate
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import time


class MLModelAgent(Agent):
    class ReceiveMsg(PeriodicBehaviour):
        async def run(self):
            msg = await self.receive(10)
            if msg:
                error_ = False
                print(f"model: {msg.sender} sent me a message: '{msg.body}'")
                with open('../utils_package/config_params.json') as config_file:
                    config_params = json.load(config_file)
                with open('../utils_package/config_settings.json') as config_file:
                    config_settings = json.load(config_file)
                with open('../utils_package/tablesData.json') as config_file:
                    tables_dataset = json.load(config_file)
                with open('../utils_package/config_agents.json') as config_file:
                    config_agents = json.load(config_file)
                database_agent = config_agents["database_agent"]
                if 'train' in msg.body or 'retrain' in msg.body:
                    try:
                        parts_of_msg = msg.body.split("|")
                        ml_model_name = parts_of_msg[1]
                        input = parts_of_msg[2]
                        input = json.loads(input)
                        target = input['target']
                        target_table = input['target_table']
                        frequency = input['frequency']
                        dataset_type = input['dataset_type']
                        training_dates = input['training_dates']
                        dataset_ja = parts_of_msg[3]
                        table_target = input['target_table']
                        ml_model_type = next((key for key, models in config_agents.get("ml_model_agents").items() if ml_model_name in models.values()), None)
                        key_params = target_table + '_' + target + '_' + str(frequency)
                        ml_model_parameters = config_params.get(key_params).get(dataset_type).get(ml_model_name).get('params')
                        metric_score = config_settings.get('error_metric')
                        settings = input.get('settings')
                        if not isinstance(settings, dict):
                            settings = json.loads(settings)
                        characteristics = input.get('characteristics')
                        array_data_df, X_train, y_train, x_scale, y_scale, final_columns_names = await self.__generate_train_data(dataset_ja, settings, tables_dataset, target_table, target, dataset_type)
                        if ml_model_name == "MLPR":
                            regressor = MLPRegressor(**ml_model_parameters)
                        elif ml_model_name == "SVR":
                            regressor = SVR(**ml_model_parameters)
                        elif ml_model_name == "KNNR":
                            regressor = KNeighborsRegressor(**ml_model_parameters)
                        elif ml_model_name == "RFR":
                            regressor = RandomForestRegressor(**ml_model_parameters)
                        elif ml_model_name == "MLPC":
                            regressor = MLPClassifier(**ml_model_parameters)
                        elif ml_model_name == "SVC":
                            regressor = SVC(**ml_model_parameters)
                        elif ml_model_name == "KNNC":
                            regressor = KNeighborsClassifier(**ml_model_parameters)
                        elif ml_model_name == "RFC":
                            regressor = RandomForestClassifier(**ml_model_parameters)
                        elif ml_model_name == "GBR":
                            regressor = GradientBoostingRegressor(**ml_model_parameters)
                        elif ml_model_name == "ABR":
                            regressor = AdaBoostRegressor(**ml_model_parameters)
                        else:
                            error_ = True
                            result = "Failed"
                    except:
                        error_ = True
                        result = "Failed"

                    if not error_:
                        # Storing the fit object for later reference
                        regressor.fit(X_train, y_train)
                        current_time = datetime.now()
                        formated_timestamp = config_settings.get('datetime_format')
                        trained_at = current_time.strftime(formated_timestamp)
                        method = TimeSeriesSplit(n_splits=2)
                        scores = cross_val_score(regressor, X_train, y_train, cv=method, scoring=metric_score)
                        scores_ = {metric_score:scores}
                        # save the data from database
                        new_msg = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                        new_msg.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                        ml_model_name_ = ml_model_name + '_' + str(
                            frequency) + 'min' + '_' + target_table + '_' + target + '_' + dataset_type
                        if 'train' in msg.body:
                            retrain_counter = 0
                            models_version = "v_" + str(retrain_counter) + "_" + trained_at
                            models_version = models_version.replace(" ", "_")
                            ml_model_info = {
                                'model_binary': regressor,
                                'train_data': array_data_df,
                                'x_train_data_norm': X_train,
                                'y_train_data_norm': y_train,
                                'x_scaler': x_scale,
                                'y_scaler': y_scale,
                                'columns_names': final_columns_names,
                                'target_name': target,
                                'model_name': ml_model_name_,
                                'ml_model': ml_model_name,
                                'model_type': ml_model_type,
                                'model_params': ml_model_parameters,
                                'train_errors': scores_,
                                'test_errors': scores_,
                                'notes': {'note': 'Trained model'},
                                'dataset_transformations': settings,
                                'default_metric': metric_score,
                                'characteristics': characteristics,
                                'retrain_counter': retrain_counter,
                                'flag_training': False,
                                'models_version': models_version,
                                'training_dates': training_dates}
                            msg_ = {'save_model': ml_model_info}
                            msg_ = json.dumps(msg_)
                            new_msg.body = msg_
                            await self.send(new_msg)
                            response = await self.receive(timeout=60)
                            if response:
                                if response.get_metadata("performative") == "inform":
                                    result = response.body
                            else:
                                error_ = True
                                result = "Failed"
                        elif 'retrain' in msg.body:
                            # get the data from database
                            try:
                                ml_model_id = parts_of_msg[4]
                                request_counter = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                                request_counter.set_metadata("performative", "request")
                                msg_ = {'get_retrain_counter': ml_model_id}
                                msg_ = json.dumps(msg_)
                                request_counter.body = msg_
                                await self.send(request_counter)
                                response = await self.receive(timeout=60)
                                if response:
                                    if response.get_metadata("performative") == "inform":
                                        retrain_counter = response.body
                                else:
                                    error_ = True
                                    result = "Failed"
                            except:
                                error_ = True
                                result = "Failed"
                            if not error_:
                                retrain_counter = int(retrain_counter)
                                retrain_counter += 1
                                models_version = "v_" + str(retrain_counter) + "_" + trained_at
                                models_version = models_version.replace(" ", "_")
                                ml_model_info = {
                                    'model_id': ml_model_id,
                                    'model_binary': regressor,
                                    'train_data': array_data_df,
                                    'x_train_data_norm': X_train,
                                    'y_train_data_norm': y_train,
                                    'x_scaler': x_scale,
                                    'y_scaler': y_scale,
                                    'columns_names': final_columns_names,
                                    'target_name': target,
                                    'model_name': ml_model_name_,
                                    'ml_model': ml_model_name,
                                    'model_type': ml_model_type,
                                    'model_params': ml_model_parameters,
                                    'train_errors': scores_,
                                    'test_errors': scores_,
                                    'notes': {'note': 'Trained model'},
                                    'dataset_transformations': settings,
                                    'default_metric': metric_score,
                                    'characteristics': characteristics,
                                    'retrain_counter': retrain_counter,
                                    'flag_training': False,
                                    'models_version': models_version,
                                    'training_dates': training_dates}
                                msg_ = {'update_model': ml_model_info}
                                msg_ = json.dumps(msg_)
                                new_msg.body = msg_
                                await self.send(new_msg)
                                response = await self.receive(timeout=60)
                                if response:
                                    if response.get_metadata("performative") == "inform":
                                        result = response.body
                                else:
                                    error_ = True
                                    result = "Failed"

                elif 'predict' in msg.body:
                    try:
                        parts_of_msg = msg.body.split("|")
                        ml_model_id = parts_of_msg[1]
                        input = parts_of_msg[2]
                        input = json.loads(input)
                        target = input['target']
                        target_table = input['target_table']
                        dataset_type = input['dataset_type']
                        first_start_timestamp = input['first_start_timestamp']
                        last_start_timestamp = input['last_end_timestamp']
                        input_data_result = {'start_at': input['start_date'],
                                             'start_time': input['start_time'],
                                             'end_date': input['end_date'],
                                             'end_time': input['end_time'],
                                             'frequency': input['frequency'],
                                             'target': target,
                                             'target_table': target_table,
                                             'dataset_type': dataset_type}
                        dataset_ja = parts_of_msg[3]
                        background_models = parts_of_msg[4]
                        if str(background_models).lower() == 'true':
                            chosen_model = False
                        else:
                            chosen_model = True
                        # get the data from database
                        msg_train_data = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                        msg_train_data.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                        msg_ = {'get_regressor_and_scalers': ml_model_id}
                        msg_ = json.dumps(msg_)
                        msg_train_data.body = msg_
                        await self.send(msg_train_data)
                        response = await self.receive(timeout=60)
                        if response:
                            if response.get_metadata("performative") == "inform":
                                info_from_database = response.body
                        else:
                            error_ = True
                            result = "Failed"
                        if not error_:
                            info_from_database = json.loads(info_from_database)
                            if info_from_database:
                                regressor = info_from_database['regressor']
                                x_scaler = info_from_database['x_scaler']
                                y_scaler = info_from_database['y_scaler']
                                settings_jo = info_from_database['settings']
                            else:
                                error_ = True
                                result = "Failed"
                            msg_historical_norm_and_versions = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                            msg_historical_norm_and_versions.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                            msg_ = {'get_model_historic_norm_and_version': ml_model_id}
                            msg_ = json.dumps(msg_)
                            msg_historical_norm_and_versions.body = msg_
                            await self.send(msg_historical_norm_and_versions)
                            response = await self.receive(timeout=60)
                            if response:
                                if response.get_metadata("performative") == "inform":
                                    info_norm_and_version = response.body
                            else:
                                error_ = True
                                result = "Failed"
                            if not error_:
                                historic_norm_test_data = info_norm_and_version['historic_norm_test_data']
                                model_version = info_norm_and_version['model_version']

                                X_test = await self.__generate_test_data(dataset_ja, x_scaler, y_scaler, settings_jo, tables_dataset,  target_table, target, dataset_type)
                                start_ = time.time()
                                predictions = regressor.predict(X_test)
                                end_ = time.time()
                                execution_time = end_ - start_
                                if y_scaler:
                                    inv_predict = y_scaler.inverse_transform(predictions.reshape(-1, 1))
                                else:
                                    inv_predict = predictions.reshape(-1, 1)
                                entry_norm_test_data = {
                                                      "predicted_timestamp": first_start_timestamp,
                                                      "predicted_end_timestamp": last_start_timestamp,
                                                      "normalized_x_test": X_test,
                                                      "normalized_y_test": predictions,
                                                      "y_predicted": inv_predict,
                                                      "model_version": model_version}
                                historic_norm_test_data.append(entry_norm_test_data)
                                ml_model_info = {
                                    'model_id': ml_model_id,
                                    'historic_norm_test_data': historic_norm_test_data,
                                }
                                msg_update_model_historic = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                                msg_update_model_historic.set_metadata("performative", "request")  # Set the "inform" FIPA performative

                                msg_ = {'update_model_historic_norm': ml_model_info}
                                msg_ = json.dumps(msg_)
                                msg_update_model_historic.body = msg_
                                await self.send(msg_update_model_historic)
                                response = await self.receive(timeout=60)
                                if response:
                                    if response.get_metadata("performative") == "inform":
                                        #save model result in database
                                        msg_save_result = Message(
                                            to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                                        msg_save_result.set_metadata("performative",
                                                             "request")  # Set the "inform" FIPA performative

                                        result_info = {
                                            'model_id': ml_model_id,
                                            'input_data': input_data_result,
                                            'result_values': inv_predict,
                                            'execution_time': execution_time,
                                            'chosen_model': chosen_model}
                                        msg_ = {'save_result': result_info}
                                        msg_ = json.dumps(msg_)
                                        msg_save_result.body = msg_
                                        await self.send(msg_save_result)
                                        response = await self.receive(timeout=60)
                                        if response:
                                            if response.get_metadata("performative") == "inform":
                                                result = {target: inv_predict}

                                        else:
                                            error_ = True
                                            result = "Failed"

                                else:
                                    error_ = True
                                    result = "Failed"
                    except:
                        error_ = True
                        result = "Failed"

                print('sending back')
                if result != "Failed":
                    publisher_agent = config_agents["publisher_agent"]
                    request_publish = Message(to=f"{publisher_agent}@{self.agent.jid.domain}/{publisher_agent}")
                    request_publish.set_metadata("performative", "inform")
                    msg_ = json.dumps(result)
                    request_publish.body = msg_
                    await self.send(request_publish)
                await self.send_reply(msg, result)

        async def __generate_train_data(self, dataset_ja, settings_jo, tables_dataset, target_table, target, dataset_type):
            dataset_ja = json.loads(dataset_ja)
            datetime_column_name = settings_jo["datetime_column_name"]
            datetime_format = settings_jo["datetime_format"]
            target_name = settings_jo["target_column_name"]
            columns_for_target = []
            data_column = tables_dataset.get(dataset_type).get(target_table).get(target)
            for entry in data_column:
                if isinstance(entry, dict):
                    for k, v in entry.items():
                        for value in v:
                            columns_for_target.append(k + "_" + value)

                else:
                    if entry != target:
                        columns_for_target.append(target + "_" + entry)
                    else:
                        columns_for_target.append(target)
            categorical_columns_names = settings_jo["categorical_columns_names"]
            columns_names = settings_jo["columns_names"]
            normalize = settings_jo["normalize"]
            transformations = settings_jo["transformations"]
            filters = settings_jo["filters"]
            data_df = pd.DataFrame(dataset_ja)
            data_df.columns = columns_names
            data_df[datetime_column_name] = pd.to_datetime(data_df[datetime_column_name], format=datetime_format)
            datetime_cols_in_df = [datetime_column_name]
            data_df.set_index(datetime_column_name, inplace=True)
            categorical_columns_names = await self.__check_for_more_categorical_cols(categorical_columns_names,
                                                                                transformations["previous_periods"],
                                                                                transformations,
                                                                                filters["problem_type"])
            try:
                data_df, datetime_cols_in_df = await self.__transform_data(data_df, transformations, target_name,
                                                                      datetime_cols_in_df, columns_for_target)
            except Exception as e:
                return ({"error": "Transforming data: " + str(e)})

            try:
                data_df = await self. __extract_categorical_data(data_df, categorical_columns_names)
            except Exception as e:
                raise Exception({"error": "Extracting categorical data: " + str(e)})

            try:
                data_df = await self.__filter_data(data_df, filters)
            except Exception as e:
                raise Exception({"error": "Filtering data: " + str(e)})

            if data_df.empty:
                raise Exception({"error": "Empty dataset"})

            x_std_scale = None
            y_std_scale = None
            array_data_df =data_df.values.tolist()
            if normalize:
                try:
                    normalized_data_df, x_std_scale, y_std_scale = await self.__normalize_data(data_df, target_name,
                                                                                              filters["problem_type"])
                except Exception as e:
                    raise Exception({"error": "Normalizing data: " + str(e)})
            else:
                normalized_data_df = data_df
            x_train, y_train, final_columns_names = await self.__split_x_y(normalized_data_df, target_name)
            return array_data_df, x_train, y_train, x_std_scale, y_std_scale, final_columns_names
        async def __generate_test_data(self, dataset_ja, x_scale, y_scale, settings_jo, tables_dataset, target_table, target, dataset_type):
            dataset_ja = json.loads(dataset_ja)
            datetime_column_name = settings_jo["datetime_column_name"]
            datetime_format = settings_jo["datetime_format"]
            target_name = settings_jo["target_column_name"]
            columns_for_target = []
            data_column = tables_dataset.get(dataset_type).get(target_table).get(target)
            for entry in data_column:
                if isinstance(entry, dict):
                    for k, v in entry.items():
                        for value in v:
                            columns_for_target.append(k + "_" + value)

                else:
                    if entry != target:
                        columns_for_target.append(target + "_" + entry)
                    else:
                        columns_for_target.append(target)
            categorical_columns_names = settings_jo["categorical_columns_names"]
            columns_names = settings_jo["columns_names"]
            normalize = settings_jo["normalize"]
            transformations = settings_jo["transformations"]
            filters = settings_jo["filters"]
            data_df = pd.DataFrame(dataset_ja)
            data_df.columns = columns_names
            data_df[datetime_column_name] = pd.to_datetime(data_df[datetime_column_name], format=datetime_format)
            datetime_cols_in_df = [datetime_column_name]
            data_df.set_index(datetime_column_name, inplace=True)
            categorical_columns_names = await self.__check_for_more_categorical_cols(categorical_columns_names,
                                                                                transformations["previous_periods"],
                                                                                transformations,
                                                                                filters["problem_type"])
            try:
                data_df, datetime_cols_in_df = await self.__transform_data(data_df, transformations, target_name,
                                                                      datetime_cols_in_df, columns_for_target)
            except Exception as e:
                return ({"error": "Transforming data: " + str(e)})

            try:
                data_df = await self.__extract_categorical_data_test(data_df, categorical_columns_names,
                                                                    settings_jo["final_columns_names"])
            except Exception as e:
                raise Exception({"error": "Extracting categorical data: " + str(e)})

            try:
                data_df = await self.__filter_data(data_df, filters)
            except Exception as e:
                raise Exception({"error": "Filtering data: " + str(e)})

            if data_df.empty:
                raise Exception({"error": "Empty dataset"})

            if normalize:
                try:
                    normalized_data_df = await self.__normalize_data_test(data_df, target_name, x_scale, y_scale)
                except Exception as e:
                    raise Exception({"error": "Normalizing data: " + str(e)})
            else:
                normalized_data_df = data_df
            x_test, y_test, final_columns_names = await self.__split_x_y(normalized_data_df, target_name)
            return x_test

        async def __extract_categorical_data(self, data_df, columns_names_list):
            for column in columns_names_list:
                try:
                    array = pd.get_dummies(data_df[column], prefix=column)
                    enc_df = pd.DataFrame(array)
                    data_df = data_df.join(enc_df)
                    data_df = data_df.drop(columns=[column])
                except Exception as e:
                    raise Exception(e)

            return data_df

        async def send_reply(self, msg, result):
            response_msg = msg.make_reply()
            response_msg.set_metadata("performative", "inform")
            response_msg.body = json.dumps(result)
            await self.send(response_msg)

        async def __extract_categorical_data_test(self, data_df, categorical_columns_names, all_columns_names):
            for cat_col in categorical_columns_names:
                if cat_col in data_df.columns:
                    categorical_column_dummies = []
                    for col in all_columns_names:
                        c = str(col)
                        if c.startswith(cat_col + "_"):
                            splited = c.split(cat_col + "_")
                            categorical_column_dummies.append(splited[1])

                    try:
                        array = pd.get_dummies(
                            data_df[cat_col].astype(pd.CategoricalDtype(categories=categorical_column_dummies)),
                            prefix=cat_col)
                        enc_df = pd.DataFrame(array)
                        data_df = data_df.join(enc_df)
                        data_df = data_df.drop(columns=[cat_col])
                    except Exception as e:
                        return Exception(e)

            return data_df

        async def __normalize_data(self, data_df, target_column, problemType):
            try:
                x_cols = list(data_df.columns)
                x_cols.remove(target_column)
                x_std_scale = StandardScaler().fit(data_df[x_cols])
                data_df[x_cols] = x_std_scale.transform(data_df[x_cols])
                if problemType.lower() == 'classification':
                    y_std_scale = None
                else:
                    y_std_scale = StandardScaler().fit(data_df[[target_column]])
                    data_df[[target_column]] = y_std_scale.transform(data_df[[target_column]])
            except Exception as e:
                raise Exception(e)

            return data_df, x_std_scale, y_std_scale

        async def __normalize_data_test(self, data_df, target_column, x_std_scale, y_std_scale):
            try:
                x_cols = list(data_df.columns)
                x_cols.remove(target_column)

                data_df[x_cols] = x_std_scale.transform(data_df[x_cols])
                if y_std_scale is not None:
                    data_df[[target_column]] = y_std_scale.transform(data_df[[target_column]])
            except Exception as e:
                raise Exception(e)

            return data_df

        async def __split_x_y(self, data_df, target_name):
            try:
                x_train = data_df.drop(columns=[target_name])
                y_train = data_df[target_name]

                cols = list(x_train.columns)

                x_train = x_train.values.tolist()
                y_train = y_train.values.tolist()
            except Exception as e:
                raise Exception(e)

            return x_train, y_train, cols

        async def __selecting_test(self, data_df, start_date, end_date):
            try:
                mask = (data_df.index >= start_date) & (data_df.index <= end_date)
                data_df = data_df.loc[mask]
            except Exception as e:
                raise Exception(e)
            return data_df

        async def __transform_data(self, data_df, transformations, target_name, datetime_cols_in_df, columns_for_target):
            for key in transformations.keys():
                apply = transformations[key]
                if key == "split_datetime":
                    if apply:
                        data_df['month'] = data_df.index.month
                        data_df['day'] = data_df.index.day
                        data_df['hour'] = data_df.index.hour
                        data_df['minute'] = data_df.index.minute
                        datetime_cols_in_df.append('month')
                        datetime_cols_in_df.append('day')
                        datetime_cols_in_df.append('hour')
                        datetime_cols_in_df.append('minute')

                if key == "day_of_week":
                    if apply:
                        day_names = data_df.index.day_name()
                        # create the day of week column
                        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                            data_df['day_of_week_' + day] = 0
                        # update if exists
                        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                            data_df.loc[day_names == day, 'day_of_week_' + day] = 1

                if key == "weekend":
                    if apply:
                        data_df['weekend'] = data_df.index.dayofweek >= 5

                if key == "trimester":
                    if apply:
                        trimesters = data_df.index.quarter
                        # create trimester
                        for trimester in [1, 2, 3, 4]:
                            data_df['trimester_' + str(trimester)] = 0
                        # update if exists
                        for trimester in [1, 2, 3, 4]:
                            data_df.loc[trimesters == trimester, 'trimester_' + str(trimester)] = 1
                if key == "week_of_year":
                    if apply:
                        data_df['week_of_year'] = data_df.index.isocalendar().week

                        array = pd.get_dummies(data_df['week_of_year'], prefix='week_of_year')
                        enc_df = pd.DataFrame(array)
                        data_df = data_df.join(enc_df)
                        data_df = data_df.drop(columns=['week_of_year'])

                if key == "day_of_year":
                    if apply:
                        data_df['day_of_year'] = data_df.index.dayofyear
                        array = pd.get_dummies(data_df['day_of_year'], prefix='day_of_year')
                        enc_df = pd.DataFrame(array)
                        data_df = data_df.join(enc_df)
                        data_df = data_df.drop(columns=['day_of_year'])

                if key == "previous_periods":
                    if apply:
                        n_previous_periods_list = transformations["number_previous_periods"]
                        for i in range(0, len(n_previous_periods_list)):
                            data_df['t-' + str(n_previous_periods_list[i])] = data_df[target_name].shift(
                                n_previous_periods_list[i])
            for column in data_df.columns:
                if column not in columns_for_target:
                    data_df.drop(columns=[column])
            data_df = data_df.dropna()
            return data_df, datetime_cols_in_df

        async def __filter_data(self, data_df, filters):
            filters_keys = filters.keys()

            for key in filters_keys:

                if key == "day_of_week":
                    days_to_keep = filters[key]

                    if not days_to_keep == []:
                        days = [0, 1, 2, 3, 4, 5, 6]
                        days_to_remove = list(set(days) - set(days_to_keep))

                        for day in days_to_remove:
                            data_df.drop(data_df[data_df.index.dayofweek == day].index, inplace=True)

                if key == "exclude_weekend":
                    if filters[key]:
                        data_df.drop(data_df[data_df.index.dayofweek >= 5].index, inplace=True)

                if key == "dataset_periods":
                    dataset_type = filters[key]
                    if dataset_type == "working_periods":
                        data_df.drop(data_df[(data_df.index.hour >= 20) | (data_df.index.hour < 8) | (
                                data_df.index.dayofweek >= 5)].index, inplace=True)
                        try:
                            data_df = data_df.drop(columns=['weekend'])
                        except:
                            pass
                    elif dataset_type == "nights_and_weekends_periods":
                        data_df.drop(data_df[(data_df.index.hour < 20) & (data_df.index.hour >= 8) & (
                                data_df.index.dayofweek < 5)].index, inplace=True)
            return data_df

        async def __check_for_more_categorical_cols(self, cols, apply, transformations, problem_type):
            if problem_type == 'classification':
                if apply:
                    n_previous_periods_list = transformations["number_previous_periods"]
                    for i in range(0, len(n_previous_periods_list)):
                        cols.append('t-' + str(n_previous_periods_list[i]))
            return cols

    async def setup(self):
        self.add_behaviour(self.ReceiveMsg(period=1))
