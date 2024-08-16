import json
import os
import pytz
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from data_model_app.serializers import ValidateModelPostSerializer, ValidateModelPostTransformationsSerializer, \
    ValidateModelPostFiltersSerializer
from marshmallow import ValidationError
from quart import Quart, request, jsonify, Response, render_template
import json
from io import StringIO
import asyncio
from quart_cors import cors
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
import logging
import requests
from data_model_app import views as views_data_model
from errors_app import views as views_errors
from forecasting_app import views as views_forecast
import pickle
from datetime import datetime
import pytz

app = Quart(__name__)
app = cors(app, allow_origin="*")
app.config['CORS_HEADERS'] = 'Content-Type'


# TRAIN

@app.route('/data_model/model', methods=['GET', 'POST', 'DELETE'])
async def data_model():
    print('in data model')
    try:
        input_jo = await request.json
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] loading json {str(e)}')
        return jsonify({"error": "Loading json: " + str(e)}, status=500)
    try:
        data, status = await views_data_model.data_model(input_jo, request.method)
        output, status = await views_forecast.forecasting_model(request.method, data)
        return jsonify(output), status
    except Exception as ex:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] ex data_model model {str(ex)}')
        e = str(ex)
        return jsonify({'error': e}), 400



# PREDICT
@app.route('/data_model/x_test', methods=['POST'])
async def x_test():
    try:
        input_jo = await request.json
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] loading json {str(e)}')
        return jsonify({"error": "Loading json: " + str(e)}, status=500)
    try:
        output, status = await views_data_model.x_test(input_jo, request.method)
        output, status = await views_forecast.predict(request.method, output)
        return jsonify(output), status
    except Exception as ex:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] ex x_test {str(ex)}')
        e = str(ex)
        return jsonify({'error': e}), 400



# RETRAIN

@app.route('/data_model/retrain_data', methods=['POST'])
async def retrain_data():
    try:
        input_jo = await request.json
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] loading json {str(e)}')
        return jsonify({"error": "Loading json: " + str(e)}, status=500)
    try:
        output, status = await views_data_model.retrain_data(input_jo, request.method)
        return jsonify(output), status
    except Exception as ex:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] ex retrain data {str(ex)}')
        e = str(ex)
        return jsonify({'error': e}), 400


# GET ALL MODELS
@app.route('/data_model/models', methods=['GET', 'DELETE'])
async def data_models():
    try:
        output, status = await views_data_model.data_models(request.method)
        return jsonify(output), status
    except Exception as ex:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] ex data_model models {str(ex)}')
        e = str(ex)
        return jsonify({'error': e}), 400


# INVERSE NORMALIZE
@app.route('/data_model/inverse_normalize_y', methods=['POST'])
async def inverse_normalize_y():
    try:
        input_jo = await request.json
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] loading json {str(e)}')
        return jsonify({"error": "Loading json: " + str(e)}, status=500)
    try:
        output, status = await views_data_model.inverse_normalize_y(input_jo, request.method)
        return jsonify(output), status
    except Exception as ex:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] ex inverse normalize {str(ex)}')
        e = str(ex)
        return jsonify({'error': e}), 400

#ERRORS
@app.route('/forecasting/errors/calculate', methods=['POST'])
async def calculate():
    try:
        input_jo = await request.json
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] loading json {str(e)}')
        return jsonify({"error": "Loading json: " + str(e)}, status=500)
    try:
        output, status = await views_errors.calculate(request.method, input_jo)
        return jsonify(output), status
    except Exception as ex:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] ex calculate errors {str(ex)}')
        e = str(ex)
        return jsonify({'error': e}), 400


@app.route('/forecasting/errors/calculate/<key>', methods=['POST'])
async def calculate_with_key(key):
    try:
        input_jo = await request.json
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] loading json {str(e)}')
        return jsonify({"error": "Loading json: " + str(e)}, status=500)
    try:
        output, status = await views_errors.calculate_with_key(request.method, input_jo, key)
        return jsonify(output), status
    except Exception as ex:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] ex calculate errors with key {str(ex)}')
        e = str(ex)
        return jsonify({'error': e}), 400





@app.route('/forecasting/model/train_predict', methods=['POST'])
async def train_predict():
    try:
        input_jo = await request.json
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] loading json {str(e)}')
        return jsonify({"error": "Loading json: " + str(e)}, status=500)
    try:
        output, status = await views_forecast.train_predict(request.method, input_jo)
        return jsonify(output), status
    except Exception as ex:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] ex train predict {str(ex)}')
        e = str(ex)
        return jsonify({'error': e}), 400


@app.route('/forecasting/model/retrain', methods=['POST'])
async def retrain():
    try:
        input_jo = await request.json
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] loading json {str(e)}')
        return jsonify({"error": "Loading json: " + str(e)}, status=500)
    try:
        output, status = await views_forecast.retrain(request.method, input_jo)
        return jsonify(output), status
    except Exception as ex:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] ex predict {str(ex)}')
        e = str(ex)
        return jsonify({'error': e}), 400


def timestamp_with_time_zone():
    current_time = datetime.now()
    timezone = "Europe/Lisbon"  # Desired timezone
    formated_timestamp = "%Y-%m-%d %H:%M:%S"
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


if __name__ == '__main__':
    app.run(debug=True)


async def get_model_file(method, json_input):
    if method == 'GET':

        try:
            # __validate_list_delete_data_model_input(request.data)
            pass
        except Exception as e:
            return {"error": "Validating input: " + str(e)}, 500

        model_name = json_input["model_name"]
        if await __check_if_data_model_exists(model_name):
            data = await getJsonFile(model_name)
            return data, 200
        else:
            return {}, 204


async def data_model(json_input, method):
    if method == 'GET':
        try:
            # __validate_list_delete_data_model_input(request.data)
            pass
        except Exception as e:
            return {"error": "Validating input: " + str(e)}, 500

        model_name = json_input["model_name"]

        if await __check_if_data_model_exists(model_name):
            return json_input, 200
        else:
            return {}, 204

    if method == 'POST':
        print(f"[{str(timestamp_with_time_zone())}] [ecma] create data model")
        try:
            await __validate_post_model(json_input)
        except ValidationError as e:
            return {"error": "Validating input: " + str(e.messages)}, 500

        model_name = json_input["model_name"]
        target_name = json_input["target_column_name"]
        print(f'[{str(timestamp_with_time_zone())}] [ecma]', model_name)

        data_ja = json_input["train_data"]

        try:
            data_df = await __generate_model(data_ja, json_input, model_name, False)
        except Exception as e:
            return {"error": "Generating model: " + str(e)}, 500
        try:
            x_train, y_train, final_columns_names = await __split_x_y(data_df, target_name)
        except Exception as e:
            return {"error": "Splitting model: " + str(e)}, 500

        try:
            await __save_settings(model_name, json_input, final_columns_names, x_train, y_train)
        except Exception as e:
            return {"error": "Saving settings: " + str(e)}, 500

        result = {}
        result["data"] = {'x_train': x_train, 'y_train': y_train}
        print(
            f"[{str(timestamp_with_time_zone())}] [ecma] data model len x train{str(len(x_train))} and len y train {str(len(y_train))}")
        result["columns_names"] = final_columns_names
        return result, 200

    if method == 'DELETE':
        model_name = json_input["model_name"]
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'data_models')
        x_file_path = os.path.join(models_folder, os.path.basename(model_name + "_x_scaler.save"))
        y_file_path = os.path.join(models_folder, os.path.basename(model_name + "_y_scaler.save"))
        settings_path = os.path.join(models_folder, os.path.basename(model_name + ".json"))

        if os.path.isfile(x_file_path) and os.path.isfile(y_file_path):
            os.remove(x_file_path)
            os.remove(y_file_path)
            os.remove(settings_path)
        else:
            return {}, 204

        return {}, 200


async def retrain_data(input_jo, method):
    if method == 'POST':
        print(f"[{str(timestamp_with_time_zone())}] [ecma] create retrain data")

        model_name = input_jo["model_name"]
        start_date = input_jo["start_date"]
        end_date = input_jo["end_date"]
        print(f'[{str(timestamp_with_time_zone())}] [ecma] {str(model_name)}, {str(start_date)}, {str(end_date)}')
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'data_models')
        try:
            print(os.path.join(models_folder, os.path.basename(model_name + ".json")))
            with open(os.path.join(models_folder, os.path.basename(model_name + ".json"))) as json_file:
                settings_jo = json.load(json_file)
        except Exception as e:
            raise Exception({"error": "Extracting settings data: " + str(e)})

        target_name = settings_jo["target_column_name"]

        data_ja = input_jo["data"]

        try:
            data_df = await __generate_model(data_ja, settings_jo, model_name, False)
        except Exception as e:
            return {"error": "Generating model: " + str(e)}, 500
        try:
            x_train, y_train, final_columns_names = await __split_x_y(data_df, target_name)
        except Exception as e:
            return {"error": "Splitting model: " + str(e)}, 500

        try:
            await __update_settings_after_retrain(x_train, y_train, settings_jo, data_ja, model_name)
        except Exception as e:
            return {"error": "Update settings after retrain: " + str(e)}, 500

        result = {}
        result["x_train"] = x_train
        result["y_train"] = y_train
        print(
            f"[{str(timestamp_with_time_zone())}] [ecma] x test generated with size {str(len(x_train))} and y test size {str(len(y_train))}")
        result["columns_names"] = final_columns_names

        return result, 200


async def x_test(input_jo, method):
    if method == 'POST':
        print(f"[{str(timestamp_with_time_zone())}] [ecma] create x test")

        model_name = input_jo["model_name"]
        start_date = input_jo["start_date"]
        end_date = input_jo["end_date"]
        print(f'[{str(timestamp_with_time_zone())}] [ecma] {str(model_name)}, {str(start_date)}, {str(end_date)}')
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'data_models')
        try:
            print(os.path.join(models_folder, os.path.basename(model_name + ".json")))
            with open(os.path.join(models_folder, os.path.basename(model_name + ".json"))) as json_file:
                settings_jo = json.load(json_file)
        except Exception as e:
            raise Exception({"error": "Extracting settings data: " + str(e)})

        target_name = settings_jo["target_column_name"]

        data_ja = input_jo["data"]

        # retrain = input_jo["retrain"]

        try:
            data_df = await __generate_model(data_ja, settings_jo, model_name, True)
        except Exception as e:
            return {"error": "Generating model: " + str(e)}, 500
        try:
            data_df = await __selecting_test(data_df, start_date, end_date)
        except Exception as e:
            return {"error": "Select model: " + str(e)}, 500
        # print(f"[{str(timestamp_with_time_zone())}] [ecma] data after {str(data_df)}")
        try:
            x_test, y_test, final_columns_names = await __split_x_y(data_df, target_name)
        except Exception as e:
            return {"error": "Splitting model: " + str(e)}, 500

        # if retrain:
        #     try:
        #         await __update_settings_after_retrain(x_test, y_test, settings_jo, data_ja, model_name)
        #     except Exception as e:
        #         return {"error": "Update settings after retrain: " + str(e)}, 500

        result = {}
        result["x_test"] = x_test
        result["y_test"] = y_test
        print(
            f"[{str(timestamp_with_time_zone())}] [ecma] x test generated with size {str(len(x_test))} and y test size {str(len(y_test))}")
        result["columns_names"] = final_columns_names

        return result, 200


async def __update_settings_after_retrain(x_train, y_train, settings_jo, data_ja, model_name):
    normalized_data = {
        "x_train": x_train,
        "y_train": y_train
    }
    settings_jo.update(
        {"train_data": data_ja})
    settings_jo.update(
        {"normalized_data": normalized_data})
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'data_models')
        file_path = os.path.join(models_folder, os.path.basename(model_name + ".json"))
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Name to save settings: {str(file_path)}")
        with open(file_path, "w") as outfile:
            outfile.write(json.dumps(settings_jo, indent=4))
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Saved")
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Save settings: {str(e)}")
        raise Exception(e)


async def data_models(method):
    if method == 'GET':
        print('hmmm')
        print('in models')
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'data_models')
        final_list = []
        try:
            files_list = os.listdir(models_folder)
        except Exception as e:
            print(f'[{str(timestamp_with_time_zone())}] [ecma] access denied {str(e)}')
            return {"error": "Accessing files: It was not possible to access to data models' folder. " + str(e)}, 500
        for name in files_list:
            if str(name).endswith(".nd"):
                files_list.remove(name)

        final_list = list(files_list)

        if len(final_list) > 0:
            for i in range(0, len(final_list)):
                if str(final_list[i]).endswith("_y_scaler.save"):
                    str_splited = str(final_list[i]).split("_y_scaler.save")
                    final_list[i] = str_splited[0]
                if str(final_list[i]).endswith("_x_scaler.save"):
                    str_splited = str(final_list[i]).split("_x_scaler.save")
                    final_list[i] = str_splited[0]
                if str(final_list[i]).endswith(".json"):
                    str_splited = str(final_list[i]).split(".json")
                    final_list[i] = str_splited[0]

            final_list = list(dict.fromkeys(final_list))

            return final_list, 200
        else:
            return final_list, 204

    if method == 'DELETE':
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'data_models')
        try:
            files_list = os.listdir(models_folder)
        except Exception as e:
            return {"error": "Accessing files: It was not possible to access to data models' folder. " + str(e)}, 500

        if not len(files_list) > 0:
            return {}, 204

        for name in files_list:
            if str(name).endswith(".nd"):
                files_list.remove(name)

        for file in files_list:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            models_folder = os.path.join(BASE_DIR, 'data_models')
            file_path = os.path.join(models_folder, os.path.basename(file))

            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    return {"error": "Deleting: Error while deleting data models. " + str(e)}, 500

        return files_list, 200


async def inverse_normalize_y(input_jo, method):
    if method == 'POST':
        model_name = input_jo["model_name"]
        data = input_jo["data"]
        print(f'[{str(timestamp_with_time_zone())}] [ecma] inverse normalize {str(data)}')
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'data_models')
        try:
            y_scaler = joblib.load(os.path.join(models_folder, os.path.basename(model_name + "_y_scaler.save")))
            with open(os.path.join(models_folder, os.path.basename(model_name + ".json"))) as json_file:
                setting_jo = json.load(json_file)
        except Exception as e:
            try:
                x_scaler = joblib.load(os.path.join(models_folder, os.path.basename(model_name + "_x_scaler.save")))
                y = data
                return y, 200
            except:
                print(f'[{str(timestamp_with_time_zone())}] [ecma] getting model {str(e)}')
                return {"error": "Getting model: " + str(e)}, 500

        try:
            target_column = setting_jo["target_column_name"]
            data_df = pd.DataFrame(np.array(data).reshape(-1, 1), columns=[target_column])
            data_df[[target_column]] = y_scaler.inverse_transform(data_df[[target_column]])
            y = data_df[target_column].values.tolist()
            print(f"[{str(timestamp_with_time_zone())}] [ecma] normalized target {str(y)}")
        except Exception as e:
            print(f'[{str(timestamp_with_time_zone())}] [ecma]  Normalizing test {str(e)}')
            return {"error": "Normalizing test: " + str(e)}, 500

        return y, 200


async def __check_if_data_model_exists(model_name):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_folder = os.path.join(BASE_DIR, 'data_models')
    file_path = os.path.join(models_folder, os.path.basename(model_name) + "_x_scaler.save")
    check_file = os.path.isfile(file_path)
    return check_file


async def getJsonFile(model_name):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'data_models')
        with open(os.path.join(models_folder, os.path.basename(model_name + ".json"))) as json_file:
            data = json.load(json_file)
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] error getting json file {str(e)}")
        raise Exception({"error": "Getting json file: " + str(e)})
    return data


async def __generate_model(data_ja, settings_jo, model_name, is_test_model):
    print(f"[{str(timestamp_with_time_zone())}] [ecma] generate model with name {str(model_name)}")
    datetime_column_name = settings_jo["datetime_column_name"]
    datetime_format = settings_jo["datetime_format"]
    target_name = settings_jo["target_column_name"]
    categorical_columns_names = settings_jo["categorical_columns_names"]
    columns_names = settings_jo["columns_names"]
    normalize = settings_jo["normalize"]
    transformations = settings_jo["transformations"]
    filters = settings_jo["filters"]
    data_df = pd.DataFrame(data_ja)
    data_df.columns = columns_names
    data_df['date_aux'] = pd.to_datetime(data_df[datetime_column_name], format=datetime_format)
    datetime_cols_in_df = ['date_aux']
    data_df.set_index('date_aux', inplace=True)
    data_df = data_df.drop(columns=[datetime_column_name])
    print(f"[{str(timestamp_with_time_zone())}] [ecma] shape df initial {str(data_df.shape)}")
    # print(f"[{str(timestamp_with_time_zone())}] [ecma] data before {str(data_df)}")
    categorical_columns_names = await __check_for_more_categorical_cols(categorical_columns_names,
                                                                        transformations["previous_periods"],
                                                                        transformations, filters["problem_type"])
    try:
        data_df, datetime_cols_in_df = await __transform_data(data_df, transformations, target_name,
                                                              datetime_cols_in_df)
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Transforming data: {str(e)}")
        raise Exception({"error": "Transforming data: " + str(e)})

    try:
        if is_test_model:
            # print(f'[{str(timestamp_with_time_zone())}] [ecma]', data_df)
            # print(f'[{str(timestamp_with_time_zone())}] [ecma]', data_df.columns)
            data_df = await __extract_categorical_data_test(data_df, categorical_columns_names,
                                                            settings_jo["final_columns_names"])
        else:
            data_df = await __extract_categorical_data(data_df, categorical_columns_names)
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Extracting categorical data: {str(e)}")
        raise Exception({"error": "Extracting categorical data: " + str(e)})

    try:
        data_df = await __filter_data(data_df, filters)
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Filtering data: {str(e)}")
        raise Exception({"error": "Filtering data: " + str(e)})

    if data_df.empty:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] empty dataframe')
        raise Exception({"error": "Empty dataset"})
    print(f"[{str(timestamp_with_time_zone())}] [ecma] shape here {str(data_df.shape)}")

    try:
        data_df = await __fix_datetime_cols(data_df, datetime_cols_in_df, transformations["datetime_cols"])
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Fixing datetime columns data: {str(e)}")
        raise Exception({"error": "Fixing datetime columns: " + str(e)})

    normalized_data_df = None
    if normalize:
        try:
            if is_test_model:
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                models_folder = os.path.join(BASE_DIR, 'data_models')
                try:
                    x_std_scale = joblib.load(
                        os.path.join(models_folder, os.path.basename(model_name + "_x_scaler.save")))
                    try:
                        y_std_scale = joblib.load(
                            os.path.join(models_folder, os.path.basename(model_name + "_y_scaler.save")))
                    except:
                        y_std_scale = None
                except Exception as e:
                    print(f'[{str(timestamp_with_time_zone())}] [ecma] line 310 {str(e)}')
                    raise Exception(e)

                normalized_data_df = await __normalize_data_test(data_df, target_name, x_std_scale, y_std_scale)
            else:
                normalized_data_df, x_std_scale, y_std_scale = await __normalize_data(data_df, target_name,
                                                                                      filters["problem_type"])
                await __save_normalize_objects(model_name, x_std_scale, y_std_scale)

        except Exception as e:
            print(f'[{str(timestamp_with_time_zone())}] [ecma] {str(e)}')
            raise Exception({"error": "Normalizing data: " + str(e)})
    else:
        normalized_data_df = data_df
    print(f"[{str(timestamp_with_time_zone())}] [ecma] model generated with shape {str(data_df.shape)}")
    return normalized_data_df


async def __extract_categorical_data(data_df, columns_names_list):
    for column in columns_names_list:
        try:
            array = pd.get_dummies(data_df[column], prefix=column)
            enc_df = pd.DataFrame(array)
            data_df = data_df.join(enc_df)
            data_df = data_df.drop(columns=[column])
        except Exception as e:
            print(f"[{str(timestamp_with_time_zone())}] [ecma] Extract categorical data: {str(e)}")
            raise Exception(e)

    return data_df


async def __extract_categorical_data_test(data_df, categorical_columns_names, all_columns_names):
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
                print(f"[{str(timestamp_with_time_zone())}] [ecma] Extract categorical data test : {str(e)}")
                raise Exception(e)

    return data_df


async def __normalize_data(data_df, target_column, problemType):
    try:
        x_cols = list(data_df.columns)
        x_cols.remove(target_column)
        x_std_scale = StandardScaler().fit(data_df[x_cols])
        data_df[x_cols] = x_std_scale.transform(data_df[x_cols])
        if problemType.lower() == 'classification':
            y_std_scale = None
            # data_df_encoded = pd.get_dummies(data_df, columns=[
            #     target_column])
            # print(data_df_encoded)
            # y_std_scale = StandardScaler().fit(data_df_encoded)
            # data_df[[target_column]] = y_std_scale.transform(data_df_encoded)
        #     print('classification')
        #     y_std_scale = MinMaxScaler().fit(data_df[[target_column]])
        #     data_df[[target_column]] = y_std_scale.transform(data_df[[target_column]])
        #     print(data_df[target_column].values.tolist())
        else:
            y_std_scale = StandardScaler().fit(data_df[[target_column]])
            data_df[[target_column]] = y_std_scale.transform(data_df[[target_column]])
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Normalize data: {str(e)}")
        raise Exception(e)

    return data_df, x_std_scale, y_std_scale


async def __normalize_data_test(data_df, target_column, x_std_scale, y_std_scale):
    try:
        x_cols = list(data_df.columns)
        x_cols.remove(target_column)

        data_df[x_cols] = x_std_scale.transform(data_df[x_cols])
        if y_std_scale is not None:
            data_df[[target_column]] = y_std_scale.transform(data_df[[target_column]])
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Normalize data test: {str(e)}")
        raise Exception(e)

    return data_df


async def __save_normalize_objects(model_name, x_std_scale, y_std_scale):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_folder = os.path.join(BASE_DIR, 'data_models')
    file_path = os.path.join(models_folder, os.path.basename(model_name))
    try:
        joblib.dump(x_std_scale, file_path + "_x_scaler.save")
        if y_std_scale is not None:
            joblib.dump(y_std_scale, file_path + "_y_scaler.save")
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Save normalized objects: {str(e)}")
        raise Exception(e)


async def __save_settings(model_name, settings_jo, final_columns_names, x_train, y_train):
    settings_jo["final_columns_names"] = final_columns_names
    normalized_data = {
        "x_train": x_train,
        "y_train": y_train
    }
    settings_jo["normalized_data"] = normalized_data
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'data_models')
        file_path = os.path.join(models_folder, os.path.basename(model_name + ".json"))
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Name to save settings: {str(file_path)}")
        with open(file_path, "w") as outfile:
            outfile.write(json.dumps(settings_jo, indent=4))
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Saved")
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Save settings: {str(e)}")
        raise Exception(e)


async def __split_x_y(data_df, target_name):
    try:
        x_train = data_df.drop(columns=[target_name])
        y_train = data_df[target_name]

        cols = list(x_train.columns)

        x_train = x_train.values.tolist()
        y_train = y_train.values.tolist()
        print(
            f"[{str(timestamp_with_time_zone())}] [ecma] x train with len {str(len(x_train))}, y train size {str(len(y_train))}")

    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] Split x and y: {str(e)}')
        raise Exception(e)

    return x_train, y_train, cols


async def __selecting_test(data_df, start_date, end_date):
    try:
        mask = (data_df.index >= start_date) & (data_df.index <= end_date)
        data_df = data_df.loc[mask]
    except Exception as e:
        print(f"[{str(timestamp_with_time_zone())}] [ecma] Select test: {str(e)}")
        raise Exception(e)
    return data_df


async def __transform_data(data_df, transformations, target_name, datetime_cols_in_df):
    for key in transformations.keys():
        apply = transformations[key]
        if key == "split_datetime":
            if apply:
                print(f'[{str(timestamp_with_time_zone())}] [ecma] applied')
                # data_df['year'] = data_df.index.year
                data_df['month'] = data_df.index.month
                data_df['day'] = data_df.index.day
                data_df['hour'] = data_df.index.hour
                data_df['minute'] = data_df.index.minute
                # print('ok here')
                datetime_cols_in_df.append('month')
                datetime_cols_in_df.append('day')
                datetime_cols_in_df.append('hour')
                datetime_cols_in_df.append('minute')

        if key == "day_of_week":
            if apply:
                print(f'[{str(timestamp_with_time_zone())}] [ecma] applied')
                day_names = data_df.index.day_name()
                # create the day of week column
                for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                    data_df['day_of_week_' + day] = 0
                # update if exists
                for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                    data_df.loc[day_names == day, 'day_of_week_' + day] = 1
                # data_df['day_of_week'] = data_df.index.day_name()
                # array = pd.get_dummies(data_df['day_of_week'], prefix='day_of_week')
                # enc_df = pd.DataFrame(array)
                # data_df = data_df.join(enc_df)
                # data_df = data_df.drop(columns=['day_of_week'])

        if key == "weekend":
            if apply:
                print(f'[{str(timestamp_with_time_zone())}] [ecma] applied')
                data_df['weekend'] = data_df.index.dayofweek >= 5

        if key == "trimester":
            if apply:
                print(f'[{str(timestamp_with_time_zone())}] [ecma] applied')
                trimesters = data_df.index.quarter
                # create trimester
                for trimester in [1, 2, 3, 4]:
                    data_df['trimester_' + str(trimester)] = 0
                # update if exists
                for trimester in [1, 2, 3, 4]:
                    data_df.loc[trimesters == trimester, 'trimester_' + str(trimester)] = 1
                # data_df['trimester'] = data_df.index.quarter
                #
                # array = pd.get_dummies(data_df['trimester'], prefix='trimester')
                # enc_df = pd.DataFrame(array)
                # data_df = data_df.join(enc_df)
                # data_df = data_df.drop(columns=['trimester'])

        if key == "week_of_year":
            if apply:
                print(f'[{str(timestamp_with_time_zone())}] [ecma] applied')
                data_df['week_of_year'] = data_df.index.isocalendar().week

                array = pd.get_dummies(data_df['week_of_year'], prefix='week_of_year')
                enc_df = pd.DataFrame(array)
                data_df = data_df.join(enc_df)
                data_df = data_df.drop(columns=['week_of_year'])

        if key == "day_of_year":
            if apply:
                print(f'[{str(timestamp_with_time_zone())}] [ecma] applied')
                data_df['day_of_year'] = data_df.index.dayofyear

                array = pd.get_dummies(data_df['day_of_year'], prefix='day_of_year')
                enc_df = pd.DataFrame(array)
                data_df = data_df.join(enc_df)
                data_df = data_df.drop(columns=['day_of_year'])

        if key == "previous_periods":
            if apply:
                print(f'[{str(timestamp_with_time_zone())}] [ecma] applied')
                n_previous_periods_list = transformations["number_previous_periods"]
                for i in range(0, len(n_previous_periods_list)):
                    data_df['t-' + str(n_previous_periods_list[i])] = data_df[target_name].shift(
                        n_previous_periods_list[i])
    data_df = data_df.dropna()
    return data_df, datetime_cols_in_df


async def __filter_data(data_df, filters):
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
        # if key == "problem_type":
        #     dataset_type = filters[key]
        #     if dataset_type == "classification":
        #         data_df = await __extract_categorical_data(data_df, added_cols)
    return data_df


async def __validate_post_model(request_data):
    # validate full json
    # valid_input = ValidateModelPostSerializer(data=request_data)
    # if not valid_input.is_valid():
    #     raise Exception(valid_input.errors)
    try:
        valid_input = ValidateModelPostSerializer().load(request_data)
    except ValidationError as err:
        return err

    # validate transformations
    # valid_input = ValidateModelPostTransformationsSerializer(data=request_data["transformations"])
    # if not valid_input.is_valid():
    #     raise Exception(valid_input.errors)
    try:
        valid_input = ValidateModelPostTransformationsSerializer().load(request_data)
    except ValidationError as err:
        return err

    # validate filters
    # valid_input = ValidateModelPostFiltersSerializer(data=request_data["filters"])
    # if not valid_input.is_valid():
    #     raise Exception(valid_input.errors)

    try:
        valid_input = ValidateModelPostFiltersSerializer().load(request_data)
    except ValidationError as err:
        return err


async def __check_for_more_categorical_cols(cols, apply, transformations, problem_type):
    print(f"[{str(timestamp_with_time_zone())}] [ecma] apply: {str(apply)}")
    if problem_type == 'classification':
        if apply:
            n_previous_periods_list = transformations["number_previous_periods"]
            for i in range(0, len(n_previous_periods_list)):
                cols.append('t-' + str(n_previous_periods_list[i]))
    return cols


async def __fix_datetime_cols(data_df, datetime_cols_already, datetime_cols):
    cols_df = data_df.columns
    for col in datetime_cols_already:
        if col in cols_df and col not in datetime_cols:
            data_df = data_df.drop(columns=[col])
    return data_df


def timestamp_with_time_zone():
    current_time = datetime.now()
    timezone = "Europe/Lisbon"  # Desired timezone
    formated_timestamp = "%Y-%m-%d %H:%M:%S"
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


import json
import os
import pickle
import re
from datetime import datetime
from ast import literal_eval
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, KFold, cross_val_score, cross_val_predict, \
    cross_validate
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
import logging
import pytz
from forecasting_app.serializers import ValidateTrainInputSerializer, ValidateRetrainInputSerializer, \
    ValidateTrainDataSerializer, \
    ValidateTrainMlModelSerializer, ValidateTestInputSerializer, ValidateGetModelInputSerializer, \
    ValidateDeleteInputSerializer, ValidateTrainTestDataInputSerializer, ValidateTrainPredictMlModelSerializer, \
    ValidateTrainCrossValidationSerializer
from marshmallow import Schema, fields, ValidationError

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='error.log',
                    filemode='a')
logging.info(
    f"###############################################\n###############################################\n###############################################\n")


async def get_model_file(method, json_input):
    if method == 'GET':
        try:
            await __validate_get_model_input(json_input)
        except ValidationError as e:
            return {"error": "Validating input: " + str(e.messages)}, 500

        model_name = json_input["model_name"]
        timestamp_format = json_input["timestamp_format"]
        if await __check_if_exists(model_name):
            pickle_data, creation_time_str = await getPklFile(model_name, timestamp_format)
            return {
                "model": pickle_data.decode('latin1'),
                "creation_timestamp": creation_time_str
            }, 200
            # response_content = f"{}\ncreation_timestamp: {creation_time_str}"
            # return HttpResponse(response_content.encode('utf-8'), content_type='application/octet-stream')
        else:
            return {}, 204


async def forecasting_model(method, json_input):
    if method == 'GET':
        try:
            await __validate_get_model_input(json_input)
        except ValidationError as e:
            return {"error": "Validating input: " + str(e.messages)}, 500

        model_name = json_input["model_name"]

        if await __check_if_exists(model_name):
            return json_input, 200
        else:
            return {}, 204

    if method == 'POST':
        logging.info(f"calling train")
        print(f'[{str(timestamp_with_time_zone())}] [ecma] in train request')

        print(f'[{str(timestamp_with_time_zone())}] [ecma] input ok')
        try:
            #o que é request.data vs request.body
            # __validate_train_input(request.data, json_input)
            await __validate_train_input(json_input)
        except ValidationError as e:
            return {"error": "Validating input: " + str(e.messages)}, 500
        print(f'[{str(timestamp_with_time_zone())}] [ecma] train input valid')
        model_base_name = json_input['model_base_name']
        data = json_input['data']
        ml_model = json_input['ml_model']
        cross_validation = json_input["cross_validation"]

        x_train = data['x_train']
        y_train = data['y_train']

        ml_model_name = ml_model['ml_model_name']

        ml_model_parameters = await __get_parameters(ml_model)
        print(f'[{str(timestamp_with_time_zone())}] [ecma] until line 80')
        try:
            created_model_name, regressor, train_score = await __train(x_train, y_train, ml_model_parameters,
                                                                       ml_model_name, True, model_base_name)
        except Exception as e:
            return {"error": "Training: " + str(e)}, 500
        print(f'[{str(timestamp_with_time_zone())}] [ecma] trained')
        try:
            cross_validation_result = await __cross_validation(ml_model_name, x_train, y_train, ml_model_parameters,
                                                               cross_validation)
        except Exception as e:
            print(f'[{str(timestamp_with_time_zone())}] [ecma] exception in cross validation', e)
            return {"error": "Cross validation: " + str(e)}, 500
        print(
            f'[{str(timestamp_with_time_zone())}] [ecma] cross validation {str(cross_validation_result)} vs {str(train_score)}')
        logging.info(f'trained {created_model_name}')
        return {"model_name": created_model_name,
                "cross_validation": train_score}, 201

    if method == 'DELETE':
        try:
            await __validate_delete_model_input(json_input)
        except ValidationError as e:
            return {"error": "Validating input: " + str(e.messages)}, 500

        model_name = json_input["model_name"]

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'ml_models')
        file_path = os.path.join(models_folder, os.path.basename(model_name))

        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            return {}, 204

        return {}, 200


async def predict(method, json_input):
    if method == 'POST':

        try:
            await __validate_test_input(json_input)
        except ValidationError as e:
            logging.info(f"Validating: {str(e)}")
            return {"error": "Validating input: " + str(e.messages)}, 500

        model_name = json_input['model_name']
        x_test = json_input['x_test']
        logging.info(f"predict with x test size {str(len(x_test))}")
        try:
            result = await __predict(x_test, model_name)
        except Exception as e:
            logging.exception(f"error: Predicting: {str(e)} ---- {str(len(x_test))} {str(model_name)}")
            return {"error": "Predicting: " + str(e) + str(len(x_test)) + str(model_name)}, 500
        ########################
        # SAVE LAST X_TEST
        ########################

        model_name_json = json_input["model_name_json"]
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            models_folder = os.path.join(BASE_DIR, 'data_models')
            name_path = os.path.join(models_folder, os.path.basename(model_name_json + ".json"))
            name_path = name_path.replace('forecasting_app', 'data_model_app')
            with open(name_path) as json_file:
                settings_jo = json.load(json_file)
        except Exception as e:
            raise Exception({"error": "Extracting settings data: " + str(e)})
        if 'not_save' not in json_input:
            if 'last_x_test' not in settings_jo:
                settings_jo["last_x_test"] = x_test
            else:
                penultimate_x_test = settings_jo["last_x_test"]
                settings_jo["last_x_test"] = x_test
                settings_jo["penultimate_x_test"] = penultimate_x_test
            try:
                with open(name_path, 'w') as json_file:
                    json_file.write(json.dumps(settings_jo, indent=4))
            except Exception as e:
                raise Exception({"error": "Extracting settings data: " + str(e)})

        return result.tolist(), 200


async def train_predict(method, input_jo):
    if method == 'POST':
        logging.info(f"calling train predict")

        try:
            await __validate_train_predict_input(input_jo)
        except ValidationError as e:
            return {"error": "Validating input: " + str(e.messages)}, 500

        model_base_name = input_jo['model_base_name']
        data = input_jo['data']
        ml_model = input_jo['ml_model']
        cross_validation = input_jo["cross_validation"]

        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        logging.info(
            f" train predict with x train size {str(len(x_train))}, y train size {str(len(y_train))}, x test size {str(len(x_test))}")
        ml_model_name = ml_model['ml_model_name']
        ml_model_save = ml_model['ml_model_save']

        ml_model_parameters = await __get_parameters(ml_model)

        try:
            created_model_name, regressor, train_score = await __train(x_train, y_train, ml_model_parameters,
                                                                       ml_model_name, ml_model_save, model_base_name)
        except Exception as e:
            logging.exception(f"error: Training:  {str(e)}")
            return {"error": "Training: " + str(e)}, 500

        try:
            logging.info(f"in train predict")
            predictions = regressor.predict(x_test)
            logging.info(
                f"predict in train predict worked in line 167 with {str(model_base_name)}and result {str(predictions)}")
        except Exception as e:
            logging.exception(f"error: Train Predict: {str(e)} {str(len(x_test))} {str(ml_model_name)}")
            return {"error": "Train Predict: " + str(e) + str(len(x_test)) + str(ml_model_name)}, 500

        # try:
        #     cross_validation_result = __cross_validation(ml_model_name, x_train, y_train, ml_model_parameters, cross_validation)
        # except Exception as e:
        #     logging.exception(f"error: Cross validation: {str(e)}")
        #     return {"error": "Cross validation: " + str(e)}, 500

        result = {}
        if ml_model_save:
            result["model_name"] = created_model_name

        result["predictions"] = predictions.tolist()
        result["cross_validation"] = train_score

        ########################
        # SAVE LAST X_TEST
        ########################

        model_name_json = input_jo["model_name_json"]

        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            models_folder = os.path.join(BASE_DIR, 'data_models')
            name_path = os.path.join(models_folder, os.path.basename(model_name_json + ".json"))
            name_path = name_path.replace('forecasting_app', 'data_model_app')
            with open(name_path) as json_file:
                settings_jo = json.load(json_file)
        except Exception as e:
            raise Exception({"error": "Extracting settings data: " + str(e)})

        if 'last_x_test' not in settings_jo:
            settings_jo["last_x_test"] = x_test
        else:
            penultimate_x_test = settings_jo["last_x_test"]
            settings_jo["last_x_test"] = x_test
            settings_jo["penultimate_x_test"] = penultimate_x_test
        try:
            with open(name_path, 'w') as json_file:
                json_file.write(json.dumps(settings_jo, indent=4))
        except Exception as e:
            raise Exception({"error": "Extracting settings data: " + str(e)})
        return result, 200


async def retrain(method, json_input):
    if method == 'POST':
        logging.info(f"calling train")
        print(f'[{str(timestamp_with_time_zone())}] [ecma] in train request')

        print(f'[{str(timestamp_with_time_zone())}] [ecma] input ok')
        try:
            await __validate_retrain_input(json_input)
        except ValidationError as e:
            return {"error": "Validating input: " + str(e.messages)}, 500
        print(f'[{str(timestamp_with_time_zone())}] [ecma] train input valid')
        model_name = json_input['model_base_name']
        data = json_input['data']
        x_train = data['x_train']
        y_train = data['y_train']
        print(f'[{str(timestamp_with_time_zone())}] [ecma] until line 80')
        try:
            model_name, train_score = await __retrain(x_train, y_train, True, model_name)
        except Exception as e:
            return {"error": "Training: " + str(e)}, 500
        print(f'[{str(timestamp_with_time_zone())}] [ecma] trained')

        logging.info(f'retrained {model_name}')
        return {"model_name": model_name,
                "cross_validation": train_score}, 201


async def forecasting_models(method):
    if method == 'GET':

        files_list = []

        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            models_folder = os.path.join(BASE_DIR, 'ml_models')
            files_list = os.listdir(models_folder)
        except Exception as e:
            return {"error": "Accessing files: It was not possible to access to ML models' folder. " + str(e)}, 500

        for name in files_list:
            if str(name).endswith(".nd"):
                files_list.remove(name)

        if len(files_list) > 0:
            return files_list, 200
        else:
            return files_list, 204

    if method == 'DELETE':

        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            models_folder = os.path.join(BASE_DIR, 'ml_models')
            files_list = os.listdir(models_folder)
        except Exception as e:
            return {"error": "Accessing files: It was not possible to access to ML models' folder. " + str(e)}, 500

        if not len(files_list) > 0:
            return {}, 204

        for name in files_list:
            if str(name).endswith(".nd"):
                files_list.remove(name)

        for file in files_list:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            models_folder = os.path.join(BASE_DIR, 'ml_models')
            file_path = os.path.join(models_folder, os.path.basename(file))

            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    return {"error": "Deleting: Error while deleting ML models. " + str(e)}, 500

        return files_list, 200


async def __check_if_exists(model_name):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_folder = os.path.join(BASE_DIR, 'ml_models')
    file_path = os.path.join(models_folder, os.path.basename(model_name))
    check_file = os.path.isfile(file_path)
    return check_file


async def getPklFile(model_name, timestamp_format):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_folder = os.path.join(BASE_DIR, 'ml_models')
    file_path = os.path.join(models_folder, os.path.basename(model_name))
    with open(file_path, 'rb') as f:
        pickle_data = f.read()
    creation_time = os.path.getctime(file_path)
    creation_time_str = time.strftime(timestamp_format, time.localtime(creation_time))
    return pickle_data, creation_time_str


async def __validate_get_model_input(request_data):
    # validate input
    # valid_input = ValidateGetModelInputSerializer(data=request_data)
    # if not valid_input.is_valid():
    #     raise Exception(valid_input.errors)
    try:
        valid_input = ValidateGetModelInputSerializer().load(request_data)
    except ValidationError as err:
        return err


async def __validate_delete_model_input(request_data):
    # validate input
    # valid_input = ValidateDeleteInputSerializer(data=request_data)
    # if not valid_input.is_valid():
    #     raise Exception(valid_input.errors)
    try:
        valid_input = ValidateDeleteInputSerializer().load(request_data)
    except ValidationError as err:
        return err


async def __validate_train_input(input_jo):
    # validate input
    # valid_input = ValidateTrainInputSerializer(data=request_data)
    # if valid_input.is_valid():
    #     model_base_name = input_jo['model_base_name']
    # else:
    #     raise Exception(valid_input.errors)
    try:
        valid_input = ValidateTrainInputSerializer().load(input_jo)
        model_base_name = input_jo['model_base_name']
    except ValidationError as err:
        return err

    # validate name
    if_has_spaces = bool(re.search(r"\s", model_base_name))
    if if_has_spaces:
        raise Exception("Model name string must not have spaces")

    # validate data
    # valid_data = ValidateTrainDataSerializer(data=request_data['data'])
    # if valid_data.is_valid():
    #     data = input_jo['data']
    # else:
    #     raise Exception(valid_data.errors)
    try:
        valid_input = ValidateTrainDataSerializer().load(input_jo['data'])
        data = input_jo['data']
    except ValidationError as err:
        return err

    x_train = data['x_train']
    y_train = data['y_train']

    if len(x_train) != len(y_train):
        raise Exception("x_train and y_train must have same size")

    # # validate model data
    # valid_ml_model = ValidateTrainMlModelSerializer(data=request_data['ml_model'])
    # if not valid_ml_model.is_valid():
    #     raise Exception(valid_ml_model.errors)
    try:
        valid_input = ValidateTrainMlModelSerializer().load(input_jo['ml_model'])
    except ValidationError as err:
        return err

    #
    # # validate model data
    # valid_ml_model = ValidateTrainCrossValidationSerializer(data=request_data['cross_validation'])
    # if not valid_ml_model.is_valid():
    #     raise Exception(valid_ml_model.errors)

    try:
        valid_input = ValidateTrainCrossValidationSerializer().load(input_jo['cross_validation'])
    except ValidationError as err:
        return err


async def __validate_retrain_input(input_jo):
    try:
        valid_input = ValidateRetrainInputSerializer().load(input_jo)
        model_name = input_jo['model_name']
    except ValidationError as err:
        return err

    # validate name
    if_has_spaces = bool(re.search(r"\s", model_name))
    if if_has_spaces:
        raise Exception("Model name string must not have spaces")

    # validate data
    try:
        valid_input = ValidateTrainDataSerializer().load(input_jo['data'])
        data = input_jo['data']
    except ValidationError as err:
        return err

    x_train = data['x_train']
    y_train = data['y_train']

    if len(x_train) != len(y_train):
        raise Exception("x_train and y_train must have same size")


async def __get_parameters(input_jo):
    ml_model_parameters = {}
    if not input_jo['ml_model_parameters'] is None and input_jo['ml_model_parameters']:
        ml_model_parameters = input_jo['ml_model_parameters']

        # validate parameters
        for key in ml_model_parameters.keys():
            obj = ml_model_parameters[key]

            # if tuple
            if isinstance(obj, str) and obj.startswith("(") and obj.endswith(")"):
                tuple = literal_eval(obj)
                ml_model_parameters[key] = tuple

    return ml_model_parameters


async def __train(x_train, y_train, ml_model_parameters, ml_model_name, ml_model_save, model_base_name):
    print(f'[{str(timestamp_with_time_zone())}] [ecma] in train method {str(ml_model_parameters)}')

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
        raise Exception("ML model name not found")
    print(f'[{str(timestamp_with_time_zone())}] [ecma] regressor chosen {str(regressor)}')
    try:
        regressor.fit(x_train, y_train)
    except Exception as exp:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] fit exception {str(exp)}')
        raise Exception(exp)
    print(f'[{str(timestamp_with_time_zone())}] [ecma] fit ok')
    try:

        metric = regressor.score(x_train, y_train)
        train_score = {"scores": [metric], "mean_score": metric}
        print(f'[{str(timestamp_with_time_zone())}] [ecma] Training score for {regressor} (R2): {train_score}')
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] exception getting score {str(e)}')
        raise Exception(e)
    created_model_name = {}

    if ml_model_save:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] in save')
        now = timestamp_with_time_zone()
        now_date_time = now.strftime("%Y%m%d%H%M%S")

        created_model_name = model_base_name + "_" + now_date_time + '.pkl'

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(BASE_DIR, 'ml_models')
        file_path = os.path.join(models_folder, os.path.basename(created_model_name))
        logging.info(f"info: Training file path: {str(file_path)}")
        try:
            pickle.dump(regressor, open(file_path, 'wb'))
            logging.info(f"saving pkl model")
            try:
                regressor_again = pickle.load(open(file_path, 'rb'))
                logging.info(f"saving pkl model okkkk")
            except Exception as e:
                logging.info(f"saving pkl model error after saving didnt got error")
                raise Exception(str(e))
        except Exception as ex:
            logging.info(f"saving pkl model didnt work", ex)
            raise Exception(ex)

    return created_model_name, regressor, train_score


async def __predict(x_test, model_name):
    logging.info(f"calling predict")
    if await __check_if_exists(model_name):
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            models_folder = os.path.join(BASE_DIR, 'ml_models')
            file_path = os.path.join(models_folder, os.path.basename(model_name))
        except Exception as e:
            raise Exception(str(e))
    else:
        raise Exception("Model not found.")
    logging.info(f"info: Predicting file path: {str(file_path)}")
    try:
        regressor = pickle.load(open(file_path, 'rb'))
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] {str(e)}')
        raise Exception(str(e))

    try:
        result = regressor.predict(x_test)
        # logging.info(f"ECMA predict to the similar old case {str(model_name)} with result {str(result)}")
        logging.info(f"predicted worked in line 374")

    except Exception as e:
        raise Exception(str(e))
    return result


async def __retrain(x_train, y_train, ml_model_save, model_name):
    print(f'[{str(timestamp_with_time_zone())}] [ecma] in retrain method')

    if await __check_if_exists(model_name):
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            models_folder = os.path.join(BASE_DIR, 'ml_models')
            file_path = os.path.join(models_folder, os.path.basename(model_name))
        except Exception as e:
            raise Exception(str(e))
    else:
        raise Exception("Model not found.")
    try:
        regressor = pickle.load(open(file_path, 'rb'))
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] {str(e)}')
        raise Exception(str(e))
    try:
        regressor.fit(x_train, y_train)
    except Exception as exp:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] fit exception {str(exp)}')
        raise Exception(exp)
    print(f'[{str(timestamp_with_time_zone())}] [ecma] fit ok')
    try:

        metric = regressor.score(x_train, y_train)
        train_score = {"scores": [metric], "mean_score": metric}
        print(f'[{str(timestamp_with_time_zone())}] [ecma] Training score for {regressor} (R2): {train_score}')
    except Exception as e:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] exception getting score {str(e)}')
        raise Exception(e)
    if ml_model_save:
        print(f'[{str(timestamp_with_time_zone())}] [ecma] in save retrain')
        print(f"[{str(timestamp_with_time_zone())}] [ecma] info: Retraining file path: {str(file_path)}")
        try:
            pickle.dump(regressor, open(file_path, 'wb'))
            logging.info(f"saving pkl model")
            try:
                regressor_again = pickle.load(open(file_path, 'rb'))
                print(f"[{str(timestamp_with_time_zone())}] [ecma] saving retrained pkl model okkkk")
            except Exception as e:
                print(f"[{str(timestamp_with_time_zone())}] [ecma] saving pkl model error after saving didnt got error")
                raise Exception(str(e))
        except Exception as ex:
            print(f"[{str(timestamp_with_time_zone())}] [ecma] saving pkl model didnt work {str(ex)}")
            raise Exception(ex)

    return model_name, train_score


async def __cross_validation(ml_model_name, x, y, ml_model_parameters, cross_validation_jo):
    print(f'[{str(timestamp_with_time_zone())}] [ecma] cross_validation here {str(cross_validation_jo)}')
    use = cross_validation_jo["use"]
    method_name = cross_validation_jo["method"]
    k_folds = cross_validation_jo["k_folds"]
    shuffle = cross_validation_jo["shuffle"]
    scoring = cross_validation_jo["scoring"]

    if use:
        regressor = None
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
            raise Exception("ML model name not found")

        method = None
        if method_name.lower() == "StratifiedKFold".lower():
            method = StratifiedKFold(n_splits=k_folds, shuffle=shuffle)
        elif method_name.lower() == "KFold".lower():
            method = KFold(n_splits=k_folds, shuffle=shuffle)
        elif method_name.lower() == "TimeSeriesSplit".lower():
            method = TimeSeriesSplit(n_splits=k_folds)
        else:
            raise Exception("Cross validation method not found")
        print(f'[{str(timestamp_with_time_zone())}] [ecma] aqui {str(scoring)}')
        try:
            # y = np.array(y).reshape(-1,1)
            try:
                print(
                    f'[{str(timestamp_with_time_zone())}] [ecma] scores based on grid search {str(regressor.score(x, y))}')
            except:
                pass
            scores = cross_val_score(regressor, x, y, cv=method, scoring=scoring)
        except Exception as e:
            print(f'[{str(timestamp_with_time_zone())}] [ecma] error in cross {str(e)}')
            raise Exception(e)
        print(f'[{str(timestamp_with_time_zone())}] [ecma] cross done {str(scores)}')

        # print(scores)
        # scores = -scores
        # print(scores)
        return {"scores": scores.tolist(), "mean_score": scores.mean()}
    else:
        return {}


async def __validate_test_input(request_data):
    # validate input
    # valid_input = ValidateTestInputSerializer(data=request_data)
    # if not valid_input.is_valid():
    #     raise Exception(valid_input.errors)
    try:
        valid_input = ValidateTestInputSerializer().load(request_data)
    except ValidationError as err:
        return err


async def __validate_train_predict_input(input_jo):
    # validate input
    # valid_input = ValidateTrainInputSerializer(data=request_data)
    # if valid_input.is_valid():
    #     model_base_name = input_jo['model_base_name']
    # else:
    #     raise Exception(valid_input.errors)
    try:
        valid_input = ValidateTrainInputSerializer().load(input_jo)
        model_base_name = input_jo['model_base_name']
    except ValidationError as err:
        return err

    # validate name
    if_has_spaces = bool(re.search(r"\s", model_base_name))
    if if_has_spaces:
        raise Exception("Model name string must not have spaces")

    # validate data
    # valid_data = ValidateTrainTestDataInputSerializer(data=request_data['data'])
    # if valid_data.is_valid():
    #     data = input_jo['data']
    # else:
    #     raise Exception(valid_data.errors)
    try:
        valid_input = ValidateTrainInputSerializer().load(input_jo['data'])
        data = input_jo['data']
        model_base_name = input_jo['model_base_name']
    except ValidationError as err:
        return err

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']

    if len(x_train) != len(y_train):
        raise Exception("x_train and y_train must have the same number of rows.")

    if len(x_train[0]) != len(x_test[0]):
        raise Exception("x_train and x_test must have the same number of columns.")

    # validate model data
    # valid_ml_model = ValidateTrainPredictMlModelSerializer(data=request_data['ml_model'])
    # if not valid_ml_model.is_valid():
    #     raise Exception(valid_ml_model.errors)
    try:
        valid_input = ValidateTrainPredictMlModelSerializer().load(input_jo['ml_model'])
    except ValidationError as err:
        return err


def timestamp_with_time_zone():
    current_time = datetime.now()
    timezone = "Europe/Lisbon"  # Desired timezone
    formated_timestamp = "%Y-%m-%d %H:%M:%S"
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


import logging
import numpy as np
from datetime import datetime
from errors_app.errors import me, mse, mase, rmse, mae, mape, smape, rmspe, rmsse, mape_2, r2, accuracy_score_, \
    _absolute_error, wape
import pytz

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='error.log',
                    filemode='a')


async def calculate(method, data_jo):
    if method == 'POST':
        print(f'[{str(timestamp_with_time_zone())}] [ecma] in post calculate')

        predicted = data_jo['predicted']
        actual = data_jo['actual']
        print(
            f"[{str(timestamp_with_time_zone())}] [ecma] len predicted {str(len(predicted))} and len actual {str(len(actual))}")
        # logging.info(f"data model len x train{str(predicted)} and len y train {str(actual)}")

        if len(predicted) != len(actual):
            print(
                f"[{str(timestamp_with_time_zone())}] [ecma] error: Actual and predicted lists must have the same size.")
            return {"error": "Actual and predicted lists must have the same size."}, 500

        try:
            result = await _calculateError(predicted, actual)
        except Exception as ex:
            try:
                result = await _calculateErrorStr(predicted, actual)
            except Exception as ex:
                print(f"[{str(timestamp_with_time_zone())}] [ecma] error: Unable to calculate errors. " + str(ex))
                return {"error": "Unable to calculate errors. " + str(ex)}, 500
        return result, 200


async def calculate_with_key(method, data_jo, key):
    if method == 'POST':
        print(f'[{str(timestamp_with_time_zone())}] [ecma] in post calculate with key')
        predicted = data_jo['predicted']
        actual = data_jo['actual']
        logging.info(f"data model len x train{str(len(predicted))} and len y train {str(len(actual))}")
        logging.info(f"data model len x train{str(predicted)} and len y train {str(actual)}")

        if len(predicted) != len(actual):
            print(
                f"[{str(timestamp_with_time_zone())}] [ecma] error: Actual and predicted lists must have the same size.")
            return {"error": "Actual and predicted lists must have the same size."}, 500
        possible_bool = ['true', 'false']
        calculate_Normal = False
        try:
            if str(predicted[0]).lower() in possible_bool:
                calculate_Normal = False
            else:
                try:
                    v1 = int(predicted[0])
                    calculate_Normal = True
                except:
                    try:
                        v1 = float(predicted[0])
                        calculate_Normal = True
                    except Exception as ex:
                        # print(ex)
                        v1 = predicted[0]
                        calculate_Normal = False
        except:
            try:
                v1 = int(predicted[0])
                calculate_Normal = True
            except:
                try:
                    v1 = float(predicted[0])
                    calculate_Normal = True
                except Exception as ex:
                    # print(ex)
                    v1 = predicted[0]
                    calculate_Normal = False
        if calculate_Normal:
            try:
                if key == 'me':
                    result = {'me': await me(actual=actual, predicted=predicted)}
                elif key == 'mse':
                    result = {'mse': await mse(actual=actual, predicted=predicted)}
                elif key == 'mase':
                    result = {'mase': await mase(actual=actual, predicted=predicted)}
                elif key == 'rmse':
                    result = {'rmse': await rmse(actual=actual, predicted=predicted)}
                elif key == 'mae':
                    result = {'mae': await mae(actual=actual, predicted=predicted)}
                elif key == 'mape':
                    result = {'mape': await mape(actual=actual, predicted=predicted)}
                elif key == 'mape_2':
                    result = {'mape_2': await mape_2(actual=actual, predicted=predicted)}
                elif key == 'smape':
                    result = {'smape': await smape(actual=actual, predicted=predicted)}
                elif key == 'rmspe':
                    result = {'rmspe': await rmspe(actual=actual, predicted=predicted)}
                elif key == 'rmsse':
                    result = {'rmsse': await rmsse(actual=actual, predicted=predicted)}
                elif key == 'r2':
                    result = {'r2': await r2(actual=actual, predicted=predicted)}
                elif key == 'ae':
                    result = {'ae': await _absolute_error(actual=actual, predicted=predicted)}
                elif key == 'wape':
                    result = {'wape': await wape(actual=actual, predicted=predicted)}
                else:
                    result = {}
            except Exception as ex:
                print(
                    f" [{str(timestamp_with_time_zone())}] [ecma] error: Unable to calculate errors. {str(ex)}, actual {str(actual)} and predicted {str(predicted)}")
                return {"error": "Unable to calculate errors. " + str(ex)}, 500
        else:
            try:
                result = await _calculateErrorStr(predicted, actual)
            except Exception as ex:
                print(
                    f"[{str(timestamp_with_time_zone())}] [ecma] error: Unable to calculate errors with string values. {str(ex)}, actual {str(actual)} and predicted {str(predicted)}")
                return {"error": "Unable to calculate errors with string values. " + str(ex)}, 500
        print(f"[{str(timestamp_with_time_zone())}] [ecma] result {str(result)}")
        return result, 200


async def _calculateError(predict, actual):
    # actual=actual.iloc[:,1]

    metrics = {}

    if len(predict) > 1:
        metrics = {
            'me': await me(actual=actual, predicted=predict),
            'mse': await mse(actual=actual, predicted=predict),
            'mase': await mase(actual=actual, predicted=predict),
            'rmse': await rmse(actual=actual, predicted=predict),
            'mae': await mae(actual=actual, predicted=predict),
            'mape': await mape(actual=actual, predicted=predict),
            'mape_2': await mape_2(actual=actual, predicted=predict),
            'smape': await smape(actual=actual, predicted=predict),
            'rmspe': await rmspe(actual=actual, predicted=predict),
            'rmsse': await rmsse(actual=actual, predicted=predict),
            'r2': await r2(actual=actual, predicted=predict),
            'wape': await wape(actual=actual, predicted=predict)
        }
    else:
        metrics = {
            'me': await me(actual=actual, predicted=predict),
            'mse': await mse(actual=actual, predicted=predict),
            'mase': np.NaN,
            'rmse': await rmse(actual=actual, predicted=predict),
            'mae': await mae(actual=actual, predicted=predict),
            'mape': await mape(actual=actual, predicted=predict),
            'mape_2': await mape_2(actual=actual, predicted=predict),
            'smape': await smape(actual=actual, predicted=predict),
            'rmspe': await rmspe(actual=actual, predicted=predict),
            'rmsse': np.NaN,
            'r2': np.NaN,
            'wape': await wape(actual=actual, predicted=predict)
        }

    for k, v in metrics.items():
        if np.isnan(v):
            metrics[k] = np.float64(0.0)
        elif np.isinf(v):
            metrics[k] = np.float64(0.0)
        if not isinstance(v, np.float64):
            metrics[k] = np.float64(v)
    return metrics


async def _calculateErrorStr(predict, actual):
    metrics = {
        'r2': await accuracy_score_(actual=actual, predicted=predict)
    }
    return metrics


def timestamp_with_time_zone():
    current_time = datetime.now()
    timezone = "Europe/Lisbon"  # Desired timezone
    formated_timestamp = "%Y-%m-%d %H:%M:%S"
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
