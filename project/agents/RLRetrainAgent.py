import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import datetime as dt
from datetime import datetime, timedelta
from peak import Agent, PeriodicBehaviour, Message
import json
import pandas as pd
class RLRetrainAgent(Agent):
    class ReceiveMsg(PeriodicBehaviour):
        async def run(self):
            msg = await self.receive()
            if msg:
                with open('utils_package/config_agents.json') as config_file:
                    config_agents = json.load(config_file)
                with open('utils_package/config_settings.json') as config_file:
                    config_settings = json.load(config_file)
                database_agent = config_agents["database_agent"]
                xai_agent = config_agents["xai_agent"]
                timestamp_format = config_settings["datetime_format"]
                list_metrics = config_settings.get("errors_metric")
                minimal_month = timedelta(days=config_settings.get('minimumDaysReplaceModel'))
                today = datetime.now().strftime(timestamp_format)
                today = datetime.strptime(today, timestamp_format)
                units_one_val_retrain = config_settings.get("units_one_val_retrain")
                min_len_compare_model = config_settings.get('min_len_compare_model')
                feature_deviation_shap = config_settings.get("feature_deviation")
                error_deviation = config_settings.get("error_deviation")
                hourHigh = config_settings.get("hourHigh")
                hourMediumAndLow = config_settings.get("hourMediumAndLow")
                msg_retrain_check_data = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                msg_retrain_check_data.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                msg_ = 'get_models_to_check_retrain'
                msg_ = json.dumps(msg_)
                msg_retrain_check_data.body = msg_
                await self.send(msg_retrain_check_data)
                response = await self.receive()
                if response:
                    if response.get_metadata("performative") == "inform":
                        info_from_database = response.body
                        if info_from_database:
                            info_model = json.loads(info_from_database)
                            for model in info_model:
                                model_id = model['model_id']
                                trained_at = model['registered_at']
                                try:
                                    trained_at = json.loads(trained_at)
                                except:
                                    pass
                                predictions = model['historic_predictions_model']
                                training_dates = model['training_dates']
                                list_periods_low_medium_priority_to_train = []
                                list_periods_high_priority_to_train = []
                                list_periods_severe_priority_to_train = []
                                retrain_reasons_periods_low_medium_priority_to_train = []
                                retrain_reasons_periods_high_priority_to_train = []
                                retrain_reasons_periods_severe_priority_to_train = []
                                predictions = self.filter_predicted_timestamps(training_dates, predictions,
                                                                                   timestamp_format)
                                characteristics = model['characteristics']
                                try:
                                    characteristics = json.loads(characteristics)
                                except:
                                    pass
                                historic_scores_model = model['historic_scores_model']
                                try:
                                    historic_scores_model = json.loads(historic_scores_model)
                                except:
                                    pass
                                #todo obter os errors pelo historic_scores_model e filtrar as well
                                if len(predictions) > 0:
                                    first_entry = predictions[0]
                                    if not isinstance(first_entry, dict):
                                        first_entry = json.loads(first_entry)
                                    unit = first_entry.get("unit")
                                    if unit in units_one_val_retrain:
                                        model_in_list_retrain = False
                                        for dict_entry in predictions:
                                            if not model_in_list_retrain:
                                                if not isinstance(dict_entry, dict):
                                                    dict_entry = json.loads(dict_entry)
                                                column = characteristics.get("target")
                                                dataset_type = characteristics.get("dataset_type")
                                                table = characteristics.get("target_table")
                                                max_time = characteristics.get("frequency")
                                                # calculate the 3 errors
                                                errors = []
                                                errors_entry = dict_entry.get('score')
                                                for metric in list_metrics:
                                                    err_inst = errors_entry[metric]
                                                    errors.append(err_inst)
                                                # if one of them > 25%
                                                for error in errors:
                                                    if error > config_settings.get('trigger_retrain'):
                                                        print('error triggered', error)
                                                        predicted_timestamp = dict_entry.get("predicted_timestamp")
                                                        predicted_timestamp = predicted_timestamp.split(" ")
                                                        start_date = predicted_timestamp[0]
                                                        start_time = predicted_timestamp[1]
                                                        new_start_date, new_start_time = self.find_start_time_predict(
                                                            start_date,
                                                            start_time,
                                                            max_time,
                                                            timestamp_format)
                                                        entry_train = {"id_model": id, "frequency": max_time,
                                                                       "start_date": new_start_date,
                                                                       "start_time": new_start_time,
                                                                       "target": column, "target_table": table,
                                                                       "dataset_type": dataset_type}
                                                        #todo: call training
                                                        break

                                    else:
                                        print('in else')
                                        # predictions só pode ser quando version in historic == models_version
                                        if len(predictions) > min_len_compare_model:
                                            print('model with len sup than limit', id)
                                            #todo: chech if model exists
                                            model_id_found = False
                                            # if result_json:
                                            #     list_models = result_json.get('models')
                                            #     model_id_found = any(item.get('model_id') == id for item in list_models)
                                            model_id = {"model_id": model_id}
                                            msg_register_model = Message(
                                                to=f"{xai_agent}@{self.agent.jid.domain}/{xai_agent}")
                                            msg_register_model.set_metadata("performative",
                                                                                "request")  # Set the "inform" FIPA performative
                                            msg_ = {'register': model_id}
                                            msg_ = json.dumps(msg_)
                                            msg_register_model.body = msg_
                                            await self.send(msg_register_model)
                                            response = await self.receive()
                                            if response:
                                                pr = []
                                                real = []
                                                pr_timestamps = []
                                                last_entry = predictions[-1]
                                                if not isinstance(last_entry, dict):
                                                    last_entry = json.loads(last_entry)
                                                predicted_timestamp = last_entry.get("predicted_timestamp")
                                                predicted_timestamp = predicted_timestamp.split(" ")
                                                start_date = predicted_timestamp[0]
                                                start_time = predicted_timestamp[1]
                                                column = characteristics.get("target")
                                                dataset_type = characteristics.get("dataset_type")
                                                table = characteristics.get("target_table")
                                                max_time = characteristics.get("frequency")
                                                for dict_entry in predictions:
                                                    if not isinstance(dict_entry, dict):
                                                        dict_entry = json.loads(dict_entry)
                                                    pr.append(dict_entry.get("predicted_value"))
                                                    real.append(dict_entry.get("real_value"))
                                                    pr_timestamps.append((dict_entry.get("predicted_timestamp")))
                                                # calculate the 3 errors
                                                errors = []
                                                base_values_real = real[:min_len_compare_model]
                                                base_values_predicted = pr[:min_len_compare_model]
                                                base_values_predicted_timestamps = pr_timestamps[
                                                                                   :min_len_compare_model]
                                                real_values_to_check = real[min_len_compare_model:]
                                                predicted_values_to_check = pr[min_len_compare_model:]
                                                predicted_timestamps_values_to_check = pr_timestamps[
                                                                                       min_len_compare_model:]
                                                base_values_real = np.array(base_values_real)
                                                base_values_predicted = np.array(base_values_predicted)
                                                real_values_to_check = np.array(real_values_to_check)
                                                predicted_values_to_check = np.array(predicted_values_to_check)
                                                initial_error = np.abs(base_values_real - base_values_predicted)
                                                mean_error = initial_error.mean()
                                                std_error = initial_error.std()  # desvio padrão do erro
                                                # calculate global
                                                data_global = {}
                                                model_id = {"model_id": model_id}
                                                msg_global_expl_model = Message(
                                                    to=f"{xai_agent}@{self.agent.jid.domain}/{xai_agent}")
                                                msg_global_expl_model.set_metadata("performative",
                                                                                "request")  # Set the "inform" FIPA performative
                                                msg_ = {'global_explanations': model_id}
                                                msg_ = json.dumps(msg_)
                                                msg_global_expl_model.body = msg_
                                                await self.send(msg_global_expl_model)
                                                response = await self.receive()
                                                if response:
                                                    global_explanations = response.body
                                                    global_explanations = json.loads(global_explanations)
                                                    try:
                                                        results = global_explanations.get("global_stats_raw")
                                                        if results is None:
                                                            results = global_explanations.get("results_raw")
                                                    except:
                                                        results = global_explanations.get("results_raw")
                                                    for k, v in results.items():
                                                        v = float(v.get("avg_impact_shap"))
                                                        data_global[k] = v
                                                # calculate local
                                                data_local = {}
                                                for pr_, timestamp_ in zip(base_values_predicted,
                                                                           base_values_predicted_timestamps):
                                                    data = {"model_id": model_id, "timestamp": timestamp_}
                                                    msg_local_expl_model = Message(
                                                        to=f"{xai_agent}@{self.agent.jid.domain}/{xai_agent}")
                                                    msg_local_expl_model.set_metadata("performative",
                                                                                       "request")  # Set the "inform" FIPA performative
                                                    msg_ = {'local_explanations': data}
                                                    msg_ = json.dumps(msg_)
                                                    msg_local_expl_model.body = msg_
                                                    await self.send(msg_local_expl_model)
                                                    response = await self.receive()
                                                    if response:
                                                        local_explanations = response.body
                                                        local_explanations = json.loads(local_explanations)
                                                        try:
                                                            results = local_explanations.get("class_specific")
                                                            if results is None:
                                                                results = local_explanations.get("results_raw")
                                                            else:
                                                                results.get(str(pr_)).get("results_raw")
                                                        except:
                                                            results = local_explanations.get("results_raw")
                                                        for entry in results:
                                                            if not isinstance(entry, dict):
                                                                entry = json.loads(entry)
                                                            feature_name = entry.get('feature_name')
                                                            feature_value = float(entry.get('feature_influence'))
                                                            if feature_name in list(data_local.keys()):
                                                                values_ = data_local.get(feature_name)
                                                                values_.append(feature_value)
                                                                data_local[feature_name] = values_
                                                            else:
                                                                data_local[feature_name] = [feature_value]
                                                for k, v in data_local.items():
                                                    mean_v = sum(v) / len(v)
                                                    data_local[k] = mean_v
                                                common_keys = data_global.keys() & data_local.keys()
                                                data_shap_values_df = {key: [data_global[key], data_local[key]] for
                                                                       key in common_keys}
                                                shap_values = pd.DataFrame(
                                                    data_shap_values_df)  # first entry global shap values second
                                                # entry mean of the local shap values of the first 96 periods
                                                mean_feature = shap_values.mean()
                                                std_feature = shap_values.std()  # standard deviation of shap value
                                                # feature deviation is from a config file
                                                for pr, real, timestamp in zip(predicted_values_to_check,
                                                                               real_values_to_check,
                                                                               predicted_timestamps_values_to_check):
                                                    # calculate error
                                                    error_ = np.abs(real - pr)
                                                    # calculate shap_value local explanation
                                                    data = {"model_id": model_id, "timestamp": timestamp}
                                                    msg_local_expl_model_pr = Message(
                                                        to=f"{xai_agent}@{self.agent.jid.domain}/{xai_agent}")
                                                    msg_local_expl_model_pr.set_metadata("performative",
                                                                                      "request")  # Set the "inform" FIPA performative
                                                    msg_ = {'local_explanations': data}
                                                    msg_ = json.dumps(msg_)
                                                    msg_local_expl_model_pr.body = msg_
                                                    await self.send(msg_local_expl_model_pr)
                                                    response = await self.receive()
                                                    if response:
                                                        local_explanations = response.body
                                                        local_explanations = json.loads(local_explanations)
                                                        try:
                                                            results = local_explanations.get("class_specific")
                                                            if results is None:
                                                                results = local_explanations.get("results_raw")
                                                            else:
                                                                results.get(str(pr)).get("results_raw")
                                                        except:
                                                            results = local_explanations.get("results_raw")
                                                        # Check if the shap value of the feature deviates from the standard shap value of
                                                        # that feature
                                                        for entry in results:
                                                            if not isinstance(entry, dict):
                                                                entry = json.loads(entry)
                                                            feature_name = entry.get('feature_name')
                                                            feature_value = entry.get('feature_influence')
                                                            value = float(feature_value)
                                                            if feature_name in mean_feature.index.tolist():
                                                                mean_val_feature = mean_feature[feature_name]
                                                                std_val_feature = std_feature[feature_name]
                                                                feature_deviation_with_multiplier = feature_deviation_shap * std_val_feature
                                                                min_interval_shap_retrain = mean_val_feature - feature_deviation_with_multiplier
                                                                max_interval_shap_retrain = mean_val_feature + feature_deviation_with_multiplier
                                                                if value < min_interval_shap_retrain or value > max_interval_shap_retrain:
                                                                    # if np.abs(value - mean_feature[feature_name]) > feature_deviation_with_
                                                                    # multiplier:
                                                                    # se o shap value - media do shap value for superior ao input * o desvio
                                                                    # padrão do shap value precisa de retreinar
                                                                    predicted_timestamp = timestamp.split(" ")
                                                                    start_date = predicted_timestamp[0]
                                                                    start_time = predicted_timestamp[1]
                                                                    new_start_date, new_start_time = self.find_start_time_predict(
                                                                        start_date,
                                                                        start_time, max_time, timestamp_format)
                                                                    entry_train = {"id_model": id,
                                                                                   "frequency": max_time,
                                                                                   "start_date": new_start_date,
                                                                                   "start_time": new_start_time,
                                                                                   "column": column, "table": table,
                                                                                   "dataset_type": dataset_type}
                                                                    feature_deviation_with_multiplier_severe = feature_deviation_shap * (
                                                                            std_val_feature + 2)
                                                                    feature_deviation_with_multiplier_high = feature_deviation_shap * (
                                                                            std_val_feature + 1)
                                                                    min_interval_shap_severe = mean_val_feature - feature_deviation_with_multiplier_severe
                                                                    max_interval_shap_severe = mean_val_feature + feature_deviation_with_multiplier_severe
                                                                    min_interval_shap_high = mean_val_feature - feature_deviation_with_multiplier_high
                                                                    max_interval_shap_high = mean_val_feature + feature_deviation_with_multiplier_high
                                                                    # adicionar o dia que foi adicionado à lista, a magnitude do desvio (
                                                                    # feature_deviation_with_multiplier)
                                                                    if max_interval_shap_high > value > min_interval_shap_high:
                                                                        print('dev low medium')
                                                                        # retrain_reasons = {
                                                                        #     "justification": {
                                                                        #         "retrain_type": "shap_values",
                                                                        #         "mean": mean_val_feature,
                                                                        #         "std": std_val_feature,
                                                                        #         "multiplier": feature_deviation_shap,
                                                                        #         "min": min_interval_shap_retrain,
                                                                        #         "max": max_interval_shap_retrain,
                                                                        #         "triggered_value": value
                                                                        #     },
                                                                        #     "triggered_timestamp": timestamp,
                                                                        #     "retrain_level": "medium_priority",
                                                                        #     "start_retrain": Repo.Repo.get_next_time(
                                                                        #         hourMediumAndLow).strftime(
                                                                        #         formated_timestamp)
                                                                        # }
                                                                        # retrain_reasons_periods_low_medium_priority_to_train.append(
                                                                        #     retrain_reasons)
                                                                        # list_periods_low_medium_priority_to_train.append(
                                                                        #     {"model_id": id,
                                                                        #      "entry_timestamp": Repo.Repo.timestamp_with_time_zone().strftime(
                                                                        #          formated_timestamp),
                                                                        #      "entry_magnitude": feature_deviation_with_multiplier,
                                                                        #      "entry_data": entry_train})
                                                                    elif max_interval_shap_severe > value > min_interval_shap_severe:
                                                                        print('dev high')
                                                                        # retrain_reasons = {
                                                                        #     "justification": {
                                                                        #         "retrain_type": "shap_values",
                                                                        #         "mean": mean_val_feature,
                                                                        #         "std": std_val_feature,
                                                                        #         "multiplier": feature_deviation_shap,
                                                                        #         "min": min_interval_shap_retrain,
                                                                        #         "max": max_interval_shap_retrain,
                                                                        #         "triggered_value": value
                                                                        #     },
                                                                        #     "triggered_timestamp": timestamp,
                                                                        #     "retrain_level": "high_priority",
                                                                        #     "start_retrain": Repo.Repo.get_next_time(
                                                                        #         hourHigh).strftime(
                                                                        #         formated_timestamp)
                                                                        # }
                                                                        # retrain_reasons_periods_high_priority_to_train.append(
                                                                        #     retrain_reasons)
                                                                        # list_periods_high_priority_to_train.append(
                                                                        #     {"model_id": id,
                                                                        #      "entry_timestamp": Repo.Repo.timestamp_with_time_zone().strftime(
                                                                        #          formated_timestamp),
                                                                        #      "entry_magnitude": feature_deviation_with_multiplier,
                                                                        #      "entry_data": entry_train})
                                                                    else:
                                                                        print('dev severe')
                                                                        # retrain_reasons = {
                                                                        #     "justification": {
                                                                        #         "retrain_type": "shap_values",
                                                                        #         "mean": mean_val_feature,
                                                                        #         "std": std_val_feature,
                                                                        #         "multiplier": feature_deviation_shap,
                                                                        #         "min": min_interval_shap_retrain,
                                                                        #         "max": max_interval_shap_retrain,
                                                                        #         "triggered_value": value
                                                                        #     },
                                                                        #     "triggered_timestamp": timestamp,
                                                                        #     "retrain_level": "severe_priority",
                                                                        #     "start_retrain": Repo.Repo.timestamp_with_time_zone().strftime(
                                                                        #         formated_timestamp)
                                                                        # }
                                                                        # retrain_reasons_periods_severe_priority_to_train.append(
                                                                        #     retrain_reasons)
                                                                        # list_periods_severe_priority_to_train.append(
                                                                        #     {"model_id": id,
                                                                        #      "entry_timestamp": Repo.Repo.timestamp_with_time_zone().strftime(
                                                                        #          formated_timestamp),
                                                                        #      "entry_magnitude": feature_deviation_with_multiplier,
                                                                        #      "entry_data": entry_train})
                                                    # Check if the prediction error deviates from the standard

                                                    error_deviation_with_multiplier = error_deviation * std_error
                                                    min_interval_error_retrain = mean_error - error_deviation_with_multiplier
                                                    max_interval_error_retrain = mean_error + error_deviation_with_multiplier
                                                    if error_ < min_interval_error_retrain or error_ > max_interval_error_retrain:
                                                        # if np.abs(
                                                        #         error_ - mean_error) > error_deviation_with_multiplier:  # mesma lógica que a feature
                                                        # error deviation is from a config file
                                                        predicted_timestamp = timestamp.split(" ")
                                                        start_date = predicted_timestamp[0]
                                                        start_time = predicted_timestamp[1]
                                                        new_start_date, new_start_time = self.find_start_time_predict(
                                                            start_date,
                                                            start_time,
                                                            max_time,
                                                            timestamp_format)
                                                        entry_train = {"id_model": id, "frequency": max_time,
                                                                       "start_date": new_start_date,
                                                                       "start_time": new_start_time,
                                                                       "column": column, "table": table,
                                                                       "dataset_type": dataset_type}

                                                        error_deviation_with_multiplier_high = error_deviation * (
                                                                    std_error + 1)
                                                        error_deviation_with_multiplier_severe = error_deviation * (
                                                                    std_error + 2)
                                                        min_interval_error_severe = mean_error - error_deviation_with_multiplier_severe
                                                        max_interval_error_severe = mean_error + error_deviation_with_multiplier_severe
                                                        min_interval_error_high = mean_error - error_deviation_with_multiplier_high
                                                        max_interval_error_high = mean_error + error_deviation_with_multiplier_high
                                                        if max_interval_error_high > error_ > min_interval_error_high:
                                                            print('dev low medium')
                                                            # retrain_reasons = {
                                                            #     "justification": {
                                                            #         "retrain_type": "error",
                                                            #         "mean": mean_error,
                                                            #         "std": std_error,
                                                            #         "multiplier": error_deviation,
                                                            #         "min": min_interval_error_retrain,
                                                            #         "max": max_interval_error_retrain,
                                                            #         "triggered_value": error_
                                                            #     },
                                                            #     "triggered_timestamp": timestamp,
                                                            #     "retrain_level": "medium_priority",
                                                            #     "start_retrain": Repo.Repo.get_next_time(
                                                            #         hourMediumAndLow).strftime(
                                                            #         formated_timestamp)
                                                            # }
                                                            # retrain_reasons_periods_low_medium_priority_to_train.append(
                                                            #     retrain_reasons)
                                                            # list_periods_low_medium_priority_to_train.append(
                                                            #     {"model_id": id,
                                                            #      "entry_timestamp": Repo.Repo.timestamp_with_time_zone().strftime(
                                                            #          formated_timestamp),
                                                            #      "entry_magnitude": error_deviation_with_multiplier,
                                                            #      "entry_data": entry_train})
                                                        elif max_interval_error_severe > error_ > min_interval_error_severe:
                                                            print('dev high')
                                                            # retrain_reasons = {
                                                            #     "justification": {
                                                            #         "retrain_type": "error",
                                                            #         "mean": mean_error,
                                                            #         "std": std_error,
                                                            #         "multiplier": error_deviation,
                                                            #         "min": min_interval_error_retrain,
                                                            #         "max": max_interval_error_retrain,
                                                            #         "triggered_value": error_
                                                            #     },
                                                            #     "triggered_timestamp": timestamp,
                                                            #     "retrain_level": "high_priority",
                                                            #     "start_retrain": Repo.Repo.get_next_time(
                                                            #         hourHigh).strftime(
                                                            #         formated_timestamp)
                                                            # }
                                                            # retrain_reasons_periods_high_priority_to_train.append(
                                                            #     retrain_reasons)
                                                            # list_periods_high_priority_to_train.append(
                                                            #     {"model_id": id,
                                                            #      "entry_timestamp": Repo.Repo.timestamp_with_time_zone().strftime(
                                                            #          formated_timestamp),
                                                            #      "entry_magnitude": error_deviation_with_multiplier,
                                                            #      "entry_data": entry_train})
                                                        else:
                                                            print('dev severe')
                                                            # retrain_reasons = {
                                                            #     "justification": {
                                                            #         "retrain_type": "error",
                                                            #         "mean": mean_error,
                                                            #         "std": std_error,
                                                            #         "multiplier": error_deviation,
                                                            #         "min": min_interval_error_retrain,
                                                            #         "max": max_interval_error_retrain,
                                                            #         "triggered_value": error_
                                                            #     },
                                                            #     "triggered_timestamp": timestamp,
                                                            #     "retrain_level": "severe_priority",
                                                            #     "start_retrain": Repo.Repo.timestamp_with_time_zone().strftime(
                                                            #         formated_timestamp)
                                                            # }
                                                            # retrain_reasons_periods_severe_priority_to_train.append(
                                                            #     retrain_reasons)
                                                            # list_periods_severe_priority_to_train.append(
                                                            #     {"model_id": id,
                                                            #      "entry_timestamp": Repo.Repo.timestamp_with_time_zone().strftime(
                                                            #          formated_timestamp),
                                                            #      "entry_magnitude": error_deviation_with_multiplier,
                                                            #      "entry_data": entry_train})

                                                # Check if the model was trained one month ago
                                                diff_training = today - trained_at
                                                if diff_training > minimal_month:
                                                    timestamp_ = start_date + " " + start_time
                                                    print('low priority')
                                                    # retrain_reasons = {
                                                    #     "justification": {
                                                    #         "retrain_type": "obsolete",
                                                    #         "days_last_train": diff_training
                                                    #     },
                                                    #     "triggered_timestamp": timestamp_,
                                                    #     "retrain_level": "low_priority",
                                                    #     "start_retrain": Repo.Repo.get_next_time(
                                                    #         hourMediumAndLow).strftime(
                                                    #         formated_timestamp)
                                                    # }
                                                    # retrain_reasons_periods_low_medium_priority_to_train.append(
                                                    #     retrain_reasons)
                                                    # new_start_date, new_start_time = Services.find_start_time_predict(
                                                    #     start_date,
                                                    #     start_time, max_time,
                                                    #     formated_timestamp)
                                                    # entry_train = {"id_model": id, "frequency": max_time,
                                                    #                "start_date": new_start_date,
                                                    #                "start_time": new_start_time,
                                                    #                "column": column, "table": table,
                                                    #                "dataset_type": dataset_type}
                                                    # list_periods_low_medium_priority_to_train.append(
                                                    #     {"model_id": id,
                                                    #      "entry_timestamp": Repo.Repo.timestamp_with_time_zone().strftime(
                                                    #          formated_timestamp),
                                                    #      "entry_magnitude": diff_training,
                                                    #      "entry_data": entry_train})

                                                # Check the overall metric of the model
                                                last_entry = historic_scores_model[-1]
                                                if not isinstance(last_entry, dict):
                                                    last_entry = json.loads(last_entry)
                                                for metric in list_metrics:
                                                    err_inst = last_entry[metric]
                                                    errors.append(err_inst)
                                                # if one of them > 25%
                                                for error in errors:
                                                    if error > config_settings.get('trigger_retrain'):
                                                        timestamp_ = start_date + " " + start_time
                                                        print('high priority')
                                                        # retrain_reasons = {
                                                        #     "justification": {
                                                        #         "retrain_type": "error",
                                                        #         "max_accepted": config_settings.get(
                                                        #             'trigger_retrain'),
                                                        #         "triggered_value": error
                                                        #     },
                                                        #     "triggered_timestamp": timestamp_,
                                                        #     "retrain_level": "high_priority",
                                                        #     "start_retrain": Repo.Repo.get_next_time(
                                                        #         hourHigh).strftime(
                                                        #         formated_timestamp)
                                                        # }
                                                        # retrain_reasons_periods_high_priority_to_train.append(
                                                        #     retrain_reasons)
                                                        # new_start_date, new_start_time = Services.find_start_time_predict(
                                                        #     start_date,
                                                        #     start_time,
                                                        #     max_time,
                                                        #     formated_timestamp)
                                                        # entry_train = {"id_model": id, "frequency": max_time,
                                                        #                "start_date": new_start_date,
                                                        #                "start_time": new_start_time,
                                                        #                "column": column, "table": table,
                                                        #                "dataset_type": dataset_type}
                                                        # list_periods_high_priority_to_train.append(
                                                        #     {"model_id": id,
                                                        #      "entry_timestamp": Repo.Repo.timestamp_with_time_zone().strftime(
                                                        #          formated_timestamp),
                                                        #      "entry_magnitude": error,
                                                        #      "entry_data": entry_train})

                                            len_train_severe_priority = len(list_periods_severe_priority_to_train)
                                            len_train_high_priority = len(list_periods_high_priority_to_train)
                                            len_train_low_medium_priority = len(
                                                list_periods_low_medium_priority_to_train)
                                            # if len_train_severe_priority > 0 or len_train_high_priority > 0 or len_train_low_medium_priority > 0:
                                            #     print('update flag of retrain')
                                            #     await Repo.Repo.change_flag_model(id)
                                            #     historic_reasons = await Repo.Repo.getHistoricReasons(id)
                                            #     max_counter = max(len_train_low_medium_priority,
                                            #                       len_train_high_priority,
                                            #                       len_train_severe_priority)
                                            #
                                            #     if len_train_severe_priority == len_train_high_priority == len_train_low_medium_priority:
                                            #         print(
                                            #             'all the same so model will be trained with severe priority')
                                            #         latest_index, latest_entry = Services.find_latest_entry(
                                            #             list_periods_severe_priority_to_train, formated_timestamp)
                                            #         train_severe_priority.append(latest_entry)
                                            #         retrain_reason = \
                                            #         retrain_reasons_periods_severe_priority_to_train[latest_index]
                                            #         historic_reasons.append(retrain_reason)
                                            #         await Repo.Repo.updateHistoricReasons(id, historic_reasons)
                                            #     elif len_train_severe_priority == len_train_high_priority and len_train_high_priority > len_train_low_medium_priority:
                                            #         print(
                                            #             'len severe and high are same so model will be trained with severe priority')
                                            #         latest_index, latest_entry = Services.find_latest_entry(
                                            #             list_periods_severe_priority_to_train, formated_timestamp)
                                            #         train_severe_priority.append(latest_entry)
                                            #         retrain_reason = \
                                            #         retrain_reasons_periods_severe_priority_to_train[latest_index]
                                            #         historic_reasons.append(retrain_reason)
                                            #         await Repo.Repo.updateHistoricReasons(id, historic_reasons)
                                            #     elif len_train_high_priority == len_train_low_medium_priority and len_train_high_priority > len_train_severe_priority:
                                            #         print(
                                            #             'len high and low are the same so model will be trained with high priority')
                                            #         latest_index, latest_entry = Services.find_latest_entry(
                                            #             list_periods_high_priority_to_train, formated_timestamp)
                                            #         train_high_priority.append(latest_entry)
                                            #         retrain_reason = retrain_reasons_periods_high_priority_to_train[
                                            #             latest_index]
                                            #         historic_reasons.append(retrain_reason)
                                            #         await Repo.Repo.updateHistoricReasons(id, historic_reasons)
                                            #     elif len_train_severe_priority == len_train_low_medium_priority and len_train_low_medium_priority > len_train_high_priority:
                                            #         print(
                                            #             'len severe and low are the same so model will be trained with severe priority')
                                            #         latest_index, latest_entry = Services.find_latest_entry(
                                            #             list_periods_severe_priority_to_train, formated_timestamp)
                                            #         train_severe_priority.append(latest_entry)
                                            #         retrain_reason = \
                                            #         retrain_reasons_periods_severe_priority_to_train[latest_index]
                                            #         historic_reasons.append(retrain_reason)
                                            #         await Repo.Repo.updateHistoricReasons(id, historic_reasons)
                                            #     elif len_train_severe_priority == max_counter:
                                            #         print(
                                            #             'model will be trained with severe priority because of max counter')
                                            #         latest_index, latest_entry = Services.find_latest_entry(
                                            #             list_periods_severe_priority_to_train, formated_timestamp)
                                            #         train_severe_priority.append(latest_entry)
                                            #         retrain_reason = \
                                            #         retrain_reasons_periods_severe_priority_to_train[latest_index]
                                            #         historic_reasons.append(retrain_reason)
                                            #         await Repo.Repo.updateHistoricReasons(id, historic_reasons)
                                            #     elif len_train_high_priority == max_counter:
                                            #         print(
                                            #             'model will be trained with high priority because of max counter')
                                            #         latest_index, latest_entry = Services.find_latest_entry(
                                            #             list_periods_high_priority_to_train, formated_timestamp)
                                            #         train_high_priority.append(latest_entry)
                                            #         retrain_reason = retrain_reasons_periods_high_priority_to_train[
                                            #             latest_index]
                                            #         historic_reasons.append(retrain_reason)
                                            #         await Repo.Repo.updateHistoricReasons(id, historic_reasons)
                                            #     elif len_train_low_medium_priority == max_counter:
                                            #         print(
                                            #             'model will be trained with low priority because of max counter')
                                            #         latest_index, latest_entry = Services.find_latest_entry(
                                            #             list_periods_low_medium_priority_to_train,
                                            #             formated_timestamp)
                                            #         train_medium_low_priority.append(latest_entry)
                                            #         retrain_reason = \
                                            #         retrain_reasons_periods_low_medium_priority_to_train[
                                            #             latest_index]
                                            #         historic_reasons.append(retrain_reason)
                                            #         await Repo.Repo.updateHistoricReasons(id, historic_reasons)
                                # todo: this is the call to start the reinforcement learning process
                                #get model's global explanations
                                # for each predicted value already with the 'real_value'
                                # get the local explanations
                                #from the time of retrain, % of deviations, e time of retrain, determine using RL if it is to retrain and the reward
                                #update training dates to send in the request
                                #if true, retrain the model
                                #set flag_training to  True

        def filter_predicted_timestamps(self, trainingDate, predicted_timestamps, formated_timestamp):
            end_datetime_str = f"{trainingDate['end_date']} {trainingDate['end_time']}"
            end_datetime = datetime.strptime(end_datetime_str, formated_timestamp)
            # Filter the predicted timestamps that are before the training date
            filtered_list = [
                entry for entry in predicted_timestamps
                if datetime.strptime(entry["predicted_timestamp"], formated_timestamp) > end_datetime
            ]
            return filtered_list


        def find_start_time_predict(self, start_date, start_time, frequency, datetime_format):
            timestamp_str = f"{start_date} {start_time}"
            datetime_obj = datetime.strptime(timestamp_str, datetime_format)
            datetime_obj = datetime_obj + timedelta(minutes=frequency)
            timestamp_str = datetime_obj.strftime(datetime_format)
            timestamp_str = timestamp_str.split(" ")
            return timestamp_str[0], timestamp_str[1]

    async def setup(self):
        b = self.ReceiveMsg(period=1)
        self.add_behaviour(b)
