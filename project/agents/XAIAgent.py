import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import datetime as dt
from peak import Agent, PeriodicBehaviour, Message
import json
import shap
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import openai
import base64
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import pickle


class XAIAgent(Agent):
    class ReceiveMsg(PeriodicBehaviour):
        async def run(self):
            msg = await self.receive()
            if msg:
                data = json.loads(msg.body)
                key = ''
                if 'register' in data:
                    data = data['register']
                    key = 'register'
                elif 'global_explanations' in msg.body:
                    data = data['global_explanations']
                    key = 'global_explanations'
                elif 'local_explanations' in msg.body:
                    data = data['local_explanations']
                    key = 'local_explanations'
                model_id = data.get("model_id")
                predicted_value = data.get("predicted_value")
                target = data.get('target')
                with open('utils_package/config_agents.json') as config_file:
                    config_agents = json.load(config_file)
                with open('utils_package/config_settings.json') as config_file:
                    config_settings = json.load(config_file)
                database_agent = config_agents["database_agent"]
                target_agent = config_agents["target_agent"]
                timestamp_format = config_settings["datetime_format"]
                msg_retrain_check_data = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                msg_retrain_check_data.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                msg_ = {'get_data_xai': model_id}
                msg_ = json.dumps(msg_)
                msg_retrain_check_data.body = msg_
                await self.send(msg_retrain_check_data)
                response = await self.receive()
                info_from_database = None
                if response:
                    if response.get_metadata("performative") == "inform":
                        info_from_database = response.body
                if info_from_database:
                    SHAP_N_SAMPLES = config_settings['shap']['n_samples']
                    NUMBER_TOP_FEATURES_ANALYSIS = config_settings['shap']['nr_top_features_analysis']
                    model_info = json.loads(info_from_database)
                    explainer, base_values, shap_values, classes = self.get_explainer(model_info, SHAP_N_SAMPLES)
                    if key == 'register':
                        msg = {'register_model': {'model_id': model_id}}
                        msg_ = json.dumps(msg)
                        await self.save_to_database(msg_, database_agent)
                    elif key == 'global_explanations':
                        response, save = self.shap_global_explanation(model_info, shap_values,
                                                                      NUMBER_TOP_FEATURES_ANALYSIS)
                        if save:
                            msg = {'save_global_explanations': response}
                            msg_ = json.dumps(msg)
                            await self.save_to_database(msg_, database_agent)

                    elif key == 'local_explanations':
                        timestamp = data.get("timestamp")
                        historic_normalized_test_data = model_info.get("historic_norm_test_data")
                        historic_predictions = model_info.get("historic_predictions_model")
                        forecast_data = self.get_forecast_data_of_timestamp(timestamp, historic_normalized_test_data,
                                                                            historic_predictions, target_agent,
                                                                            timestamp_format)
                        response, save = self.shap_local_explanation(timestamp, model_info, forecast_data)
                        if save:
                            local_explanations = save['local_explanations']
                            msg = {'update_local_explanations': local_explanations}
                            msg_ = json.dumps(msg)
                            await self.save_to_database(msg_, database_agent)

        def shap_global_explanation(self, model_info, shap_values, number_of_top_features):
            if 'global_explanations' in model_info and model_info["global_explanations"]:
                return model_info["global_explanations"], False
            else:
                classes = model_info["classes"]
                feature_names = model_info["columns_names"]
                model_name = model_info["model_name"]
                target_name = model_info["target_name"]

                response = self.get_model_global_analysis(model_name, shap_values,
                                                          target_name, feature_names, classes, number_of_top_features)

                return response, True

        def shap_local_explanation(self, timestamp, model_info, forecast_data):
            flag_create_explanation = False
            if 'local_explanations' in model_info:
                local_explanations = model_info["local_explanations"]

                if timestamp in local_explanations:
                    response = local_explanations[timestamp]
                    model_info = None
                else:
                    flag_create_explanation = True
            else:
                model_info["local_explanations"] = {}
                flag_create_explanation = True

            if flag_create_explanation:
                x_used = np.array(forecast_data['x_test_norm']).reshape(1, -1)
                explainer = model_info["explainer"]

                model_name = model_info["model_name"]
                base_values = model_info["base_values"]
                target_name = model_info["target_name"]
                feature_names = model_info['columns_names']
                classes_names = model_info['classes']
                y_real = forecast_data['y_real']
                y_predicted = forecast_data['y_predicted']

                response = self.get_dynamic_local_analysis(
                    explainer, base_values, x_used, y_predicted, y_real, feature_names,
                    model_name, target_name, classes_names)

                model_info["local_explanations"][timestamp] = response
            return response, model_info

        def get_explainer(self, model_info, SHAP_N_SAMPLES):
            if 'explainer' in model_info:
                return model_info["explainer"], model_info['base_values'], model_info['global_shap_values'], model_info[
                    'classes']
            else:
                x_train = model_info['x_train_data_norm']
                background_reduced = self.get_kmeans_sample(x_train, SHAP_N_SAMPLES)
                background_reduced = np.array(background_reduced)
                model_class_name = model_info['ml_model']
                model_binary = model_info['model_binary'].encode('latin1')
                model = pickle.loads(model_binary)
                if hasattr(model, 'classes_'):
                    classes = model.classes_.tolist()  # Convert to list for JSON serialization
                else:
                    classes = None
                if model_class_name == 'KNNR':
                    explainer = shap.KernelExplainer(model.predict, background_reduced)
                elif model_class_name == 'SVR':
                    explainer = shap.KernelExplainer(model.predict, background_reduced)
                elif model_class_name == 'MLPR':
                    explainer = shap.KernelExplainer(model.predict, background_reduced)
                elif model_class_name == 'RFR':
                    explainer = shap.PermutationExplainer(model.predict, background_reduced)
                elif model_class_name == 'KNNC':
                    explainer = shap.KernelExplainer(model.predict_proba, background_reduced)
                elif model_class_name == 'SVC':
                    explainer = shap.KernelExplainer(model.predict, background_reduced)
                elif model_class_name == 'MLPC':
                    explainer = shap.KernelExplainer(model.predict_proba, background_reduced)
                elif model_class_name == 'RFC':
                    explainer = shap.PermutationExplainer(model.predict_proba, background_reduced)

                explainer_data = np.array(background_reduced)
                shap_values = explainer.shap_values(explainer_data)  #global_shap_values

                if hasattr(explainer, 'expected_value'):
                    base_values = explainer.expected_value if isinstance(explainer.expected_value,
                                                                         (list, np.ndarray)) else [
                        explainer.expected_value]
                else:
                    base_values = [np.mean(model.predict(x_train))]
                return explainer, base_values, shap_values, classes

        def get_kmeans_sample(self, data, n_clusters):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            kmeans.fit(data)
            cluster_centers = kmeans.cluster_centers_
            return cluster_centers

        def get_model_global_analysis(self, model_name, shap_values, target_feature_name, feature_names,
                                      classes_names, number_top_features):
            if len(shap_values.shape) > 2:
                class_results = {}
                mean_abs_shap_values_all_classes = np.mean(np.abs(shap_values), axis=(0, -1))
                global_feature_importance = dict(zip(feature_names, mean_abs_shap_values_all_classes))
                global_sorted_features = sorted(global_feature_importance, key=global_feature_importance.get,
                                                reverse=True)

                global_stats_raw, global_stats_str = self.get_global_stats_info(
                    shap_values.reshape(-1, shap_values.shape[-2]),  # Reshape to combine all class instances
                    feature_names,
                    target_feature_name,
                    global_sorted_features,
                    global_feature_importance
                )

                global_basic_text = self.generate_model_global_basic_explanation(model_name, feature_names,
                                                                                 global_sorted_features)
                global_technical_text = self.generate_model_global_technical_explanation(model_name, feature_names,
                                                                                         global_sorted_features,
                                                                                         global_stats_raw,
                                                                                         number_top_features)

                global_result = {
                    "global_stats_raw": global_stats_raw,
                    "global_stats_str": global_stats_str,
                    "technical_text": global_technical_text,
                    "basic_text": global_basic_text,
                }

                for class_index, class_name in enumerate(classes_names):
                    class_shap_values = shap_values[:, :, class_index]
                    mean_abs_shap_values = np.mean(np.abs(class_shap_values), axis=0)
                    class_feature_importance = dict(zip(feature_names, mean_abs_shap_values))
                    class_sorted_features = sorted(class_feature_importance, key=class_feature_importance.get,
                                                   reverse=True)

                    global_stats_raw, global_stats_str = self.get_global_stats_info(class_shap_values, feature_names,
                                                                                    target_feature_name,
                                                                                    class_sorted_features,
                                                                                    class_feature_importance)
                    basic_text = self.generate_model_global_basic_explanation(model_name, feature_names,
                                                                              class_sorted_features,
                                                                              class_name)
                    technical_text = self.generate_model_global_technical_explanation(model_name, feature_names,
                                                                                      class_sorted_features,
                                                                                      global_stats_raw, class_name,
                                                                                      number_top_features)

                    result = {
                        "results_raw": global_stats_raw,
                        "results_str": global_stats_str,
                        "technical_text": technical_text,
                        "basic_text": basic_text
                    }

                    class_results[class_name] = result

                global_result["class_specific"] = class_results

                return global_result

            else:
                mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
                feature_importance = dict(zip(feature_names, mean_abs_shap_values))
                sorted_features = sorted(feature_importance, key=feature_importance.get, reverse=True)

                global_stats_raw, global_stats_str = self.get_global_stats_info(shap_values, feature_names,
                                                                                target_feature_name,
                                                                                sorted_features, feature_importance)
                basic_text = self.generate_model_global_basic_explanation(model_name, feature_names, sorted_features)
                technical_text = self.generate_model_global_technical_explanation(model_name, feature_names,
                                                                                  sorted_features,
                                                                                  global_stats_raw, number_top_features)

                result = {
                    "results_raw": global_stats_raw,
                    "results_str": global_stats_str,
                    "technical_text": technical_text,
                    "basic_text": basic_text
                }

                return result

        def calculate_shap_statistics(self, shap_values, feature_name, feature_names):
            feature_index = feature_names.index(feature_name)
            feature_shap_values = shap_values[:, feature_index]
            positive_impact_percentage = np.sum(feature_shap_values > 0) / len(feature_shap_values) * 100
            negative_impact_percentage = np.sum(feature_shap_values < 0) / len(feature_shap_values) * 100
            sparsity = np.sum(feature_shap_values == 0) / len(feature_shap_values) * 100
            skewness = pd.Series(feature_shap_values).skew()
            percentiles = np.percentile(feature_shap_values, [25, 50, 75])
            outliers = len(
                feature_shap_values[(feature_shap_values > percentiles[2]) | (feature_shap_values < percentiles[0])])

            return {
                "positive_impact_percentage": positive_impact_percentage,
                "negative_impact_percentage": negative_impact_percentage,
                "sparsity": sparsity,
                "skewness": skewness,
                "percentiles": percentiles,
                "outliers": outliers
            }

        def generate_model_global_basic_explanation(self, model_name, feature_names, sorted_features, classe_name=None):
            most_significant_feature = sorted_features[0]
            least_significant_feature = sorted_features[-1]

            text = ""
            if classe_name:
                text += f"In the model {model_name}, the class '{classe_name}' "
            else:
                text += f"The model ({model_name}) "

            text += f"uses {len(feature_names)} features to make predictions. "
            text += f"The most important feature identified is {most_significant_feature}, which has a considerable effect on the model's output. "
            text += f"The least significant feature is {least_significant_feature}"

            return text

        def generate_model_global_technical_explanation(self, model_name, feature_names, sorted_features, stats,
                                                        NUMBER_TOP_FEATURES_ANALYSIS, classe_name=None):
            most_significant_feature = sorted_features[0]
            least_significant_feature = sorted_features[-1]

            text = ""
            if classe_name:
                text += f"In the model {model_name}, the class '{classe_name}' "
            else:
                text += f"The model ({model_name}) "

            text += f"utilizes a variety of {len(feature_names)} features, each with differing levels of impact on the predictions. "
            text += f"Upon analysis, {NUMBER_TOP_FEATURES_ANALYSIS} features have been identified as having significant influence on the model's output. "
            text += f"The most significant feature is {most_significant_feature}. "
            text += f"Feature {most_significant_feature} exhibits {('rightward' if stats[most_significant_feature]['skewness'] > 0 else 'leftward')} skewness with a measure of {stats[most_significant_feature]['skewness']:.2f}, "
            text += f"indicating {'more frequent unusually high' if stats[most_significant_feature]['skewness'] > 0 else 'more frequent unusually low'} values in its impact. "
            text += f"Large gaps in SHAP values indicate {most_significant_feature} exhibits sparsity or contains isolated values. "
            text += f"The least significant feature is {least_significant_feature}, which has the smallest impact on the model's predictions. "

            return text

        def generate_model_local_basic_explanation(self, model_name, feature_names, sorted_features):
            most_significant_feature = sorted_features[0]["feature_name"]
            least_significant_feature = sorted_features[-1]["feature_name"]

            text = f"The model ({model_name}) uses multiple {len(feature_names)} features to make predictions. "
            text += (
                f"The most important feature identified for this specific prediction is {most_significant_feature}, while "
                f"the least significant feature is {least_significant_feature}.")

            return text

        def generate_model_local_technical_explanation(self, classes_names, model_name, feature_names, sorted_features,
                                                       stats,
                                                       increasing_features,
                                                       decreasing_features, base_value, predicted_value, true_value):
            most_significant_feature = sorted_features[0]["feature_name"]
            least_significant_feature = sorted_features[-1]["feature_name"]

            error_analysis_text = f"The analysis of the model's prediction reveals a predicted value of {predicted_value:.4f} compared to a base value of {base_value:.2f}.\n"

            if increasing_features:
                feature_name = increasing_features[0]['feature_name']
                percentage_impact = increasing_features[0]['percentage_impact']
                error_analysis_text += f"Among the features analyzed, {feature_name} stands out as having the most positive influence, contributing to an increase in the output by approximately {percentage_impact:.4f}%. "

            if decreasing_features:
                feature_name = decreasing_features[0]['feature_name']
                percentage_impact = decreasing_features[0]['percentage_impact']
                error_analysis_text += f"Conversely, {feature_name} exerts the most negative influence, leading to a reduction in the output by approximately {percentage_impact:.4f}%."

            if not classes_names:
                absolute_error = abs(predicted_value - true_value)
                relative_error = absolute_error / abs(true_value) * 100

                error_analysis_text += f"\nThe absolute error for this prediction is {absolute_error:.2f} and the relative error is {relative_error:.2f}%. "
                error_threshold = 0.1
                if absolute_error > error_threshold:
                    error_analysis_text += (
                        f"The observed error of {absolute_error:.2f} exceeds the threshold of {error_threshold}, "
                        "indicating a potential discrepancy between the model’s prediction and the actual outcome. "
                    )
                else:
                    error_analysis_text += (
                        f"The observed error of {absolute_error:.2f} is within the acceptable threshold of {error_threshold}, "
                        "suggesting that the model’s prediction is reasonably accurate for this instance. "
                    )

            stats = next(item for item in stats if item["feature_name"] == most_significant_feature)

            text = f"The model ({model_name}) utilizes a variety of {len(feature_names)} features, each with differing levels of impact on the predictions. "
            text += f"The most significant feature for this prediction is {most_significant_feature}, which exhibits "
            text += f"{'rightward' if stats['feature_influence'] > 0 else 'leftward'} influence with a measure of {stats['feature_influence']:.2f}. "
            text += f"The least significant feature is {least_significant_feature}, which has the smallest impact on the model's predictions."

            return error_analysis_text + "\n" + text

        def get_global_stats_info(self, shap_values, feature_names, target_feature_name, sorted_features,
                                  feature_importance):
            global_stats_raw = {}
            global_stats_str = ""

            for rank, feature in enumerate(sorted_features[:len(feature_names)], start=1):
                importance = feature_importance[feature]
                stats = self.calculate_shap_statistics(shap_values, feature, feature_names)

                global_stats_raw[f"{feature}"] = {
                    "rank": rank,
                    "avg_impact_shap": round(importance, 2),
                    "pos_impact_instances_pct": round(stats['positive_impact_percentage'], 2),
                    "neg_impact_instances_pct": round(stats['negative_impact_percentage'], 2),
                    "sparsity_pct": round(stats['sparsity'], 2),
                    "skewness": round(stats['skewness'], 2),
                    "percentile_25": round(stats['percentiles'][0], 2),
                    "percentile_50": round(stats['percentiles'][1], 2),
                    "percentile_75": round(stats['percentiles'][2], 2),
                    "outliers_count": stats['outliers'],
                    "outliers_pct": round(stats['outliers'] / len(shap_values) * 100, 2)
                }

                global_stats_str += f"Forecasting Target Feature: {target_feature_name}\n"
                global_stats_str += (
                    f"Feature: {feature}\n"
                    f"Rank: {rank}\n"
                    f"Average Impact (SHAP): {importance:.2f}\n"
                    f"Positive Impact Instances (%): {stats['positive_impact_percentage']:.2f}\n"
                    f"Negative Impact Instances (%): {stats['negative_impact_percentage']:.2f}\n"
                    f"Predominantly, high values of {feature} (red) increase the output.\n"
                    f"Predominantly, low values of {feature} (blue) decrease the output.\n"
                    f"Sparsity or isolated values observed in SHAP values for {feature} ({stats['sparsity']:.2f}%).\n"
                    f"Rightward skewed distribution of SHAP values for {feature} (skewness: {stats['skewness']:.2f}).\n"
                    f"Percentiles (25th, 50th, 75th): {stats['percentiles'][0]:.2f}, {stats['percentiles'][1]:.2f}, {stats['percentiles'][2]:.2f}\n"
                    f"Outliers: {stats['outliers']} instances ({(stats['outliers'] / len(shap_values) * 100):.2f}%).\n"
                    "\n"
                )

            return global_stats_raw, global_stats_str

        def get_local_stats_info(self, shap_values_specific_raw, feature_names, total_shap_value, x_data):
            local_stats_raw = []
            local_stats_str = ""

            for i, feature_name in enumerate(feature_names):
                feature_shap_value = shap_values_specific_raw[i]
                feature_value = x_data.iloc[0][feature_name]

                percentage_impact = (abs(feature_shap_value) / total_shap_value) * 100
                direction = "increases" if feature_shap_value > 0 else "decreases"
                color = "red" if feature_shap_value > 0 else "blue"

                feature_info = {
                    "feature_name": feature_name,
                    "color": color,
                    "direction": direction,
                    "feature_influence": abs(feature_shap_value),
                    "feature_value": feature_value,
                    "percentage_impact": percentage_impact
                }

                local_stats_raw.append(feature_info)

                local_stats_str += (
                    f"{feature_name} ({color}):\n"
                    f"- Influence: {abs(feature_shap_value):.2f}\n"
                    f"- Feature Value: {feature_value}\n"
                    f"- Predominant Color: {color}\n"
                    f"- Predominant Direction: {direction} by {percentage_impact:.4f}%.\n"
                )

            increasing_features = [f for f in local_stats_raw if f['direction'] == "increases"]
            decreasing_features = [f for f in local_stats_raw if f['direction'] == "decreases"]

            return local_stats_raw, local_stats_str, increasing_features, decreasing_features

        def get_dynamic_local_analysis(self, explainer, base_values, x, predicted_y, true_y,
                                       feature_names, model_name, target_name, classes_names):

            x_data_local_instance = pd.DataFrame(x, columns=feature_names)
            x_data_local_instance = x_data_local_instance.iloc[[0]]
            explanation = explainer(x_data_local_instance)
            shap_values_local_instance = explanation.values[0] if hasattr(explanation[0], 'values') else explanation[0]
            mean_shap_value_local_instance = np.sum(np.abs(shap_values_local_instance))

            if len(shap_values_local_instance.shape) > 1:
                overall_stats_raw, overall_stats_str, overall_increasing_features, overall_decreasing_features = (
                    self.get_local_stats_info(shap_values_local_instance.mean(axis=1), feature_names,
                                              mean_shap_value_local_instance,
                                              x_data_local_instance)
                )
                overall_sorted_features = sorted(overall_stats_raw, key=lambda x: x['feature_influence'], reverse=True)
                base_value = base_values[0] if isinstance(base_values, (list, np.ndarray)) else base_values

                overall_basic_text = self.generate_model_local_basic_explanation(model_name, feature_names,
                                                                                 overall_sorted_features)
                overall_technical_text = self.generate_model_local_technical_explanation(classes_names, model_name,
                                                                                         feature_names,
                                                                                         overall_sorted_features,
                                                                                         overall_stats_raw,
                                                                                         overall_increasing_features,
                                                                                         overall_decreasing_features,
                                                                                         base_value,
                                                                                         predicted_y, true_y)

                result = {
                    "overall_results_raw": overall_stats_raw,
                    "overall_results_str": overall_stats_str,
                    "overall_technical_text": overall_technical_text,
                    "overall_basic_text": overall_basic_text
                }

                class_results = {}

                for class_index, class_name in enumerate(classes_names):
                    class_shap_values_local_instance = shap_values_local_instance[:, class_index]
                    class_mean_shap_value_local_instance = np.sum(np.abs(class_shap_values_local_instance))

                    class_local_stats_raw, class_local_stats_str, class_increasing_features, class_decreasing_features = (
                        self.get_local_stats_info(class_shap_values_local_instance, feature_names,
                                                  class_mean_shap_value_local_instance, x_data_local_instance))

                    class_sorted_features = sorted(class_local_stats_raw, key=lambda x: x['feature_influence'],
                                                   reverse=True)

                    base_value = base_values[0] if isinstance(base_values, (list, np.ndarray)) else base_values

                    class_basic_text = self.generate_model_local_basic_explanation(model_name, feature_names,
                                                                                   class_sorted_features)
                    class_technical_text = self.generate_model_local_technical_explanation(classes_names, model_name,
                                                                                           feature_names,
                                                                                           class_sorted_features,
                                                                                           class_local_stats_raw,
                                                                                           class_increasing_features,
                                                                                           class_decreasing_features,
                                                                                           base_value, predicted_y,
                                                                                           true_y)

                    class_results[class_name] = {
                        "results_raw": class_local_stats_raw,
                        "results_str": class_local_stats_str,
                        "technical_text": class_technical_text,
                        "basic_text": class_basic_text
                    }

                result["class_specific"] = class_results
            else:
                local_stats_raw, local_stats_str, increasing_features, decreasing_features = (
                    self.get_local_stats_info(shap_values_local_instance, feature_names, mean_shap_value_local_instance,
                                              x_data_local_instance))

                sorted_features = sorted(local_stats_raw, key=lambda x: x['feature_influence'], reverse=True)

                base_value = base_values[0] if isinstance(base_values, (list, np.ndarray)) else base_values

                basic_text = self.generate_model_local_basic_explanation(model_name, feature_names, sorted_features)
                technical_text = self.generate_model_local_technical_explanation(classes_names, model_name,
                                                                                 feature_names,
                                                                                 sorted_features, local_stats_raw,
                                                                                 increasing_features,
                                                                                 decreasing_features,
                                                                                 base_value, predicted_y, true_y)

                result = {
                    "results_raw": local_stats_raw,
                    "results_str": local_stats_str,
                    "technical_text": technical_text,
                    "basic_text": basic_text
                }

            return result

        async def save_to_database(self, msg, database_agent):
            msg_save_data_db = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
            msg_save_data_db.set_metadata("performative", "request")  # Set the "inform" FIPA performative
            msg_save_data_db.body = msg
            await self.send(msg_save_data_db)
            await self.receive()

        async def get_forecast_data_of_timestamp(self, timestamp, historic_normalized_test_data, historic_predictions,
                                                 target_agent, timestamp_format):
            normalized_dict = {entry['predicted_timestamp']: entry for entry in historic_normalized_test_data}
            # Check if the timestamp exists in both dictionaries
            output = {}
            if timestamp in normalized_dict:
                normalized_entry = normalized_dict[timestamp]
                # Create output dictionary
                start_tp = normalized_entry['predicted_timestamp']
                end_tp = normalized_entry['predicted_end_timestamp']
                real = next((prediction["real_value"] for prediction in historic_predictions if
                             prediction.get("predicted_timestamp") == timestamp), None)
                if real is None:
                    prediction = next((prediction for prediction in historic_predictions if
                                       prediction.get("predicted_timestamp") == timestamp), None)
                    frequency = int(prediction['frequency'])
                    target = prediction['target']
                    target_table = prediction['target_table']
                    start_dates = timestamp.split(' ')
                    end_date, end_time = self.calculate_end_date(timestamp, frequency, timestamp_format)
                    msg_ = {'start_date': start_dates[0], 'start_time': start_dates[1],
                            'end_date': end_date, 'end_time': end_time, 'frequency': frequency,
                            'target': target, 'target_table': target_table}
                    msg_ = json.dumps(msg_)
                    msg_ = 'get_real_data|' + msg_
                    request_real_data = Message(to=f"{target_agent}@{self.agent.jid.domain}/{target_agent}")
                    request_real_data.set_metadata("performative", "request")
                    request_real_data.body = json.dumps(msg_)
                    await self.send(request_real_data)
                    response = await self.receive()
                    if response:
                        if response.get_metadata("performative") == "inform":
                            real_data = response.body
                            real_data = json.loads(real_data)
                            real = real_data['value']
                output = {
                    "timestamp_start": timestamp,
                    "timestamp_end": end_tp,
                    "x_test_norm": normalized_entry['normalized_x_test'],
                    "y_test_norm": normalized_entry['normalized_y_test'],
                    "y_predicted": normalized_entry['y_predicted'],
                    "y_real": real
                }
            return output

        async def calculate_end_date(self, predicted_timestamp, frequency, timestamp_format):
            try:
                predicted_timestamp = datetime.strptime(predicted_timestamp, timestamp_format)
            except:
                predicted_timestamp = datetime.strptime(predicted_timestamp + ':00',
                                                        timestamp_format)
            end_timestamp = predicted_timestamp + timedelta(minutes=frequency)
            end_timestamp = end_timestamp.strftime(timestamp_format)
            end_timestamp_dates = end_timestamp.split(" ")
            return end_timestamp_dates[0], end_timestamp_dates[1]

    async def setup(self):
        b = self.ReceiveMsg(period=1)
        self.add_behaviour(b)
