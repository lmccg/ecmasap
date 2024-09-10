from peak import Agent, Message, PeriodicBehaviour, CyclicBehaviour
import json
from datetime import datetime, timedelta
import numpy as np
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score, \
    accuracy_score
from sqlalchemy import func

global EPSILON
EPSILON = 1e-10
class EvaluatorAgent(Agent):
    class ReceiveMsg(PeriodicBehaviour):
        async def run(self):
            msg = await self.receive()
            if msg:
                result = "Failed"
                print(f"model: {msg.sender} sent me a message: '{msg.body}'")
                with open('utils_package/config_params.json') as config_file:
                    config_params = json.load(config_file)
                with open('utils_package/config_settings.json') as config_file:
                    config_settings = json.load(config_file)
                with open('utils_package/tablesData.json') as config_file:
                    tables_dataset = json.load(config_file)
                with open('utils_package/config_agents.json') as config_file:
                    config_agents = json.load(config_file)
                metric_instant_error = config_settings['instant_error']
                metric_score = config_settings.get('error_metric')
                database_agent = config_agents["database_agent"]
                target_agent = config_agents["target_agent"]
                timestamp_format = config_settings["datetime_format"]
                request_models = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                request_models.set_metadata("performative", "request")
                request_models.body = 'get_models_to_evaluate'
                await self.send(request_models)
                response = await self.receive()
                if response:
                    if response.get_metadata("performative") == "inform":
                        models_to_evaluate = response.body
                        models_to_evaluate = json.loads(models_to_evaluate)
                        for model in models_to_evaluate:
                            if not isinstance(model, dict):
                                model = json.loads(model)
                            historic_predictions = model.get('historic_predictions_model')
                            if not isinstance(historic_predictions, (list, dict)):
                                historic_predictions = json.loads(historic_predictions)
                            model_scores = model.get('train_errors')
                            historic_scores_model = model.get('historic_scores_model')
                            if not isinstance(historic_scores_model, (list, dict)):
                                historic_scores_model = json.loads(historic_scores_model)
                            updated_historic_predictions = []
                            update_model_predictions = False
                            model_info = {'model_id ': model.get('model_id')}
                            model_predictions_realized = []
                            model_real_values = []
                            for prediction in historic_predictions:
                                updated_entry = prediction.copy()
                                if 'real_value' not in prediction:
                                    update_model_predictions = True
                                    predicted = [float(prediction['predicted_value'])]
                                    model_predictions_realized.append(predicted)
                                    predicted_timestamp = prediction['predicted_timestamp']
                                    frequency = int(prediction['frequency'])
                                    target = prediction['target']
                                    target_table = prediction['target_table']
                                    start_dates = predicted_timestamp.split(' ')
                                    request_real_data = Message(to=f"{target_agent}@{self.agent.jid.domain}/{target_agent}")
                                    request_real_data.set_metadata("performative", "request")
                                    end_date, end_time = self.calculate_end_date(predicted_timestamp, frequency, timestamp_format)
                                    msg_ = {'start_date': start_dates[0], 'start_time': start_dates[1],
                                            'end_date': end_date, 'end_time': end_time, 'frequency': frequency,
                                            'target': target, 'target_table': target_table}
                                    msg_ = json.dumps(msg_)
                                    msg_ = 'get_real_data|'+msg_
                                    request_real_data.body = json.dumps(msg_)
                                    await self.send(request_real_data)
                                    response = await self.receive()
                                    if response:
                                        if response.get_metadata("performative") == "inform":
                                            real_data = response.body
                                            real_data = json.loads(real_data)
                                            real_value = [float(real_data['value'])]
                                            model_real_values.append(real_value)
                                            instant_error = await self.calculate_instant_error(real_value, predicted, metric_instant_error)
                                            errors = await self.calculate_other_errors(instant_error,
                                                                                       real_value, predicted)

                                            updated_entry['real_value'] = real_value
                                            updated_entry['score'] = errors
                                            updated_entry['metric'] = metric_instant_error
                                            updated_entry['unit'] = real_data['unit']
                                            updated_historic_predictions.append(updated_entry)
                                else:
                                    predicted = [float(prediction['predicted_value'])]
                                    model_predictions_realized.append(predicted)
                                    real_value = [float(prediction['real_value'])]
                                    model_real_values.append(real_value)
                                    updated_historic_predictions.append(prediction)
                            test_error = model_scores
                            if len(model_predictions_realized) > 3 and len(model_real_values) > 3:
                                errors = await self.calculate_errors(model_predictions_realized, model_real_values)
                                r2 = errors[metric_score]
                                old_r2 = model_scores.get(metric_score)
                                new_r2 = 0.55*old_r2 + 0.45*r2
                                test_error = {metric_score: new_r2}
                                errors.update({'updated_at': func.now()})
                                historic_scores_model.append(errors)
                            else:
                                errors = test_error
                                errors.update({'updated_at': func.now()})
                                historic_scores_model.append(errors)

                            if update_model_predictions:
                                model_info.update({'historic_predictions_model': updated_historic_predictions})
                                model_info.update({'historic_scores_model': historic_scores_model})
                                model_info.update({"test_errors": test_error})
                                # save the data from database
                                new_msg = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                                new_msg.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                                msg_ = {'update_model_historics': model_info}
                                msg_ = json.dumps(msg_)
                                new_msg.body = msg_
                                await self.send(new_msg)
                                response = await self.receive()
                                if response:
                                    if response.get_metadata("performative") == "inform":
                                        result = response.body
                                else:
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

        async def send_reply(self, msg, result):
            response_msg = msg.make_reply()
            response_msg.set_metadata("performative", "inform")
            response_msg.body = json.dumps(result)
            await self.send(response_msg)

        async def calculate_end_date(self, predicted_timestamp, frequency, timestamp_format):
            try:
                predicted_timestamp = datetime.strptime(predicted_timestamp,timestamp_format)
            except:
                predicted_timestamp = datetime.strptime(predicted_timestamp + ':00',
                                                                 timestamp_format)
            end_timestamp = predicted_timestamp + timedelta(minutes=frequency)
            end_timestamp = end_timestamp.strftime(timestamp_format)
            end_timestamp_dates = end_timestamp.split(" ")
            return end_timestamp_dates[0], end_timestamp_dates[1]

        async def calculate_errors(self, actual, predicted):
            if len(predicted) != len(actual):
                return "Failed"

            try:
                result = await self._calculateError(predicted, actual)
            except:
                try:
                    result = await self._calculateErrorStr(predicted, actual)
                except:
                    return "Failed"
            return result

        async def calculate_other_errors(self, errors, actual, predicted):
            if len(predicted) != len(actual):
                return "Failed"

            try:
                errors = await self._calculateOtherErrors(errors, predicted, actual)
            except:
                pass
            return errors

        async def calculate_instant_error(self, actual, predicted, metric):

            if len(predicted) != len(actual):
                return "Failed"
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
                        v1 = predicted[0]
                        calculate_Normal = False
            if calculate_Normal:
                try:
                    if metric == 'me':
                        result = {'me': await self.me(actual=actual, predicted=predicted)}
                    elif metric == 'mse':
                        result = {'mse': await self.mse(actual=actual, predicted=predicted)}
                    elif metric == 'mase':
                        result = {'mase': await self.mase(actual=actual, predicted=predicted)}
                    elif metric == 'rmse':
                        result = {'rmse': await self.rmse(actual=actual, predicted=predicted)}
                    elif metric == 'mae':
                        result = {'mae': await self.mae(actual=actual, predicted=predicted)}
                    elif metric == 'mape':
                        result = {'mape': await self.mape(actual=actual, predicted=predicted)}
                    elif metric == 'mape_2':
                        result = {'mape_2': await self.mape_2(actual=actual, predicted=predicted)}
                    elif metric == 'smape':
                        result = {'smape': await self.smape(actual=actual, predicted=predicted)}
                    elif metric == 'rmspe':
                        result = {'rmspe': await self.rmspe(actual=actual, predicted=predicted)}
                    elif metric == 'rmsse':
                        result = {'rmsse': await self.rmsse(actual=actual, predicted=predicted)}
                    elif metric == 'r2':
                        result = {'r2': await self.r2(actual=actual, predicted=predicted)}
                    elif metric == 'ae':
                        result = {'ae': await self._absolute_error(actual=actual, predicted=predicted)}
                    elif metric == 'wape':
                        result = {'wape': await self.wape(actual=actual, predicted=predicted)}
                    else:
                        result = {}
                except:
                   return "Failed"
            else:
                try:
                    result = await self._calculateErrorStr(predicted, actual)
                except Exception as ex:
                    return "Failed"
            return result

        async def _calculateError(self, predict, actual):

            if len(predict) > 1:
                metrics = {
                    'me': await self.me(actual=actual, predicted=predict),
                    'mse': await self.mse(actual=actual, predicted=predict),
                    'mase': await self.mase(actual=actual, predicted=predict),
                    'rmse': await self.rmse(actual=actual, predicted=predict),
                    'mae': await self.mae(actual=actual, predicted=predict),
                    'mape': await self.mape(actual=actual, predicted=predict),
                    'mape_2': await self.mape_2(actual=actual, predicted=predict),
                    'smape': await self.smape(actual=actual, predicted=predict),
                    'rmspe': await self.rmspe(actual=actual, predicted=predict),
                    'rmsse': await self.rmsse(actual=actual, predicted=predict),
                    'r2': await self.r2(actual=actual, predicted=predict),
                    'wape': await self.wape(actual=actual, predicted=predict)
                }
            else:
                metrics = {
                    'me': await self.me(actual=actual, predicted=predict),
                    'mse': await self.mse(actual=actual, predicted=predict),
                    'mase': np.NaN,
                    'rmse': await self.rmse(actual=actual, predicted=predict),
                    'mae': await self.mae(actual=actual, predicted=predict),
                    'mape': await self.mape(actual=actual, predicted=predict),
                    'mape_2': await self.mape_2(actual=actual, predicted=predict),
                    'smape': await self.smape(actual=actual, predicted=predict),
                    'rmspe': await self.rmspe(actual=actual, predicted=predict),
                    'rmsse': np.NaN,
                    'r2': np.NaN,
                    'wape': await self.wape(actual=actual, predicted=predict)
                }

            for k, v in metrics.items():
                if np.isnan(v):
                    metrics[k] = np.float64(0.0)
                elif np.isinf(v):
                    metrics[k] = np.float64(0.0)
                if not isinstance(v, np.float64):
                    metrics[k] = np.float64(v)
            return metrics
        async def _calculateOtherErrors(self, errors, predict, actual):

            errors.update({'rmspe': await self.rmspe(actual=actual, predicted=predict)})
            errors.update({'smape': await self.smape(actual=actual, predicted=predict)})
            errors.update({'wape': await self.wape(actual=actual, predicted=predict)})
            for k, v in errors.items():
                if np.isnan(v):
                    errors[k] = np.float64(0.0)
                elif np.isinf(v):
                    errors[k] = np.float64(0.0)
                if not isinstance(v, np.float64):
                    errors[k] = np.float64(v)
            return errors

        async def _calculateErrorStr(self, predict, actual):
            metrics = {
                'r2': await self.accuracy_score_(actual=actual, predicted=predict)
            }
            return metrics


        async def _error(self, actual: np.ndarray, predicted: np.ndarray):
            """ Simple error """
            return actual - predicted

        async def _absolute_error(self, actual: np.ndarray, predicted: np.ndarray):
            """ Absolute error """
            actual = np.ravel(actual)
            predicted = np.ravel(predicted)
            possible_bool = ['true', 'false']
            if str(predicted[0]).lower() in possible_bool:
                await self.accuracy_score_(actual=actual, predicted=predicted)
            else:
                return np.abs(np.mean(await self._error(actual, predicted)))
            # return np.abs(actual - predicted)

        async def _percentage_error_2(self, actual: np.ndarray, predicted: np.ndarray):
            """
            Percentage error
            Note: result is NOT multiplied by 100
            """
            predicted_aux = []
            for i in range(0, len(predicted)):
                if predicted[i] < 0:
                    predicted_aux.append(0)
                else:
                    predicted_aux.append(predicted[i])

            errors = []
            for i in range(0, len(actual)):
                e = 0

                a = actual[i]
                p = predicted_aux[i]

                if a == 0 and p > 0:
                    e = 1
                elif a == 0 and p == 0:
                    e = 0
                else:
                    e = np.abs(a - p) / a

                if e > 1:
                    e = 1

                errors.append(e)

            return errors

        async def _naive_Errorsing(self, actual: np.ndarray, seasonality: int = 1):
            """ Naive Errorsing method which just repeats previous samples """
            return actual[:-seasonality]

        async def mse(self, actual: np.ndarray, predicted: np.ndarray):
            """ Mean Squared Error """
            actual = np.ravel(actual)
            predicted = np.ravel(predicted)
            return mean_squared_error(actual, predicted, squared=True)

        async def rmse(self, actual: np.ndarray, predicted: np.ndarray):
            """ Root Mean Squared Error """
            actual = np.ravel(actual)
            predicted = np.ravel(predicted)

            return mean_squared_error(actual, predicted, squared=False)

        async def me(self, actual: np.ndarray, predicted: np.ndarray):
            """ Mean Error """

            actual = np.ravel(actual)
            predicted = np.ravel(predicted)

            return np.mean(await self._error(actual, predicted))

        async def mae(self, actual: np.ndarray, predicted: np.ndarray):
            """ Mean Absolute Error """

            actual = np.ravel(actual)
            predicted = np.ravel(predicted)

            return mean_absolute_error(actual, predicted)

        async def mape(self, actual: np.ndarray, predicted: np.ndarray):
            """
            Mean Absolute Percentage Error
            Properties:
                    + Easy to interpret
                    + Scale independent
                    - Biased, not symmetric
                    - Undefined when actual[t] == 0
            Note: result is NOT multiplied by 100
            """

            actual = np.ravel(actual)
            predicted = np.ravel(predicted)

            return mean_absolute_percentage_error(actual, predicted)

        async def mape_2(self, actual: np.ndarray, predicted: np.ndarray):
            """
            Mean Absolute Percentage Error
            Properties:
                    + Easy to interpret
                    + Scale independent
                    - Biased, not symmetric
                    - Undefined when actual[t] == 0
            Note: result is multiplied by 100
            """

            actual = np.ravel(actual)
            predicted = np.ravel(predicted)

            return np.mean(np.abs(await self._percentage_error_2(actual, predicted)))

        async def smape(self, actual: np.ndarray, predicted: np.ndarray):
            """mape = skmetrics.mean_absolute_percentage_error(y_true, y_pred)
            Symmetric Mean Absolute Percentage Error
            Note: result is multiplied by 100
            """
            actual = np.ravel(actual)
            predicted = np.ravel(predicted)
            # Supress/hide the warning
            np.seterr(invalid='ignore')
            smape = 2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))
            smape = np.nan_to_num(smape)
            return np.mean(smape) * 100

        async def wape(self, actual: np.ndarray, predicted: np.ndarray):
            """
            Weight Absolute Percentage error
            Note: result is multiplied by 100
            """

            sum_errors = 0
            sum_actual = 0
            for i in range(0, len(actual)):
                a = actual[i]
                p = predicted[i]

                sum_errors += np.abs(a - p)
                sum_actual += a

            return (sum_errors / sum_actual) * 100

        async def mase(self, actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
            """
            Mean Absolute Scaled Error
            Baseline (benchmark) is computed with naive Errorsing (shifted by @seasonality)
            """

            actual = np.ravel(actual)
            predicted = np.ravel(predicted)

            naive_mae = await self.mae(actual[seasonality:], await self._naive_Errorsing(actual, seasonality))
            if naive_mae == 0:
                return 0
            return await self.mae(actual, predicted) / await self.mae(actual[seasonality:],
                                                            await self._naive_Errorsing(actual, seasonality))

            actual = np.ravel(actual)
            predicted = np.ravel(predicted)

            mase = mean_squared_error(actual, predicted)

            return np.sqrt(np.mean(np.sqrt(mse)))

        async def rmspe(self, actual: np.ndarray, predicted: np.ndarray):
            """
            Root Mean Squared Percentage Error
            Note: result is multiplied by 100
            """

            actual = np.ravel(actual)
            predicted = np.ravel(predicted)

            mse = mean_squared_error(actual, predicted)

            return np.sqrt(np.mean(np.sqrt(mse)))

        async def rmsse(self, actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
            """ Root Mean Squared Scaled Error """

            actual = np.ravel(actual)
            predicted = np.ravel(predicted)

            q = np.abs(await self._error(actual, predicted)) / await self.mae(actual[seasonality:],
                                                                    await self._naive_Errorsing(actual, seasonality))
            return np.sqrt(np.mean(np.square(q)))

        async def r2(self, actual: np.ndarray, predicted: np.ndarray):
            actual = np.ravel(actual)
            predicted = np.ravel(predicted)

            return r2_score(actual, predicted)

        async def accuracy_score_(self, actual, predicted):
            return accuracy_score(actual, predicted)

    async def setup(self):
        self.add_behaviour(self.ReceiveMsg(period=1))
