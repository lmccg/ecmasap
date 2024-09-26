from peak import Agent, CyclicBehaviour, PeriodicBehaviour, Message, Template
import asyncio
import json
from datetime import datetime, timedelta
import datetime as dt
import base64
import zlib
import uuid
import os
import pickle
from utils_package.utils import timestamp_with_time_zone
training = False
class TargetAgent(Agent):
    class ReceiveMsg(CyclicBehaviour):
        async def run(self):
            msg = await self.receive()
            try:
                if msg:
                    msg_id = str(uuid.uuid4())
                    print(f"{timestamp_with_time_zone()} TargetAgent: {msg.sender} sent me a message: '{msg.body}'")
                    parts_of_msg = msg.body.split("|")
                    request_type = parts_of_msg[0]
                    print(timestamp_with_time_zone(), 'request_type:', request_type)
                    if request_type == 'predict' or request_type == 'train' or request_type == 'retrain':
                        target = parts_of_msg[1]
                        request_data = parts_of_msg[2]
                        request_data = json.loads(request_data)
                        process = True
                        if request_type == 'predict' and (
                                'start_date' not in request_data or 'start_time' not in request_data):
                            process = False
                        elif (request_type == 'train' or request_type == 'retrain') and (
                                'end_date' not in request_data or 'end_time' not in request_data):
                            process = False
                        if process:
                            with open('utils_package/config_agents.json') as config_file:
                                config_agents = json.load(config_file)
                            with open('utils_package/config_settings.json') as config_file:
                                config_settings = json.load(config_file)
                            with open('utils_package/tablesData.json') as config_file:
                                tables_dataset = json.load(config_file)
                            datetime_format = config_settings.get('datetime_format')
                            if request_type == 'predict':
                                weeks_data = config_settings.get("weeks_historical_cases_test")
                                first_start_timestamp, last_end_timestamp, request_data = await self.fix_timestamps_test(
                                    request_data,
                                    datetime_format)
                            elif request_type == 'train' or request_type == 'retrain':
                                weeks_data = config_settings.get("weeks_historical_cases_train")
                                _, _, request_data = await self.fix_timestamps_train(request_data, datetime_format,
                                                                                     weeks_data)
                            print(timestamp_with_time_zone(), 'timestamp fixed')
                            dataset_types = config_settings.get('possible_data_types')
                            dataset_types = list(dataset_types.keys())
                            if "dataset_type" in request_data:
                                dataset_type = request_data["dataset_type"]
                                if dataset_type not in dataset_types:
                                    dataset_type, request_data = await self.checkDatasetType(request_data,
                                                                                             datetime_format)
                                else:
                                    request_data = await self.fix_transformations_dataset(dataset_type, request_data)
                            else:
                                dataset_type, request_data = await self.checkDatasetType(request_data, datetime_format)
                            table_target = request_data['target_table']
                            model_type = tables_dataset.get("time_series_columns").get(table_target).get(target)
                            agents = config_agents['ml_model_agents'][model_type]
                            # get the historical data
                            print(timestamp_with_time_zone(), 'get historical data')
                            historical_data_agent = config_agents["historical_data_agent"]
                            msg_hst_data = Message(
                                to=f"{historical_data_agent}@{self.agent.jid.domain}/{historical_data_agent}")
                            msg_hst_data.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                            msg_hst_data.set_metadata("thread", msg_id)
                            if isinstance(request_data, dict):
                                request_hst_data = json.dumps(request_data)
                            else:
                                request_hst_data = request_data
                            request_hst_data = 'get_historical_data|' + request_hst_data
                            msg_hst_data.body = request_hst_data
                            await self.send(msg_hst_data)
                            # self.agent.add_behaviour(self.agent.WaitResponse())
                            response_hist_data = None
                            while not response_hist_data:
                                if self.agent.received_messages:
                                    for hst_msg in self.agent.received_messages:
                                        if hst_msg.get_metadata('thread') == msg_id and hst_msg.get_metadata("performative") == "inform" and 'historical_data_response' in hst_msg.body:
                                            response_hist_data = hst_msg
                                            self.agent.received_messages.remove(hst_msg)
                                            break
                                if response_hist_data:
                                    break
                                await asyncio.sleep(1)

                            # # Check if the result from InnerBehaviour has been set
                            # while not self.agent.received_messages:
                            #     # print("Waiting for response from hist data agent...")
                            #     await asyncio.sleep(1)  # Wait for the WaitResponse behavior to process the message
                            #
                            # if self.agent.response_hist_data:
                            #     response_hist_data = self.agent.response_hist_data
                            #     # print(f"ReceiveMsg received result from WaitHistoricalData: {response_hist_data}")
                            #     # Reset shared_data after processing
                            #     self.agent.response_hist_data = None
                            #

                            if response_hist_data:
                                encoded_data = response_hist_data.body
                                encoded_data = encoded_data.split("|")
                                encoded_data = encoded_data[1]
                                input_data = self.load_object_from_file(encoded_data)
                                # Decode the Base64-encoded string
                                # compressed_data = base64.b64decode(encoded_data)
                                #
                                # # Decompress the data
                                # input_data = zlib.decompress(compressed_data).decode('utf-8')

                                try:
                                    input_data = json.loads(input_data)
                                except:
                                    pass
                                input_dataset = input_data[0]
                                # ask predictions
                                if request_type == 'predict':
                                    # get the models to predict
                                    agent_get_ml_models = config_agents["rl_selector_agent"]
                                    predict_msg = Message(
                                        to=f"{agent_get_ml_models}@{self.agent.jid.domain}/{agent_get_ml_models}")
                                    predict_msg.set_metadata("performative",
                                                             "request")  # Set the "inform" FIPA performative
                                    if not isinstance(request_data, dict):
                                        request_data = json.loads(request_data)

                                    print(timestamp_with_time_zone(), 'line 69')
                                    request_data = await self.get_input_characterization_predict(request_data,
                                                                                                 tables_dataset,
                                                                                                 dataset_type,
                                                                                                 first_start_timestamp,
                                                                                                 last_end_timestamp)
                                    print(timestamp_with_time_zone(), request_data)
                                    if isinstance(request_data, dict):
                                        request_data = json.dumps(request_data)
                                    predict_msg.body = request_data
                                    await self.send(predict_msg)
                                    response = await self.receive()
                                    models = []
                                    if response:
                                        if response.get_metadata("performative") == "inform":
                                            models = response.body
                                            models = json.loads(models)

                                    # Wait for a response
                                    print(timestamp_with_time_zone(), 'lets send some requests')
                                    tasks = []
                                    predict_one_model = False
                                    while models:
                                        entry = models.pop(0)
                                        ml_model = entry[1]
                                        type_model = entry[2]
                                        agents = config_agents['ml_model_agents'][type_model]
                                        agent = next(
                                            (key for key, value in agents.items() if value.lower() == ml_model.lower()),
                                            None)
                                        model = entry[0]
                                        #model é model id e model id será o meu boolean
                                        await self._send_and_collect_response(request_type, agent, model, request_data,
                                                                              input_dataset, msg_id, False)
                                        # result =
                                        # if result != 'Failed':
                                        #     predict_one_model = True
                                        #     response_msg = msg.make_reply()
                                        #     response_msg.set_metadata("performative", "inform")
                                        #     response_msg.body = json.dumps(result)
                                        #     await self.send(response_msg)
                                        #     break
                                    # if not predict_one_model:
                                    #     response_msg = msg.make_reply()
                                    #     response_msg.set_metadata("performative", "inform")
                                    #     response_msg.body = "It was not possible to send a response for request received."
                                    #     await self.send(response_msg)

                                    if models:
                                        for entry in models:
                                            ml_model = entry[1]
                                            type_model = entry[2]
                                            agents = config_agents['ml_model_agents'][type_model]
                                            agent = next(
                                                (key for key, value in agents.items() if
                                                 value.lower() == ml_model.lower()),
                                                None)
                                            model = entry[0]
                                            #model é model id
                                            await self._send_and_collect_response(request_type, agent, model,
                                                                                  request_data, input_dataset, msg_id,
                                                                                  True)
                                        #     task =
                                        #     tasks.append(task)
                                        # await asyncio.gather(*tasks)

                                        response_msg = msg.make_reply()
                                        response_msg.set_metadata("performative", "inform")
                                        response_msg.body = "Request executed!"
                                        await self.send(response_msg)
                                elif request_type == 'train':
                                    date_column = input_data[1]
                                    datetimeFormat = input_data[2]
                                    columns_in_df_from_hc = input_data[3]
                                    categorical_columns = input_data[4]
                                    response = msg.make_reply()
                                    response.set_metadata("performative", "inform")
                                    response.body = "Requested train"
                                    await self.send(response)
                                    await self.set_training_on()
                                    training_dates = await self.get_training_dates(request_data, config_settings)
                                    request_data = await self.get_input_characterization_train(request_data,
                                                                                               tables_dataset,
                                                                                               training_dates,
                                                                                               date_column,
                                                                                               datetimeFormat,
                                                                                               categorical_columns,
                                                                                               columns_in_df_from_hc)
                                    asyncio.create_task(
                                        self.run_train_in_background(request_data, request_type, input_dataset,
                                                                     dataset_types, agents, msg_id))
                                elif request_type == 'retrain':
                                    date_column = input_data[1]
                                    datetimeFormat = input_data[2]
                                    columns_in_df_from_hc = input_data[3]
                                    categorical_columns = input_data[4]
                                    response = msg.make_reply()
                                    response.set_metadata("performative", "inform")
                                    response.body = "Requested retrain"
                                    await self.send(response)
                                    model_id = request_data.get('model_id')
                                    await self.set_training_on()
                                    training_dates = await self.get_training_dates(request_data, config_settings)
                                    request_data = await self.get_input_characterization_train(request_data,
                                                                                               tables_dataset,
                                                                                               training_dates,
                                                                                               date_column,
                                                                                               datetimeFormat,
                                                                                               categorical_columns,
                                                                                               columns_in_df_from_hc)
                                    asyncio.create_task(
                                        self.run_retrain_in_background(request_data, request_type, input_dataset,
                                                                       model_id, agents, msg_id))
                            else:
                                response_msg = msg.make_reply()
                                response_msg.set_metadata("performative", "inform")
                                response_msg.body = "Please fix the timestamps provided! For train and retrain is mandatory to have 'end_date' and 'end_time' fields. For predict is mandatory to have 'start_date' and 'start_time' fields"
                                await self.send(response_msg)

                    elif request_type == 'get_real_data':
                        with open('utils_package/config_agents.json') as config_file:
                            config_agents = json.load(config_file)

                        # get the real data
                        historical_data_agent = config_agents["historical_data_agent"]
                        msg_real_data = Message(
                            to=f"{historical_data_agent}@{self.agent.jid.domain}/{historical_data_agent}")
                        msg_real_data.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                        msg_requested = json.dumps({request_type: parts_of_msg[2]})
                        msg_real_data.body = msg_requested
                        await self.send(msg_real_data)
                        response = await self.receive()
                        if response:
                            if response.get_metadata("performative") == "inform":
                                input_data = response.body
                                response_msg = msg.make_reply()
                                response_msg.set_metadata("performative", "inform")
                                response_msg.body = input_data
                                await self.send(response_msg)
                    elif request_type == 'get_training_status':
                        status = json.dumps(await self.set_get_training_status())
                        response_msg = msg.make_reply()
                        response_msg.set_metadata("performative", "inform")
                        response_msg.body = status
                        await self.send(response_msg)

                    elif request_type == 'needed_models_to_train':
                        with open('utils_package/config_agents.json') as config_file:
                            config_agents = json.load(config_file)
                        with open('utils_package/config_settings.json') as config_file:
                            config_settings = json.load(config_file)
                        with open('utils_package/config_tables_db.json') as f:
                            config_tables_db = json.load(f)
                        frequencies = config_settings['frequencies_list']
                        dataset_types = config_settings.get('possible_data_types')
                        dataset_types = list(dataset_types.keys())
                        tables = config_tables_db.get('tables')
                        formated_timestamp = config_settings.get('datetime_format')
                        # get the model for that target
                        database_agent = config_agents["database_agent"]
                        msg_model_target = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                        msg_model_target.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                        data = []
                        for entry in tables:
                            for table, columns in entry.items():
                                for column in columns:
                                    for frequency in frequencies:
                                        for dataset_type in dataset_types:
                                            msg_ = {'get_model_for': {
                                                'target': column,
                                                'target_table': table,
                                                'dataset_type': dataset_type,
                                                'frequency': frequency
                                            }}
                                            msg_requested = json.dumps(msg_)
                                            msg_model_target.body = msg_requested
                                            await self.send(msg_model_target)
                                            self.agent.add_behaviour(self.agent.WaitTrainResponse(period=1))
                                            response = None
                                            # We wait for the behaviour to update `response_get`
                                            while not self.agent.response_get:
                                                await asyncio.sleep(
                                                    1)  # Wait for the WaitResponse behavior to process the message

                                            # Once we have the response
                                            if self.agent.response_get:
                                                response = self.agent.response_get
                                                # Reset shared_data after processing
                                                self.agent.response_get = None
                                            if response:
                                                if response.get_metadata("performative") == "inform":
                                                    exists = response.body
                                                    exists = json.loads(exists)
                                                    if not exists:
                                                        now = datetime.now()
                                                        formatted_string = now.strftime(formated_timestamp)
                                                        list_formatted_string = formatted_string.split(' ')
                                                        train_data = msg_.get('get_model_for')
                                                        train_data.update({'end_date': list_formatted_string[0]})
                                                        train_data.update({'end_time': list_formatted_string[1]})
                                                        data.append(train_data)
                        response_msg = msg.make_reply()
                        response_msg.set_metadata("performative", "inform")
                        data = json.dumps(data)
                        response_msg.body = data
                        await self.send(response_msg)

                    else:
                        with open('utils_package/config_agents.json') as config_file:
                            config_agents = json.load(config_file)
                        main_agent = config_agents["main_agent"]
                        response_msg = Message(to=f"{main_agent}@{self.agent.jid.domain}/{main_agent}")
                        response_msg.set_metadata("performative", "inform")
                        response_msg.body = msg.body
                        await self.send(response_msg)


            except Exception as e:
                print(timestamp_with_time_zone(), 'exception', e, 'msg', msg.body)
                try:
                    print(timestamp_with_time_zone(), msg.body, 'response', response_hist_data)
                except:
                    pass
                response_msg = msg.make_reply()
                response_msg.set_metadata("performative", "inform")
                response_msg.body = "Failed!"
                await self.send(response_msg)

        async def run_train_in_background(self, request_data, request_type, input_data, dataset_types, agents, msg_id):
            tasks = []
            # for dataset_type in dataset_types:
            for agent, model in agents.items():
                if not isinstance(request_data, dict):
                    request_data = json.loads(request_data)
                # request_data['dataset_type'] = dataset_type
                if isinstance(request_data, dict):
                    request_data = json.dumps(request_data)
                if isinstance(input_data, list):
                    input_data = json.dumps(input_data)
                await self._send_and_collect_response(request_type, agent, model, request_data, input_data, msg_id)
            await self.set_training_off()
            #     task =
            #     tasks.append(task)
            # await asyncio.gather(*tasks)

        async def set_get_training_status(self):
            global training
            return training

        async def set_training_on(self):
            global training
            training = True

        async def set_training_off(self):
            global training
            training = False


        async def get_training_dates(self, request_data, config_settings):
            start_date = request_data.get('start_date')
            end_date = request_data.get('end_date')
            start_time = request_data.get('start_time')
            end_time = request_data.get('end_time')
            datetime_format = config_settings.get('datetime_format')
            training_weeks = config_settings.get('training_weeks')
            if start_date is None:
                str_historical_start_train_dataset = datetime.strptime(end_date + " " + end_time, datetime_format)
                str_historical_start_train_dataset = str_historical_start_train_dataset - dt.timedelta(
                    weeks=training_weeks)
                str_historical_start_train_dataset = str_historical_start_train_dataset.strftime(datetime_format)
                str_historical_start_train_dataset = str_historical_start_train_dataset.split(" ")
                start_date = str_historical_start_train_dataset[0]
                start_time = str_historical_start_train_dataset[1]
            training_dates = {'start_date': start_date, 'end_date': end_date, 'start_time': start_time,
                              'end_time': end_time}
            return training_dates

        async def run_retrain_in_background(self, request_data, request_type, input_data, model_id, agents, msg_id):
            if not isinstance(request_data, dict):
                request_data = json.loads(request_data)
            model = request_data['model']
            agent = next(
                (key for key, value in agents.items() if value.lower() == model.lower()),
                None)
            if isinstance(request_data, dict):
                request_data = json.dumps(request_data)
            if isinstance(input_data, list):
                input_data = json.dumps(input_data)
            await self._send_and_collect_response(request_type, agent, model, request_data, input_data, msg_id,
                                                  model_id)
            await self.set_training_off()
        async def round_to_frequency(self, dt, frequency_minutes, round_up=False):
            # Convert the time to total minutes (hour * 60 + minute)
            minutes = dt.hour * 60 + dt.minute
            # Round the minutes to the nearest multiple of the frequency
            if round_up:
                rounded_minutes = ((minutes + frequency_minutes - 1) // frequency_minutes) * frequency_minutes
            else:
                rounded_minutes = (minutes // frequency_minutes) * frequency_minutes
            # Create a new datetime with the rounded minutes and zeroed out seconds and microseconds
            return dt.replace(hour=rounded_minutes // 60, minute=rounded_minutes % 60, second=0, microsecond=0)

        async def fix_timestamps_train(self, input_data, datetime_format, weeks_data):
            try:
                start_date = input_data.get('start_date', None)
                start_time = input_data.get('start_time', None)
                end_date = input_data.get('end_date', None)
                end_time = input_data.get('end_time', None)
                frequency_minutes = int(input_data['frequency'])
                # Parse the input strings into datetime objects
                if start_date is not None:
                    try:
                        start_datetime = datetime.strptime(f"{start_date} {start_time}", datetime_format)
                    except:
                        start_datetime = datetime.strptime(f"{start_date} {start_time}:00", datetime_format)
                    if end_date and end_time:
                        try:
                            end_datetime = datetime.strptime(f"{end_date} {end_time}", datetime_format)
                        except:
                            end_datetime = datetime.strptime(f"{end_date} {end_time}:00", datetime_format)
                    else:
                        end_datetime = start_datetime
                else:
                    try:
                        end_datetime = datetime.strptime(f"{end_date} {end_time}", datetime_format)
                    except:
                        end_datetime = datetime.strptime(f"{end_date} {end_time}:00", datetime_format)
                    start_datetime = end_datetime
                # Round the start time down and the end time up to the nearest frequency
                # Round the start time down to the nearest frequency
                adjusted_start = await self.round_to_frequency(start_datetime, frequency_minutes, round_up=False)

                # Determine if the dates are the same and times are exactly equal
                if start_date == end_date and start_datetime == end_datetime:
                    # If start and end are exactly the same, make them identical after rounding
                    adjusted_end = adjusted_start
                else:
                    # Otherwise, round the end time up to the nearest frequency
                    adjusted_end = await self.round_to_frequency(end_datetime, frequency_minutes, round_up=True)

                    # Ensure the adjusted end is after the adjusted start
                    if adjusted_end <= adjusted_start:
                        adjusted_end = adjusted_start + timedelta(minutes=frequency_minutes)
                diff_weeks = abs((adjusted_end - adjusted_start).days) / 7
                if diff_weeks != weeks_data:
                    adjusted_start = adjusted_start - dt.timedelta(weeks=weeks_data)
                    adjusted_end = adjusted_end - dt.timedelta(minutes=frequency_minutes)
                start_datetime = adjusted_start.strftime(datetime_format)
                end_datetime = adjusted_end.strftime(datetime_format)
                start_date = start_datetime.split(" ")[0]
                start_time = start_datetime.split(" ")[1]
                end_date = end_datetime.split(" ")[0]
                end_time = end_datetime.split(" ")[1]
                input_data['start_date'] = start_date
                input_data['start_time'] = start_time
                input_data['end_date'] = end_date
                input_data['end_time'] = end_time
                return start_datetime, end_datetime, input_data
            except Exception as e:
                print(e)
                return e

        async def fix_timestamps_test(self, input_data, datetime_format):
            try:
                start_date = input_data.get('start_date', None)
                start_time = input_data.get('start_time', None)
                end_date = input_data.get('end_date', None)
                end_time = input_data.get('end_time', None)
                frequency_minutes = int(input_data['frequency'])
                # Parse the input strings into datetime objects
                if start_date is not None:
                    try:
                        start_datetime = datetime.strptime(f"{start_date} {start_time}", datetime_format)
                    except:
                        start_datetime = datetime.strptime(f"{start_date} {start_time}:00", datetime_format)
                    if end_date and end_time:
                        try:
                            end_datetime = datetime.strptime(f"{end_date} {end_time}", datetime_format)
                        except:
                            end_datetime = datetime.strptime(f"{end_date} {end_time}:00", datetime_format)
                    else:
                        end_datetime = start_datetime
                else:
                    try:
                        end_datetime = datetime.strptime(f"{end_date} {end_time}", datetime_format)
                    except:
                        end_datetime = datetime.strptime(f"{end_date} {end_time}:00", datetime_format)
                    start_datetime = end_datetime
                # Round the start time down and the end time up to the nearest frequency
                # Round the start time down to the nearest frequency
                adjusted_start = await self.round_to_frequency(start_datetime, frequency_minutes, round_up=False)

                # Determine if the dates are the same and times are exactly equal
                if start_date == end_date and start_datetime == end_datetime:
                    # If start and end are exactly the same, make them identical after rounding
                    adjusted_end = adjusted_start
                else:
                    # Otherwise, round the end time up to the nearest frequency
                    adjusted_end = await self.round_to_frequency(end_datetime, frequency_minutes, round_up=True)

                    # Ensure the adjusted end is after the adjusted start
                    if adjusted_end <= adjusted_start:
                        adjusted_end = adjusted_start + timedelta(minutes=frequency_minutes)
                start_datetime = adjusted_start.strftime(datetime_format)
                end_datetime = adjusted_end.strftime(datetime_format)
                start_date = start_datetime.split(" ")[0]
                start_time = start_datetime.split(" ")[1]
                end_date = end_datetime.split(" ")[0]
                end_time = end_datetime.split(" ")[1]
                input_data['start_date'] = start_date
                input_data['start_time'] = start_time
                input_data['end_date'] = end_date
                input_data['end_time'] = end_time
                return start_datetime, end_datetime, input_data
            except Exception as e:
                print(e)
                return e

        async def fix_transformations_dataset(self, dataset_type, input_data):
            dayofweek = input_data.get('dayofweek', [])
            exclude_weekends = input_data.get('exclude_weekends', False)
            weekends = input_data.get('weekends', False)
            dayofweek_final = []
            # Determine the overall category based on counts
            if dataset_type == "working_periods":
                for day in dayofweek:
                    if day < 5:
                        dayofweek_final.append(day)
                exclude_weekends = True
                weekends = False
            elif dataset_type == "nights_and_weekends_periods":
                for day in dayofweek:
                    if day >= 5:
                        dayofweek_final.append(day)
                exclude_weekends = False
                weekends = True
            elif dataset_type == "general_periods":
                dayofweek_final = dayofweek
                if weekends and exclude_weekends:
                    exclude_weekends = False
                elif not weekends and not exclude_weekends:
                    exclude_weekends = True
            input_data['dayofweek'] = dayofweek_final
            input_data['exclude_weekends'] = exclude_weekends
            input_data['weekends'] = weekends
            input_data['dataset_type'] = dataset_type
            return input_data

        async def checkDatasetType(self, input_data,
                                   datetimeFormat):
            start_date = input_data['start_date']
            start_time = input_data['start_time']
            end_date = input_data['end_date']
            end_time = input_data['end_time']
            dayofweek = input_data.get('dayofweek', [])
            exclude_weekends = input_data.get('exclude_weekends', False)
            weekends = input_data.get('weekends', False)
            frequency = int(input_data['frequency'])
            dayofweek_final = []
            try:
                timestamp_start_date = datetime.strptime(start_date + ' ' + start_time, datetimeFormat)
            except:
                timestamp_start_date = datetime.strptime(start_date + ' ' + start_time + ':00', datetimeFormat)
            try:
                timestamp_end_date = datetime.strptime(end_date + ' ' + end_time, datetimeFormat)
            except:
                timestamp_end_date = datetime.strptime(end_date + ' ' + end_time + ':00', datetimeFormat)
            date_range_data = [timestamp_start_date + i * timedelta(minutes=frequency) for i in range(int(((
                                                                                                                   timestamp_end_date - timestamp_start_date).total_seconds() / 60) / frequency) + 1)]
            date_range_data_left = []
            for date in date_range_data:
                date_week_day = date.weekday()
                if len(dayofweek) > 0:
                    for d in dayofweek:
                        if int(date_week_day) == int(d):
                            date_range_data_left.append(date)
                else:
                    date_range_data_left.append(date)
            date_range_data_final = []
            for date in date_range_data_left:
                date_week_day = date.weekday()
                if exclude_weekends and int(date_week_day) < 5:
                    date_range_data_final.append(date)
                if not exclude_weekends:
                    date_range_data_final.append(date)
            working_hours = range(8, 20)  # Working hours from 08:00 to 19:59 Monday to Friday
            nights_and_weekends = set(range(20, 24)) | set(range(0, 8))  # Nights and weekends

            # Initialize counters
            working_count = 0
            nights_and_weekends_count = 0
            other_count = 0

            for dt in date_range_data_final:
                # Check if the datetime falls within working hours
                if dt.weekday() < 5 and dt.hour in working_hours:
                    working_count += 1
                    other_count += 1
                # Check if the datetime falls within nights and weekends
                elif dt.weekday() >= 5 or dt.hour in nights_and_weekends:
                    nights_and_weekends_count += 1
                    other_count += 1
                else:
                    other_count += 1
            dataset_type = "general_periods"
            # Determine the overall category based on counts
            if working_count == len(date_range_data_final):
                dataset_type = "working_periods"
                for day in dayofweek:
                    if day < 5:
                        dayofweek_final.append(day)
                exclude_weekends = True
                weekends = False
            elif nights_and_weekends_count == len(date_range_data_final):
                dataset_type = "nights_and_weekends_periods"
                for day in dayofweek:
                    if day >= 5:
                        dayofweek_final.append(day)
                exclude_weekends = False
                weekends = True
            elif other_count == len(date_range_data_final):
                dataset_type = "general_periods"
                dayofweek_final = dayofweek
                if weekends and exclude_weekends:
                    exclude_weekends = False
                elif not weekends and not exclude_weekends:
                    exclude_weekends = True
            input_data['dayofweek'] = dayofweek_final
            input_data['exclude_weekends'] = exclude_weekends
            input_data['weekends'] = weekends
            input_data['dataset_type'] = dataset_type
            return dataset_type, input_data

        async def get_input_characterization_train(self, input_data, tables_dataset, training_dates, date_column,
                                                   datetimeFormat, categorical_columns, columns_in_df_from_hc):
            table_target = input_data['target_table']
            dataset_type = input_data['dataset_type']
            target = input_data['target']
            columns_for_target = [date_column]
            data_column = tables_dataset.get(dataset_type).get(table_target).get(target)
            for entry in data_column:
                if isinstance(entry, dict):
                    for k, v in entry.items():
                        for value in v:
                            columns_for_target.append(k + "_" + value)

                else:
                    if entry != target:
                        columns_for_target.append(table_target + "_" + entry)
            with open('utils_package/config_tables_db.json') as config_file:
                config_tables_db = json.load(config_file)
            columns_with_sunny_time = config_tables_db.get("columns_with_sunny_time")
            if target in columns_with_sunny_time:
                columns_for_target.append('sun_time')
            columns_for_target.append(target)
            columns_for_target = [column_ for column_ in columns_for_target if column_ in columns_in_df_from_hc]
            model_type = tables_dataset.get("time_series_columns").get(table_target).get(target)
            settings = tables_dataset.get("settings")
            settings['filters']['dayofweek'] = input_data['dayofweek']
            settings['filters']['exclude_weekends'] = input_data['exclude_weekends']
            settings['filters']['problem_type'] = model_type
            settings["transformations"]['weekends'] = input_data['weekends']
            settings["datetime_column_name"] = date_column
            settings["datetime_format"] = datetimeFormat
            settings["target_column_name"] = target
            settings["categorical_columns_names"] = categorical_columns
            settings["columns_names"] = columns_for_target
            characterization = {}
            characterization.update({"frequency": input_data['frequency']})
            characterization.update({"target_table": table_target})
            characterization.update({"target": target})
            characterization.update({"time_series": model_type})
            characterization.update({"dataset_type": dataset_type})
            characterization.update({"features": columns_for_target})
            input_data.update({'characteristics': characterization})
            input_data.update({'dataset_type': dataset_type})
            input_data.update({'training_dates': training_dates})
            input_data.update({"settings": settings})
            return input_data

        async def get_input_characterization_predict(self, input_data, tables_dataset, dataset_type,
                                                     first_start_timestamp, last_end_timestamp):
            table_target = input_data['target_table']
            target = input_data['target']
            columns_for_target = []
            data_column = tables_dataset.get(dataset_type).get(table_target).get(target)
            for entry in data_column:
                if isinstance(entry, dict):
                    for k, v in entry.items():
                        for value in v:
                            columns_for_target.append(k + "_" + value)

                else:
                    if entry != target:
                        columns_for_target.append(table_target + "_" + entry)
                    else:
                        columns_for_target.append(target)
            model_type = tables_dataset.get("time_series_columns").get(table_target).get(target)
            characterization = {}
            characterization.update({"frequency": input_data['frequency']})
            characterization.update({"target_table": input_data['target_table']})
            characterization.update({"target": input_data['target']})
            characterization.update({"time_series": model_type})
            characterization.update({"dataset_type": dataset_type})
            characterization.update({"features": columns_for_target})
            input_data.update({'characteristics': characterization})
            input_data.update({'dataset_type': dataset_type})
            input_data.update({"first_start_timestamp": first_start_timestamp})
            input_data.update({"last_end_timestamp": last_end_timestamp})
            return input_data

        def load_object_from_file(self, file_path):
            # Open the file and load the object using pickle
            with open(file_path, 'rb') as f:
                object = pickle.load(f)
            os.remove(file_path)
            # Return the loaded object
            return object

        def save_object_to_file(self, object, file_name):
            # Get the absolute path for the current working directory
            current_dir = os.getcwd()

            # Define the full path for the file
            file_path = os.path.join(current_dir, 'aux_files')
            file_path = os.path.join(file_path, file_name)

            # Save the object to a file using pickle
            with open(file_path, 'wb') as f:
                pickle.dump(object, f)

            # Return the file path
            return file_path

        async def _send_and_collect_response(self, request_type, agent, model, request_data, input_data, msg_id, model_id=None):
            print(timestamp_with_time_zone(), 'in send')
            receptor = f"{agent}@{self.agent.jid.domain}/{agent}"

            # Construct the body data for the message
            body_data = f"{request_type}|{model}|{request_data}|{input_data}"
            if model_id:
                body_data += f"|{model_id}"
            json_data = body_data

            # Compress the JSON data
            # compressed_data = zlib.compress(json_data.encode('utf-8'))
            #
            # # Encode the compressed data using Base64
            # encoded_data = base64.b64encode(compressed_data).decode('utf-8')

            # Create a message and set the encoded data as the body
            new_msg = Message(to=receptor)
            print(timestamp_with_time_zone(), 'sending to', receptor, 'in send and collect')
            new_msg.set_metadata("performative", "request")
            new_msg.set_metadata("thread", msg_id)
            # new_msg.body = encoded_data
            name_file = model+'_'+agent+'.pkl'
            path = self.save_object_to_file(json_data, name_file)
            new_msg.body = 'open|'+path
            print(f'{timestamp_with_time_zone()} Sending compressed and encoded message to {agent}')
            await self.send(new_msg)

            print(f"{timestamp_with_time_zone()} Request sent to ML agent for {agent}")
            response = None
            while not response:
                if self.agent.received_messages:
                    for msg in self.agent.received_messages:
                        if msg.get_metadata('thread') == msg_id and msg.get_metadata(
                                "performative") == "inform" and 'ml_response' in msg.body:
                            response = msg
                            self.agent.received_messages.remove(msg)
                            break
                if response:
                    break
                await asyncio.sleep(1)
            # self.agent.add_behaviour(self.agent.WaitResponse())
            # # response = None
            # # if self.agent.response_get:
            # #     response = self.agent.response_get
            # #     print(f"OuterBehaviour received result from InnerBehaviour: {response}")
            # #     # Reset shared_data after processing
            # #     self.agent.response_hist_data = None
            # # return response
            # response = None
            #
            # # We wait for the behaviour to update `response_get`
            # while not self.agent.response_get:
            #     print("Waiting for response from ML agent...")
            #     await asyncio.sleep(1)  # Wait for the WaitResponse behavior to process the message
            #
            # # Once we have the response
            # if self.agent.response_get:
            #     response = self.agent.response_get
            #     print(f"Received response from ML agent: {response.body}")
            #     # Reset shared_data after processing
            #     self.agent.response_get = None
            print(f"{timestamp_with_time_zone()} Received response from ML agent: {response.body}")
            return response

            # Wait for the response from Agent
            # while True:
            #     response = await self.receive(timeout=180)  # Adjust timeout as necessary
            #     print(response, thread_id)
            #     if response and response.get_metadata("thread") == thread_id:
            #         print(f"{datetime.now()} Response received from agent for {agent}: {response.body}")
            #         return response.body
            #     else:
            #         print(f"{datetime.now()} No response received for {agent}")
            #         return "Failed"

        # async def _send_and_collect_response(self, request_type, agent, model, request_data, input_data, model_id=None):
        #     print('in send')
        #     receptor = f"{agent}@{self.agent.jid.domain}/{agent}"
        #     new_msg = Message(to=receptor)
        #     new_msg.set_metadata("performative", "request")  # Set the "inform" FIPA performative
        #     body_data = f"{request_type}|{model}|{request_data}|{input_data}"
        #     if model_id:
        #         body_data += f"|{model_id}"
        #     json_data = body_data
        #     # await self.send(new_msg)
        #     # print(f"{datetime.now()} Request sent to ml agent for {agent}")
        #     #
        #     # # Wait for the response from Agent ML
        #     # response = await self.receive(timeout=180)  # Adjust timeout as necessary
        #     # if response:
        #     #     print(f"{datetime.now()} Response received from ml agent for {agent}: {response.body}")
        #     #     return response.body
        #     # else:
        #     #     print(f"{datetime.now()} No response received for {agent}")
        #     #     return "Failed"
        #
        #     async with asyncio.Lock():
        #         # Define chunk size and split the data
        #         chunk_size = 4096  # Define an appropriate chunk size
        #         total_length = len(json_data)
        #         num_chunks = (total_length + chunk_size - 1) // chunk_size
        #         # Send the number of chunks first
        #         initial_msg = Message(to=receptor)
        #         print(f'initial_msg: {initial_msg}')
        #         initial_msg.set_metadata("performative", "request")
        #         initial_msg.body = json.dumps({"num_chunks": num_chunks})
        #         await self.send(initial_msg)
        #         print('first msg sent')
        #         # Send the chunks
        #         async with asyncio.Lock():
        #             for i in range(0, total_length, chunk_size):
        #                 chunk = json_data[i:i + chunk_size]
        #                 chunk_msg = Message(to=receptor)
        #                 chunk_msg.set_metadata("performative", "request")
        #                 chunk_msg.body = chunk
        #                 async with asyncio.Lock():
        #                     await self.send(chunk_msg)
        #                 async with asyncio.Lock():
        #                     await asyncio.sleep(0.1)  # Optional: slight delay to prevent message flooding
        #
        #         print(f"{datetime.now()} Request sent to ML agent for {agent}")
        #
        #     # Wait for the response from Agent
        #     response = await self.receive(timeout=180)  # Adjust timeout as necessary
        #     if response:
        #         print(f"{datetime.now()} Response received from agent for {agent}: {response.body}")
        #         return response.body
        #     else:
        #         print(f"{datetime.now()} No response received for {agent}")
        #         return "Failed"

    class WaitHistoricalData(PeriodicBehaviour):
        async def run(self):
            msg = await self.receive()
            if msg:
                if msg.get_metadata("performative") == "inform" and 'historical_data_response' in msg.body:
                    # Store the result in the shared_data
                    self.agent.response_hist_data = msg
                    # print(f"WaitHistoricalData setting shared data: {self.agent.response_hist_data}")
                    self.kill()  # Stop the InnerBehaviour after handling the response
                else:
                    print(f"{timestamp_with_time_zone()} got another message {msg}")
    class WaitResponse(CyclicBehaviour):
        async def run(self):
            # Inner Behaviour: runs periodically after receiving "start"
            # print("InnerBehaviour running... Waiting for specific response.")
            msg = await self.receive()
            if msg:
                # print(f"WaitResponse received: {msg.body}")
                if msg.get_metadata("performative") == "inform":
                    # Store the result in the shared_data
                    self.agent.received_messages.append(msg)
                    # print(f"WaitResponse setting shared data: {self.agent.response_get}")
                    # self.kill()  # Stop the InnerBehaviour after handling the response
    class WaitTrainResponse(PeriodicBehaviour):
        async def run(self):
            # Inner Behaviour: runs periodically after receiving "start"
            # print("InnerBehaviour running... Waiting for specific response.")
            msg = await self.receive()
            if msg:
                if msg.get_metadata("performative") == "inform":
                    # Store the result in the shared_data
                    self.agent.response_get = msg
                    # print(f"WaitResponse setting shared data: {self.agent.response_get}")
                    self.kill()  # Stop the InnerBehaviour after handling the response

    # Setup function for the target agent
    async def setup(self):

        self.response_hist_data = None
        self.response_get = None
        self.received_messages = []
        self.id_agent_for_messages= None

        # Start ReceiveMsg behaviour
        receive_template = Template()
        receive_template.set_metadata("performative", "request")
        self.add_behaviour(self.ReceiveMsg(), receive_template)

        # Template for WaitHistoricalData
        # historical_template = Template()
        # historical_template.set_metadata("performative", "inform")
        # # historical_template.body = "historical_data_response"
        wait_template = Template()
        wait_template.set_metadata("performative", "inform")
        # ml_template.body = "ml_response"
        # Template for WaitResponse
        wait_train_template = Template()
        wait_train_template.set_metadata("performative", "inform")
        # ml_template.body = "ml_response"

        # Add both behaviours with their specific templates
        # self.add_behaviour(self.WaitHistoricalData(period=1), template=historical_template)
        self.add_behaviour(self.WaitResponse(), template=wait_template)
        self.add_behaviour(self.WaitTrainResponse(period=1), template=wait_train_template)

    # async def setup(self):
    #     self.response_hist_data = None
    #     self.response_get = None
    #     b = self.ReceiveMsg()
    #     self.add_behaviour(b)
