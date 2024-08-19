from peak import Agent, CyclicBehaviour, Message
import asyncio
import json
from datetime import datetime, timedelta
import datetime as dt


class TargetAgent(Agent):
    class ReceiveMsg(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"TargetAgent: {msg.sender} sent me a message: '{msg.body}'")
                parts_of_msg = msg.body.split("|")
                request_type = parts_of_msg[0]
                target = parts_of_msg[1]
                request_data = parts_of_msg[2]
                request_data = json.loads(request_data)
                process = True
                if request_type == 'predict' and ('start_date' not in request_data or 'start_time' not in request_data):
                    process = False
                elif (request_type == 'train' or request_type == 'retrain') and ('end_date' not in request_data or 'end_time' not in request_data):
                    process = False
                if process:
                    print(parts_of_msg)
                    with open('../utils_package/config_agents.json') as config_file:
                        config_agents = json.load(config_file)
                    with open('../utils_package/config_settings.json') as config_file:
                        config_settings = json.load(config_file)
                    with open('../utils_package/tablesData.json') as config_file:
                        tables_dataset = json.load(config_file)
                    datetime_format = config_settings.get('datetime_format')
                    if request_type == 'predict':
                        first_start_timestamp, last_end_timestamp, request_data = await self.fix_timestamps(request_data,
                                                                                                      datetime_format)
                    elif request_type == 'train' or request_type == 'retrain':
                        _, _, request_data = await self.fix_timestamps(request_data, datetime_format)
                    dataset_type, request_data = await self.checkDatasetType(request_data, datetime_format)
                    table_target = request_data['target_table']
                    model_type = tables_dataset.get("time_series_columns").get(table_target).get(target)
                    agents = config_agents['ml_model_agents'][model_type]
                    # get the historical data
                    historical_data_agent = config_agents["historical_data_agent"]
                    msg_data = Message(to=f"{historical_data_agent}@{self.agent.jid.domain}/{historical_data_agent}")
                    msg_data.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                    if isinstance(request_data, dict):
                        request_data = json.dumps(request_data)
                    msg_data.body = request_data
                    await self.send(msg_data)
                    response = await self.receive(timeout=60)
                    input_dataset = []
                    if response:
                        if response.get_metadata("performative") == "inform":
                            input_data = response.body
                            input_data = json.loads(input_data)
                    input_dataset = input_data[0]
                    # ask predictions
                    if request_type == 'predict':
                        # get the models to predict
                        agent_get_ml_models = config_agents["rl_selector_agent"]
                        predict_msg = Message(to=f"{agent_get_ml_models}@{self.agent.jid.domain}/{agent_get_ml_models}")
                        predict_msg.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                        if not isinstance(request_data, dict):
                            request_data = json.loads(request_data)

                        request_data = await self.get_input_characterization_predict(request_data, tables_dataset,
                                                                                     dataset_type, first_start_timestamp,
                                                                                     last_end_timestamp)
                        if isinstance(request_data, dict):
                            request_data = json.dumps(request_data)
                        predict_msg.body = request_data
                        await self.send(predict_msg)
                        response = await self.receive(timeout=60)
                        models = []
                        if response:
                            if response.get_metadata("performative") == "inform":
                                models = response.body
                                models = json.loads(models)

                        # Wait for a response
                        print('lets send some requests')
                        tasks = []
                        predict_one_model = False
                        while models:
                            entry = models.pop(0)
                            ml_model = entry[1]
                            type_model = entry[2]
                            agents = config_agents['ml_model_agents'][type_model]
                            agent = next((key for key, value in agents.items() if value.lower() == ml_model.lower()),
                                         None)
                            model = entry[0]
                            #model é model id e model id será o meu boolean
                            result = await self._send_and_collect_response(request_type, agent, model, request_data,
                                                                           input_dataset, False)
                            if result != 'Failed':
                                predict_one_model = True
                                response_msg = msg.make_reply()
                                response_msg.set_metadata("performative", "inform")
                                response_msg.body = json.dumps(result)
                                await self.send(response_msg)
                                break
                        if not predict_one_model:
                            response_msg = msg.make_reply()
                            response_msg.set_metadata("performative", "inform")
                            response_msg.body = "It was not possible to send a response for request received."
                            await self.send(response_msg)

                        if models:
                            for entry in models:
                                ml_model = entry[1]
                                type_model = entry[2]
                                agents = config_agents['ml_model_agents'][type_model]
                                agent = next(
                                    (key for key, value in agents.items() if value.lower() == ml_model.lower()),
                                    None)
                                model = entry[0]
                                #model é model id
                                task = self._send_and_collect_response(request_type, agent, model, request_data, input_dataset,
                                                                       True)
                                tasks.append(task)
                            await asyncio.gather(*tasks)

                            response_msg = msg.make_reply()
                            response_msg.set_metadata("performative", "inform")
                            response_msg.body = "Request executed!"
                            await self.send(response_msg)
                    elif request_type == 'train':
                        date_column = input_data[1]
                        datetimeFormat = input_data[2]
                        categorical_columns = input_data[3]
                        dataset_types = config_settings.get('possible_data_types')
                        dataset_types = list(dataset_types.keys())
                        response = msg.make_reply()
                        response.set_metadata("performative", "inform")
                        response.body = "Requested train"
                        await self.send(response)
                        training_dates = await self.get_training_dates(request_data, config_settings)
                        request_data = await self.get_input_characterization_train(request_data, tables_dataset,
                                                                                     training_dates, date_column, datetimeFormat, categorical_columns)
                        asyncio.create_task(
                            self.run_train_in_background(request_data, request_type, input_dataset, dataset_types, agents))
                    elif request_type == 'retrain':
                        date_column = input_data[1]
                        datetimeFormat = input_data[2]
                        categorical_columns = input_data[3]
                        response = msg.make_reply()
                        response.set_metadata("performative", "inform")
                        response.body = "Requested retrain"
                        await self.send(response)
                        model_id = parts_of_msg[3]
                        training_dates = await self.get_training_dates(request_data, config_settings)
                        request_data = await self.get_input_characterization_train(request_data, tables_dataset,
                                                                                   training_dates, date_column, datetimeFormat, categorical_columns)
                        asyncio.create_task(
                            self.run_retrain_in_background(request_data, request_type, input_dataset, model_id, agents))
                else:
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = "Please fix the timestamps provided! For train and retrain is mandatory to have 'end_date' and 'end_time' fields. For predict is mandatory to have 'start_date' and 'start_time' fields"
                    await self.send(response_msg)
        async def run_train_in_background(self, request_data, request_type, input_data, dataset_types, agents):
            tasks = []
            for dataset_type in dataset_types:
                for agent, model in agents.items():
                    if not isinstance(request_data, dict):
                        request_data = json.loads(request_data)
                    request_data['dataset_type'] = dataset_type
                    if isinstance(request_data, dict):
                        request_data = json.dumps(request_data)
                    task = self._send_and_collect_response(request_type, agent, model, request_data, input_data)
                    tasks.append(task)
            await asyncio.gather(*tasks)

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

        async def run_retrain_in_background(self, request_data, request_type, input_data, model_id, agents):
            if not isinstance(request_data, dict):
                request_data = json.loads(request_data)
            model = request_data['model']
            agent = next(
                (key for key, value in agents.items() if value.lower() == model.lower()),
                None)
            if isinstance(request_data, dict):
                request_data = json.dumps(request_data)
            await self._send_and_collect_response(request_type, agent, model, request_data, input_data,
                                                           model_id)

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

        async def fix_timestamps(self, input_data, datetime_format):
                start_date = input_data['start_date']
                start_time = input_data['start_time']
                end_date = input_data['end_date']
                end_time = input_data['end_time']
                frequency_minutes = input_data['frequency']
                # Parse the input strings into datetime objects
                start_datetime = datetime.strptime(f"{start_date} {start_time}", datetime_format)
                if end_date and end_time:
                    end_datetime = datetime.strptime(f"{end_date} {end_time}", datetime_format)
                else:
                    end_datetime = start_datetime

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




        async def checkDatasetType(self, input_data,
                                   datetimeFormat):
            start_date = input_data['start_date']
            start_time = input_data['start_time']
            end_date = input_data['end_date']
            end_time = input_data['end_time']
            dayofweek = input_data.get('dayofweek', [])
            exclude_weekends = input_data.get('exclude_weekends', False)
            weekends = input_data.get('weekends', False)
            frequency = input_data['frequency']
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

        async def get_input_characterization_train(self, input_data, tables_dataset, training_dates, date_column, datetimeFormat, categorical_columns):
            table_target = input_data['target_table']
            dataset_type = input_data['dataset_type']
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
            settings = tables_dataset.get("settings")
            settings['filters']['dayofweek'] = input_data['dayofweek']
            settings['filters']['exclude_weekends'] = input_data['exclude_weekends']
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

        async def _send_and_collect_response(self, request_type, agent, model, request_data, input_data, model_id=None):
            receptor = f"{agent}@{self.agent.jid.domain}/{agent}"
            new_msg = Message(to=receptor)
            new_msg.set_metadata("performative", "request")  # Set the "inform" FIPA performative
            if model_id:
                new_msg.body = request_type + "|" + model + "|" + request_data + "|" + input_data + "|" + model_id
            else:
                new_msg.body = request_type + "|" + model + "|" + request_data + "|" + input_data
            await self.send(new_msg)
            print(f"{datetime.now()} Request sent to ml agent for {agent}")

            # Wait for the response from Agent C
            response = await self.receive(timeout=180)  # Adjust timeout as necessary
            if response:
                print(f"{datetime.now()} Response received from ml agent for {agent}: {response.body}")
                return response.body
            else:
                print(f"{datetime.now()} No response received for {agent}")
                return "Failed"

    # Setup function for the target agent
    async def setup(self):
        print(f"Agent {self.jid} starting...")
        b = self.ReceiveMsg()
        self.add_behaviour(b)
