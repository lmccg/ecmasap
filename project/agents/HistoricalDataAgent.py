from peak import Agent, PeriodicBehaviour
import json
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
class TargetAgent(Agent):
    class GetData(PeriodicBehaviour):
        async def run(self):
            msg = await self.receive(10)
            if msg:
                with open('../utils_package/config_urls.json') as config_file:
                    config_urls = json.load(config_file)
                with open('../utils_package/config_settings.json') as config_file:
                    config_settings = json.load(config_file)
                with open('../utils_package/config_tables_db.json') as config_file:
                    config_tables_db = json.load(config_file)
                input_data = msg.body
                input_data = json.loads(input_data)
                if 'get_real_data' in input_data:
                    data = await self.get_real_data(input_data, config_settings, config_urls)
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = json.dumps(data)
                    await self.send(response_msg)
                    print(f"Data sent back to {msg.sender}: {data}")
                else:
                    # Make an async HTTP request to the external service
                    data = await self.obtain_data(input_data, config_settings, config_urls, config_tables_db)
                    response_msg = msg.make_reply()
                    response_msg.set_metadata("performative", "inform")
                    response_msg.body = json.dumps(data)
                    await self.send(response_msg)
                    print(f"Data sent back to {msg.sender}: {data}")

        async def get_real_data(self, input_data, config_settings, config_urls):
            try:
                time_out_val = 1800
                categorical_columns = []
                # fix format time for get file
                start_date = input_data['start_date']
                start_time = input_data['start_time']
                end_date = input_data['end_date']
                end_time = input_data['end_time']
                frequency = input_data['frequency']
                column = input_data['target']
                table = input_data['target_table'].lower()
                if start_time.count(":") >= 2:
                    # Find the index of the second colon
                    second_colon_index = start_time.find(":", start_time.find(":") + 1)

                    # Remove everything starting from the second colon onward
                    start_time = start_time[:second_colon_index]
                if end_time.count(":") >= 2:
                    # Find the index of the second colon
                    second_colon_index = end_time.find(":", end_time.find(":") + 1)

                    # Remove everything starting from the second colon onward
                    end_time = end_time[:second_colon_index]

                data = [str(frequency), table, column, start_date, end_date, start_time, end_time]
                fields = config_urls.get('fields_to_replace_emul')
                url_data = config_urls.get('url_table_column_data')
                new_url = await self.replaceFields(url_data, fields, data)
                async with aiohttp.ClientSession() as session:
                    async with session.get(new_url, timeout=time_out_val) as response:
                        response = await response.json()
                        status_code = response.status
                if status_code in config_settings.get('errors').values():
                    return None
                return response
            except Exception as ex:
                return ex

        async def obtain_data(self, input_data, config_settings, config_urls, config_tables_db):
            try:
                time_out_val = 1800
                categorical_columns = []
                # fix format time for get file
                start_date = input_data['start_date']
                start_time = input_data['start_time']
                end_date = input_data['end_date']
                end_time = input_data['end_time']
                frequency = input_data['frequency']
                column = input_data['target']
                table = input_data['target_table'].lower()
                dataset_type = input_data['dataset_type']
                if start_time.count(":") >= 2:
                    # Find the index of the second colon
                    second_colon_index = start_time.find(":", start_time.find(":") + 1)

                    # Remove everything starting from the second colon onward
                    start_time = start_time[:second_colon_index]
                if end_time.count(":") >= 2:
                    # Find the index of the second colon
                    second_colon_index = end_time.find(":", end_time.find(":") + 1)

                    # Remove everything starting from the second colon onward
                    end_time = end_time[:second_colon_index]

                data = [str(frequency), table, column, start_date, end_date, start_time, end_time]
                fields = config_urls.get('fields_to_replace_emul')
                url_data = config_urls.get('url_table_column_data')
                date_column = config_settings.get("date_column")
                datetime_format = config_settings.get("datetime_format")
                config_errors = config_settings.get('errors').values()
                new_url = await self.replaceFields(url_data, fields, data)
                async with aiohttp.ClientSession() as session:
                    async with session.get(new_url, timeout=time_out_val) as response:
                        await response.json()
                        status_code = response.status
                if status_code in config_settings.get('errors').values():
                    return -2, None, None, None, None, None, None, None, None
                url_resources = config_urls.get("url_resources")
                async with aiohttp.ClientSession() as session:
                    async with session.get(url_resources, timeout=time_out_val) as response:
                        response_resources = await response.json()
                resources_list = response_resources.get('resources')
                with open('../utils_package/tablesData.json') as f:
                    tables_for_dataset = json.load(f)
                columns_table = tables_for_dataset.get(dataset_type).get(table)
                column_for_that_value = columns_table.get(column)
                list_columns = []
                for c in column_for_that_value:
                    if isinstance(c, dict):
                        for new_table, other_col in c.items():
                            for col in other_col:
                                list_columns.append({col: new_table})
                    else:
                        list_columns.append({c: table})
                final_df = pd.DataFrame()
                first_one = True
                for dict_columns in list_columns:
                    for new_col, new_table in dict_columns.items():
                        data = [str(frequency), new_table, new_col, start_date, end_date, start_time, end_time]
                        new_url = await self.replaceFields(url_data, fields, data)
                        async with aiohttp.ClientSession() as session:
                            async with session.get(new_url, timeout=time_out_val) as response:
                                resp_data = await response.json()
                                status_code = response.status
                        if status_code in config_errors:
                            return -2, None, None, None, None, None, None, None, None
                        if isinstance(resp_data, list):
                            if len(resp_data) > 0:
                                first_row = resp_data[0]
                            else:
                                first_row = resp_data
                        else:
                            try:
                                if not isinstance(resp_data, dict):
                                    resp_data = json.loads(resp_data)
                                first_row = resp_data
                                resp_data = [resp_data]
                            except Exception as ex:
                                resp_data = []
                                first_row = {}
                        if new_col == column:
                            new_col_df = column
                            try:
                                unit_target = first_row['unit']
                            except Exception as ex:
                                found_unit = False
                                for entry in resources_list:
                                    if entry.get('name') == table:
                                        properties = entry.get('properties')
                                        for p in properties:
                                            if p.get('name') == column:
                                                unit_target = p.get('unit')
                                                found_unit = True
                                                break
                                        if found_unit:
                                            break
                        else:
                            new_col_df = new_table + "_" + new_col
                        try:

                            dataframe_column = [date_column, new_col_df]
                            new_df = pd.DataFrame(columns=dataframe_column)
                            # print(Utils.timestamp_with_time_zone(), 'line 239')
                            col_data = []
                            date_col = []
                            for entry in resp_data:
                                first_row = True
                                # print('index', index, 'value', entry.get('value'))
                                if isinstance(entry.get('value'), str):
                                    if new_col_df not in categorical_columns and new_col_df != column:
                                        categorical_columns.append(new_col_df)
                                    col_data.append(entry.get('value'))
                                else:
                                    col_data.append(entry.get('value'))
                                date_value = entry.get('start_at')
                                date_col.append(date_value)
                            if len(date_col) > 0:
                                new_df[new_col_df] = col_data
                                new_df[date_column] = date_col
                                if first_one:
                                    final_df = new_df.copy()
                                    first_one = False
                                else:
                                    final_df = pd.merge(final_df, new_df, how='inner', on=[date_column])
                        except Exception as e:
                            pass
                new_df = final_df
                new_df = new_df[[col for col in new_df.columns if col != column] + [column]]
                response = await self.fix_dataset(new_df, start_date, end_date, start_time, end_time, frequency,
                                                            column, date_column, datetime_format,
                                                            config_settings, config_tables_db)
                columns_with_sunny_time = config_tables_db.get("columns_with_sunny_time")
                if column in columns_with_sunny_time:
                    categorical_columns.append('sun_time')
                response = response + (categorical_columns,)
                return response
            except Exception as ex:
                return ex

        async def fix_dataset(self, dataset, startDate, endDate, startTime, endTime, time, target_name,
                              date_column,
                              datetimeFormat, config_settings, config_tables_db):

            response = await self.missingValues(dataset, startDate, endDate, startTime, endTime, time,
                                                   target_name, date_column, datetimeFormat, config_settings,
                                                   config_tables_db)
            # print('then', response[1])
            return response

        async def remove_dataset_ouliers(self, dataset):
            list_outliers_index = []
            for c in dataset:
                if not isinstance(dataset[c].iloc[0], (str, bool, np.bool_, np.str_)):
                    list_outliers_index.extend(await self.column_outliers(dataset, c))
            cleaned_dataset = await self.remove_outliers(dataset, list_outliers_index)
            return cleaned_dataset

        async def column_outliers(self, df, c):
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 4 * iqr
            upper_bound = q3 + 4 * iqr
            o = df.index[(df[c] < lower_bound) | (df[c] > upper_bound)]
            return o

        async def remove_outliers(self, df, outliers):
            outliers = sorted(set(outliers))
            df.drop(outliers)
            return df

        async def missingValues(self, df, startDate, endDate, startTime, endTime, time, target_name,
                                date_columm,
                                datetimeFormat, config_settings, config_tables_db):
            hour_format = config_settings.get("hour_format")
            binary_columns = config_tables_db.get("binary_columns")
            columns_with_sunny_time = config_tables_db.get("columns_with_sunny_time")
            df = await self.remove_dataset_ouliers(df)
            try:
                start_date = datetime.strptime((startDate + " " + startTime), datetimeFormat)
            except:
                start_date = datetime.strptime((startDate + " " + startTime + ":00"), datetimeFormat)
            try:
                end_date = datetime.strptime((endDate + " " + endTime), datetimeFormat)
            except:
                end_date = datetime.strptime((endDate + " " + endTime + ":00"), datetimeFormat)
            # Define the frequency in minutes
            frequency = time  # Generate a date range using the start date, end date, and frequency
            date_range = [start_date + i * timedelta(minutes=frequency) for i in
                          range(int(((end_date - start_date).total_seconds() / 60) / frequency) + 1)]
            # Create a dataframe using the date range
            df2 = pd.DataFrame({target_name: np.nan, date_columm: date_range})
            # Format the date column to include only hour and minute information
            df2[date_columm] = df2[date_columm].dt.strftime(datetimeFormat)
            target_column_values = df2[date_columm].values.tolist()
            for i in df.index:
                r = df[date_columm][i]
                if r not in target_column_values:
                    df = df.drop(i)
            df3 = pd.merge(df2, df, on=date_columm, how='outer')
            df3 = df3.sort_values(date_columm)  # Reset the index of the dataframe
            df3 = df3.reset_index(drop=True)
            x_merge = target_name + '_x'
            y_merge = target_name + '_y'
            df3[target_name] = df3[y_merge]
            df3 = df3.drop(columns=x_merge)
            df3 = df3.drop(columns=y_merge)
            columns_with_missing_values = []
            for c in df3:
                kc = df3[c].isnull().sum()
                if kc > 0:
                    columns_with_missing_values.append(c)
            list_week_day = []
            list_hour = []
            for index in df3.index:
                raw = df3.at[index, date_columm]
                hour_raw = raw.split(" ")[1]
                hour_raw = datetime.strptime(hour_raw, hour_format).time()
                date_raw = datetime.strptime(raw, datetimeFormat)
                # get day of week as an integer
                week_day = date_raw.weekday()
                list_week_day.append(week_day)
                list_hour.append(hour_raw)
            df3['dayofweek'] = list_week_day
            df3['hour'] = list_hour
            if target_name in columns_with_sunny_time:
                list_sunny_time = []
                for index in df3.index:
                    raw = df3.at[index, date_columm]
                    dt = datetime.strptime(raw, datetimeFormat)
                    month = dt.month
                    hour = dt.hour
                    if 6 <= hour < 19 and 3 <= month <= 10:
                        list_sunny_time.append('yes')
                    else:
                        list_sunny_time.append('no')
                df3['sun_time'] = list_sunny_time
            for c in columns_with_missing_values:
                # replacing missing values with the average of the same week days, in the same hour
                try:
                    if c in binary_columns:
                        mode_values = df3.groupby(['dayofweek', 'hour'])[c].apply(lambda x: x.mode())
                        values = df3[c]
                        for v in values.items():  # tuple(index,value)
                            if pd.isnull(v[1]):
                                index = v[0]
                                raw = df3.loc[index]
                                hour_raw = raw[date_columm].split(" ")[1]
                                hour_raw = datetime.strptime(hour_raw, hour_format).time()
                                date_raw = datetime.strptime(raw[date_columm], datetimeFormat)
                                week_day = date_raw.weekday()
                                try:
                                    mode_raw = mode_values[week_day][hour_raw]
                                    if len(mode_raw.dropna()) > 0:  # Ensure there are mode values
                                        mode_count = mode_raw.value_counts().idxmax()  # Select the mode that occurs most frequently
                                        df3.loc[index, c] = mode_count
                                except Exception as e:
                                    # try:
                                    mode_count = df3[c].dropna().mode()
                                    if len(mode_count) > 0:  # Ensure there are non-NaN mode values
                                        df3.loc[index, c] = mode_count.iloc[0]  # Select the first mode value

                    else:
                        df3[c] = pd.to_numeric(df3[c])
                        mean = df3.groupby([df3['dayofweek'], df3['hour']], sort=False)[c].apply(lambda x: x.mean())
                        values = df3[c]
                        for v in values.items():  # tuple(index,value)
                            if np.isnan(v[1]) or v[1] < 0:
                                index = v[0]
                                raw = df3.loc[index]
                                hour_raw = raw[date_columm].split(" ")[1]
                                hour_raw = datetime.strptime(hour_raw, hour_format).time()
                                date_raw = datetime.strptime(raw[date_columm], datetimeFormat)
                                # get day of week as an integer
                                week_day = date_raw.weekday()
                                mean_raw = mean[week_day][hour_raw]
                                try:
                                    if index > 0:
                                        previous = index - 1
                                        previuos_value = df3.at[previous, c]
                                        if np.isnan(previuos_value):
                                            previuos_value = 0
                                    else:
                                        previuos_value = 0
                                    try:
                                        if index < len(values) - 1:
                                            last = index + 1
                                            last_value = df3.at[last, c]
                                            if np.isnan(last_value):
                                                last_value = 0
                                        else:
                                            last_value = 0
                                        if np.isnan(mean_raw):
                                            mean_raw = 0
                                        if mean_raw == 0 and previuos_value == 0 and last_value == 0:
                                            new_value = df3[c].mean()
                                        else:
                                            new_value = (0.5 * mean_raw) + (
                                                        0.5 * ((previuos_value + last_value) / 2))
                                        if np.isnan(new_value):
                                            new_value = 0
                                        df3.loc[index, c] = new_value
                                    except:
                                        pass
                                except:
                                    pass
                except:
                    pass
            target_column = df3.pop(target_name)
            timestamp_column = df3.pop(date_columm)
            df3.insert(0, date_columm, timestamp_column)
            df3[target_name] = target_column
            df3 = df3.drop(columns='hour')
            df3 = df3.drop(columns='dayofweek')
            df_dataset = df3.values
            df_dataset = df_dataset.tolist()
            return df_dataset, date_columm, datetimeFormat

        async def replaceFields(self, url, fields, data):
            newUrl = url
            for field, info in zip(fields, data):
                if info is None:
                    info = ''
                newUrl = newUrl.replace(field, info)
            newUrl = newUrl.replace(" ", "")
            return newUrl


    # Setup function for the historical data agent
    async def setup(self):
        print(f"Agent {self.jid} starting...")
        b = self.GetData()
        self.add_behaviour(b)