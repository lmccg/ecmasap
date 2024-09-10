import pandas as pd
from pandas.api.types import is_categorical_dtype
from datetime import datetime, timedelta
import numpy as np
import pytz
import json
import os
class Dataset:
    async def fix_dataset(file):
        if not isinstance(file, pd.DataFrame):
            try:
                dataset = pd.read_csv(file)
            except:
                try:
                    dataset = pd.read_excel(file)
                except:
                    return -1
        else:
            dataset = file
        response = await Dataset.missingValues(dataset)
        return response

    async def remove_dataset_ouliers(dataset):
        list_outliers_index = []
        for c in dataset:
            if not isinstance(dataset[c].iloc[0], (str, bool, np.bool_, np.str_)):
                list_outliers_index.extend(await Dataset.column_outliers(dataset, c))
        cleaned_dataset = await Dataset.remove_outliers(dataset, list_outliers_index)
        return cleaned_dataset

    async def column_outliers(df, c):
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 4 * iqr
        upper_bound = q3 + 4 * iqr
        o = df.index[(df[c] < lower_bound) | (df[c] > upper_bound)]
        return o

    async def remove_outliers(df, outliers):
        outliers = sorted(set(outliers))
        df.drop(outliers)
        return df

    async def missingValues(df):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE_DIR, 'config.json'), encoding='utf-8') as f:
            config = json.load(f)
        datetimeFormat = config.get("datetime_format")
        hour_format = config.get("hour_format")
        date_columm = config.get('date_col')
        df = await Dataset.remove_dataset_ouliers(df)

        columns_with_missing_values = []
        for c in df:
            kc = df[c].isnull().sum()
            if kc > 0:
                columns_with_missing_values.append(c)
            # print("Number of missing values before preprocessing: {} in column {}".format(kc, c))
        list_week_day = []
        list_hour = []
        for index in df.index:
            raw = df.at[index, date_columm]
            hour_raw = raw.split(" ")[1]
            hour_raw = datetime.strptime(hour_raw, hour_format).time()
            date_raw = datetime.strptime(raw, datetimeFormat)
            # get day of week as an integer
            week_day = date_raw.weekday()
            list_week_day.append(week_day)
            list_hour.append(hour_raw)
        df['dayofweek'] = list_week_day
        df['hour'] = list_hour
        
        for c in columns_with_missing_values:
            # replacing missing values with the average of the same week days, in the same hour
            try:
                check_column = Dataset.is_binary_column_or_categorical(df, c)
                if not check_column:
                    try:
                        df[c] = pd.to_numeric(df[c])
                    except:
                        check_column = True
                if check_column:
                    mode_values = df.groupby(['dayofweek', 'hour'])[c].apply(lambda x: x.mode())
                    values = df[c]
                    for v in values.items():  # tuple(index,value)
                        if pd.isnull(v[1]):
                            index = v[0]
                            raw = df.loc[index]
                            hour_raw = raw[date_columm].split(" ")[1]
                            hour_raw = datetime.strptime(hour_raw, hour_format).time()
                            date_raw = datetime.strptime(raw[date_columm], datetimeFormat)
                            week_day = date_raw.weekday()
                            try:
                                mode_raw = mode_values[week_day][hour_raw]
                                if len(mode_raw.dropna()) > 0:  # Ensure there are mode values
                                    mode_count = mode_raw.value_counts().idxmax()  # Select the mode that occurs most frequently
                                    df.loc[index, c] = mode_count
                            except Exception as e:
                                # try:
                                mode_count = df[c].dropna().mode()
                                if len(mode_count) > 0:  # Ensure there are non-NaN mode values
                                    df.loc[index, c] = mode_count.iloc[0]  # Select the first mode value

                else:
                    mean = df.groupby([df['dayofweek'], df['hour']], sort=False)[c].apply(lambda x: x.mean())
                    values = df[c]
                    for v in values.items():  # tuple(index,value)
                        if np.isnan(v[1]) or v[1] < 0:
                            index = v[0]
                            raw = df.loc[index]
                            hour_raw = raw[date_columm].split(" ")[1]
                            hour_raw = datetime.strptime(hour_raw, hour_format).time()
                            date_raw = datetime.strptime(raw[date_columm], datetimeFormat)
                            # get day of week as an integer
                            week_day = date_raw.weekday()
                            mean_raw = mean[week_day][hour_raw]
                            try:
                                if index > 0:
                                    previous = index - 1
                                    previous_value = df.at[previous, c]
                                    if np.isnan(previous_value):
                                        previous_value = 0
                                else:
                                    previous_value = 0
                                try:
                                    if index < len(values) - 1:
                                        last = index + 1
                                        last_value = df.at[last, c]
                                        if np.isnan(last_value):
                                            last_value = 0
                                    else:
                                        last_value = 0
                                    if np.isnan(mean_raw):
                                        mean_raw = 0
                                    if mean_raw == 0 and previous_value == 0 and last_value == 0:
                                        new_value = df[c].mean()
                                    else:
                                        new_value = (0.5 * mean_raw) + (0.5 * ((previous_value + last_value) / 2))
                                    if np.isnan(new_value):
                                        new_value = 0
                                    df.loc[index, c] = new_value
                                except Exception as e:
                                    print(f'[{str(Dataset.timestamp_with_time_zone())}] [mlt] exception dataset line 185 {str(e)}')
                                    pass
                            except Exception as e:
                                print(f'[{str(Dataset.timestamp_with_time_zone())}] [mlt] exception dataset line 188 {str(e)}')
                                pass
            except Exception as e:
                print(f'[{str(Dataset.timestamp_with_time_zone())}] [mlt] exception dataset line 191 {str(e)} column {str(c)}')
                pass
        timestamp_column = df.pop(date_columm)
        df.insert(0, date_columm, timestamp_column)
        df = df.drop(columns='hour')
        df = df.drop(columns='dayofweek')
        return df
    
    def timestamp_with_time_zone():
        with open('./config_settings.json') as f:
            config_settings = json.load(f)
        current_time = datetime.now()
        timezone = config_settings.get('timezone')  # Desired timezone
        formated_timestamp = config_settings.get('datetime_format')
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

    def is_binary_column_or_categorical(df, column_name):
        value_counts = df[column_name].value_counts().index
        binary_ = set(value_counts).issubset({0, 1})
        boolean_ = set(value_counts).issubset({False, True})
        try:
            categorical = is_categorical_dtype(df[column_name])
        except:
            categorical = isinstance(df[column_name].dtype, pd.CategoricalDtype)
        string_ = df[column_name].dtype == 'string'
        if binary_:
            return binary_
        elif boolean_:
            return boolean_
        elif categorical:
            return categorical
        elif string_:
            return string_
        else:
            return False

