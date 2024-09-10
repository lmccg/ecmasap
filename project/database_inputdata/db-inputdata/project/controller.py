import repository as Repo
import os, json, random
import pandas as pd
import numpy as np
from quart import jsonify
from jsonpath_ng.ext import parse
from datetime import datetime, time, timedelta
from dateutil.relativedelta import relativedelta
from dataset import Dataset as dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'config.json'), encoding='utf-8') as f:
    config = json.load(f)


async def internal_server_error(e):
    response = {
        "error": "Internal Server Error",
        "message": str(e)
    }
    return jsonify(response)


async def getAllTablesController():
    listAllTables = []
    tables = config.get('tables').get('db')
    for table in tables:
        listTables = []
        infos = table.get('data').get('info')
        for info in infos:
            listTables.append(
                {"name": info.get("info"), "description": info.get("description"), "unit": info.get("unit")})
        listAllTables.append(
            {"name": table.get('name'), "description": table.get("description"), "properties": listTables})
    tables = config.get('tables').get('emul')
    for table in tables:
        listTables = []
        infos = table.get('data').get('info')
        for info in infos:
            listTables.append(
                {"name": info.get("info"), "description": info.get("description"), "unit": info.get("unit")})
        listAllTables.append(
            {"name": table.get('name'), "description": table.get("description"), "properties": listTables})

    result = {"resources": listAllTables}
    # {name:,description:,properties:[{name:,description:}] }
    return result

async def formatJsonForATableInEmul(listOfcolumnsInTable, resultJson):
    valid_columns = config.get('json_path_columns_emul')
    jsonpath_expression = parse(valid_columns)
    matches = [match.value for match in jsonpath_expression.find(config)]
    for i in range(0, len(listOfcolumnsInTable)):
        column = config["tables"]["emul"][0]["data"]["info"][i]["info"]
        tableColumn = config["tables"]["emul"][0]["data"]["info"][i]["table"]
        newJSON = {column: {}}
        newJSON[column] = await getDataEmulAll(column,tableColumn, matches)
        resultJson.append(newJSON)
    return resultJson    

async def getDataEmulAll(column, table_column, matches):
    if table_column == 'sensor':
        return await ActivityAtual(column)
    elif table_column == 'cons':
        column_parts = column.split('_')
        if column_parts[1] == 'light':
            return await IndividualConsumptionInRooms(column, matches)
        elif column_parts[1] == 'ac':
            return await StatesOfAC(column, matches)
           
async def formatJsonForATableInDb(listOfcolumnsInTable, idx, requestedAt, resultJson):
    for i in range(0, len(listOfcolumnsInTable)):
        # our column
        column = config["tables"]["db"][idx]["data"]["info"][i]["info"]
        # inicial json
        newJSON = {column: {}}

        unit = config["tables"]["db"][idx]["data"]["info"][i]["unit"]
        try:
            # when there is not sum
            nameTable = config["tables"]["db"][idx]["data"]["info"][i]["tableResource"][0]["name"]
            columnName = config["tables"]["db"][idx]["data"]["info"][i]["tableResource"][0]["column"]
            multiplier = config["tables"]["db"][idx]["data"]["info"][i]["tableResource"][0]["multiplier"]
            # replace query
            query = config.get('queryResource')
            query = query.replace('COLUMN', columnName)
            query = query.replace('TABLE', nameTable)
            # execute query
            data = await Repo.Repo.executeQuery(query)

            value = 0
            if unit == '%':
                value = await cap_value(data[0][1] * multiplier)
            else:
                value = data[0][1] * multiplier

            newJSON[column]["value"] = round(value, 2)
            newJSON[column]["unit"] = unit
            newJSON[column]["requested_at"] = requestedAt
            formatted_datetime = data[0][0].strftime("%Y-%m-%d %H:%M:%S.%f")
            newJSON[column]["read_at"] = formatted_datetime
            # append to the final json
            resultJson.append(newJSON)

        except:
            # when there is sum
            listSum = []
            listSumJson = config["tables"]["db"][idx]["data"]["info"][i]["tableResource"][0]["sum"]
            for i in range(0, len(listSumJson)):
                nameTable = listSumJson[i]["name"]
                columnName = listSumJson[i]["column"]
                multiplier = listSumJson[i]["multiplier"]
                # replace query
                query = config.get('queryResource')
                query = query.replace('COLUMN', columnName)
                query = query.replace('TABLE', nameTable)
                # execute query
                data = await Repo.Repo.executeQuery(query)

                if unit == '%':
                    value = await cap_value(data[0][1] * multiplier)
                    listSum.append(value)
                else:
                    listSum.append(data[0][1] * multiplier)

            newJSON[column]["value"] = round(sum(listSum), 2)
            newJSON[column]["unit"] = unit
            newJSON[column]["requested_at"] = requestedAt
            formatted_datetime = data[0][0].strftime("%Y-%m-%d %H:%M:%S.%f")
            newJSON[column]["read_at"] = formatted_datetime
            # append to the final json
            resultJson.append(newJSON)

    return resultJson


async def cap_value(value):
    if value > 100:
        return 100
    elif value < 0:
        return 0
    else:
        return value


async def getRtDataFromAColumn(table, matchesTable, column):
    try:
        # verify if the column is on config.json
        if column is not None:
            valid_columns = config.get('json_path_columns')
            jsonpath_expression = parse(valid_columns)
            matches = [match.value for match in jsonpath_expression.find(config)]
            if column not in matches:
                result = {"Error": "column name is not valid!"}
                return jsonify(result)

        # get index of table in config.json
        idx = matchesTable.index(table)

        # get jsonFinal
        listOfcolumnsInTable = config["tables"]["db"][idx]["data"]["info"]
        resultJson = await formatJsonForAColumnInATable(listOfcolumnsInTable, column, idx)

        return jsonify(resultJson)
    except Exception as e:
        result = {"error": str(e)}
        return jsonify(result)


async def formatJsonForAColumnInATable(listOfcolumnsInTable, column, idx):
    for i in range(0, len(listOfcolumnsInTable)):

        if column == config["tables"]["db"][idx]["data"]["info"][i]["info"]:
            try:
                # when there is not sum
                unit = config["tables"]["db"][idx]["data"]["info"][i]["unit"]
                nameTable = config["tables"]["db"][idx]["data"]["info"][i]["tableResource"][0]["name"]
                columnName = config["tables"]["db"][idx]["data"]["info"][i]["tableResource"][0]["column"]
                multiplier = config["tables"]["db"][idx]["data"]["info"][i]["tableResource"][0]["multiplier"]
                query = config.get('queryResource')
                # replace query
                query = query.replace('COLUMN', columnName)
                query = query.replace('TABLE', nameTable)

                # execute query
                data = await Repo.Repo.executeQuery(query)
                value = 0
                if unit == '%':
                    value = await cap_value(data[0][1] * multiplier)
                else:
                    value = data[0][1] * multiplier
                # final json
                resultJson = {
                    "requested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "read_at": data[0][0].strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "value": round(value, 2),
                    "unit": unit
                }

            except:
                # when there is sum
                listSum = []
                listSumJson = config["tables"]["db"][idx]["data"]["info"][i]["tableResource"][0]["sum"]
                unit = config["tables"]["db"][idx]["data"]["info"][i]["unit"]
                for i in range(0, len(listSumJson)):
                    nameTable = listSumJson[i]["name"]
                    columnName = listSumJson[i]["column"]
                    multiplier = listSumJson[i]["multiplier"]
                    # replace query
                    query = config.get('queryResource')
                    query = query.replace('COLUMN', columnName)
                    query = query.replace('TABLE', nameTable)
                    # execute query

                    data = await Repo.Repo.executeQuery(query)
                    if unit == '%':
                        value = await cap_value(data[0][1] * multiplier)
                        listSum.append(value)
                    else:
                        listSum.append(data[0][1] * multiplier)

                query = config.get('queryResource')
                # replace query
                query = query.replace('COLUMN', columnName)
                query = query.replace('TABLE', nameTable)
                # execute query
                data = await Repo.Repo.executeQuery(query)
                # final json
                resultJson = {
                    "requested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "read_at": data[0][0].strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "value": round(sum(listSum), 2),
                    "unit": unit
                }
            return resultJson


async def getRtDataEmul(column, info_table):
    valid_columns = config.get('json_path_columns_emul')
    jsonpath_expression = parse(valid_columns)
    matches = [match.value for match in jsonpath_expression.find(config)]
    if column is not None:
        if column not in matches:
            result = {"Error": "column name is not valid!"}
            return jsonify(result)
        for i in info_table:
            if i.get('info') == column:
                table_column = i.get('table')
        column_parts = column.split('_')
        if table_column == 'sensor':
            return jsonify(await ActivityAtual(column))
        elif table_column == 'cons':
            if column_parts[1] == 'light':
                return jsonify(await IndividualConsumptionInRooms(column, matches))
            elif column_parts[1] == 'ac':
                return jsonify(await StatesOfAC(column, matches))
    else:
        result = {"Error": "please set a column name!"}
        return jsonify(result)

async def calculateProbActivity(hour, date):
    
    time_intervals = config['activity']
    hour = int(hour) 
    probability = 0
    for interval in time_intervals:
        start = datetime.strptime(interval['start'], "%H:%M").time()
        end = datetime.strptime(interval['end'], "%H:%M").time()
        prob = interval['probability']

        if start < end:
            if start <= time(hour) < end: 
                probability = prob
                break
        else:  
            if start <= time(hour) or time(hour) < end:  
                probability = prob

    if await is_business_day(date):
        value = random.choices([True, False], weights=[probability, 100 - probability])[0]
    else:
        value = False

    return value

async def ActivityAtual(room):
    now = datetime.now()
    hour = now.strftime("%H")
    date = now.strftime('%Y-%m-%d')

    resultJson = {
        "requested_at": now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "read_at": now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "value": await calculateProbActivity(hour, date),
        "unit": "activity"
    }
    return resultJson


async def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))


async def IndividualConsumptionInRoomsEmulToDB(column, method_, startDate, startTime, endDate, endTime,
                                               datetime_column, interval, matches):
    df_column = pd.DataFrame(columns=['start_at', "end_at", column, "unit"])
    index = matches.index(column)
    totalConsumptionsTableDb = config["tables"]["emul"][0]["data"]["info"][index]["totalConsumptionsTableDb"]
    totalConsumptionscolumnDb = config["tables"]["emul"][0]["data"]["info"][index]["totalConsumptionscolumnDb"]

    # replace query for totalComsumptions
    query = config.get('queryAnalyticsData')
    query_replaces = config.get('query_replace')
    op = method_ + '(' + totalConsumptionscolumnDb + ')'
    query = await replaceQueryAnalyticsData(query, query_replaces, totalConsumptionsTableDb, datetime_column, op,
                                            interval, startDate,
                                            startTime, endDate, endTime)

    # execute query
    data = await Repo.Repo.executeQuery(query)
    totalConsumptionOfLights = {}
    for date, val in data:
        totalConsumptionOfLights.update({date: val})
    columnSplit = column.split('_')
    entry_data = {}
    for i in range(0, len(config["tables"]["emul"][0]["data"]["info"])):
        info = config["tables"]["emul"][0]["data"]["info"][i]["info"]
        infoSplit = info.split('_')
        if infoSplit[0][0] == columnSplit[0][0] and infoSplit[1] == columnSplit[1]:
            eachTableDb = config["tables"]["emul"][0]["data"]["info"][i]["tableDb"]
            eachColumnDb = config["tables"]["emul"][0]["data"]["info"][i]["columnDb"]
            # replace query
            query = config.get('queryAnalyticsData')
            query_replaces = config.get('query_replace')
            op = method_ + '(' + eachColumnDb + ')'
            query = await replaceQueryAnalyticsData(query, query_replaces, eachTableDb, datetime_column, op, interval,
                                                    startDate,
                                                    startTime, endDate, endTime)
            # execute query
            data = await Repo.Repo.executeQuery(query)
            for date, val in data:
                if date not in entry_data:
                    entry_data.update({date: val})
                else:
                    old_val = entry_data[date]
                    new_val = val + old_val
                    entry_data.update({date: new_val})
    tableDb = config["tables"]["emul"][0]["data"]["info"][index]["tableDb"]
    columnDb = config["tables"]["emul"][0]["data"]["info"][index]["columnDb"]

    # replace query for totalComsumptions
    query = config.get('queryAnalyticsData')
    query_replaces = config.get('query_replace')
    op = method_ + '(' + columnDb + ')'
    query = await replaceQueryAnalyticsData(query, query_replaces, tableDb, datetime_column, op, interval, startDate,
                                            startTime, endDate, endTime)
    # execute query
    data = await Repo.Repo.executeQuery(query)
    date_columns = []
    consumptions = []
    for date, val in data:
        date_columns.append(date)
        consumptions.append(val)
    for date_column, consumptionW in zip(date_columns, consumptions):
        unit = config["tables"]["emul"][0]["data"]["info"][index]["unit"]
        if columnSplit[3] == "consumption":
            end_date_obj = date_column + timedelta(minutes=interval) - timedelta(seconds=1)
            try:
                totalConsumptionOfLightsEntry = totalConsumptionOfLights.get(date_column)
            except:
                totalConsumptionOfLightsEntry = 0
            try:
                sumAllLightsEntry = entry_data.get(date_column)
            except:
                sumAllLightsEntry = 0
            if sumAllLightsEntry is None and totalConsumptionOfLightsEntry is None:
                df_column = pd.concat([df_column, pd.DataFrame(
                    {'start_at': [date_column], "end_at": [end_date_obj], column: None,
                     "unit": unit})], ignore_index=True)
            else:
                if sumAllLightsEntry is None:
                    sumAllLightsEntry = 0
                if totalConsumptionOfLightsEntry is None:
                    totalConsumptionOfLightsEntry = 0
                df_column = pd.concat([df_column, pd.DataFrame(
                {'start_at': [date_column], "end_at": [end_date_obj], column: [round(await divisionOfConsumption(totalConsumptionOfLightsEntry, sumAllLightsEntry) * consumptionW, 2)],
                 "unit": unit})], ignore_index=True)

        elif columnSplit[3] == "state":
            end_date_obj = date_column + timedelta(minutes=interval) - timedelta(seconds=1)
            df_column = pd.concat([df_column, pd.DataFrame(
                {'start_at': [date_column], "end_at": [end_date_obj], column: [consumptionW],
                 "unit": unit})], ignore_index=True)
    if df_column.empty:
        return df_column
    df_column = df_column.sort_values(by='start_at')
    datetimeFormat = '%Y-%m-%d %H:%M:%S'
    start_date_og = datetime.strptime((startDate + " " + startTime + ":00"), datetimeFormat)
    end_date_og = datetime.strptime((endDate + " " + endTime + ":00"), datetimeFormat)
    start_date = datetime.strptime((startDate + " " + "00:00:00"), datetimeFormat)
    end_date = datetime.strptime((endDate + " " + "23:59:00"), datetimeFormat)
    df_column = await check_df_analytics(df_column, start_date, end_date, start_date_og, end_date_og, interval,
                                         column, unit)
    return df_column

async def IndividualConsumptionInRooms(column, matches):
    index = matches.index(column)
    totalConsumptionsTableDb = config["tables"]["emul"][0]["data"]["info"][index]["totalConsumptionsTableDb"]
    totalConsumptionscolumnDb = config["tables"]["emul"][0]["data"]["info"][index]["totalConsumptionscolumnDb"]
    tableDb = config["tables"]["emul"][0]["data"]["info"][index]["tableDb"]
    columnDb = config["tables"]["emul"][0]["data"]["info"][index]["columnDb"]

    # replace query for totalComsumptions
    query = config.get('queryResource')
    query = query.replace('COLUMN', totalConsumptionscolumnDb)
    query = query.replace('TABLE', totalConsumptionsTableDb)
    # execute query
    data = await Repo.Repo.executeQuery(query)
    totalConsumptionOfLights = data[0][1]

    columnSplit = column.split('_')

    allValues = []
    for i in range(0, len(config["tables"]["emul"][0]["data"]["info"])):
        info = config["tables"]["emul"][0]["data"]["info"][i]["info"]
        infoSplit = info.split('_')
        if infoSplit[0][0] == columnSplit[0][0] and infoSplit[1] == columnSplit[1]:
            eachTableDb = config["tables"]["emul"][0]["data"]["info"][i]["tableDb"]
            eachColumnDb = config["tables"]["emul"][0]["data"]["info"][i]["columnDb"]
            # replace query
            query = config.get('queryResource')
            query = query.replace('COLUMN', eachColumnDb)
            query = query.replace('TABLE', eachTableDb)
            # execute query
            data = await Repo.Repo.executeQuery(query)
            allValues.append(data[0][1])

    sumAllLights = sum(allValues)

    tableDb = config["tables"]["emul"][0]["data"]["info"][index]["tableDb"]
    columnDb = config["tables"]["emul"][0]["data"]["info"][index]["columnDb"]

    # replace query for totalComsumptions
    query = config.get('queryResource')
    query = query.replace('COLUMN', columnDb)
    query = query.replace('TABLE', tableDb)
    # execute query
    data = await Repo.Repo.executeQuery(query)
    consumptionW = data[0][1]
    unit = config["tables"]["emul"][0]["data"]["info"][index]["unit"]

    if columnSplit[3] == "consumption":
        resultJson = {
            "requested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "read_at": data[0][0].strftime("%Y-%m-%d %H:%M:%S.%f"),
            "value": round(await divisionOfConsumption(totalConsumptionOfLights, sumAllLights) * consumptionW, 2),
            "unit": unit
        }

        return resultJson

    elif columnSplit[3] == "state":
        resultJson = {
            "requested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "read_at": data[0][0].strftime("%Y-%m-%d %H:%M:%S.%f"),
            "value": consumptionW,
            "unit": unit
        }

        return resultJson


async def divisionOfConsumption(totalConsumption, sum):
    if sum == 0:
        return 0
    return totalConsumption / sum


async def StatesOfAC(column, matches):
    columnSplit = column.split('_')
    query = config.get('queryResource')
    index = matches.index(column)
    tableDb = config["tables"]["emul"][0]["data"]["info"][index]["tableDb"]
    columnDb = config["tables"]["emul"][0]["data"]["info"][index]["columnDb"]

    # replace query
    query = query.replace('COLUMN', columnDb)
    query = query.replace('TABLE', tableDb)
    # execute query
    data = await Repo.Repo.executeQuery(query)
    value = data[0][1]
    result = await verifyStateOfAC(value)
    unit = config["tables"]["emul"][0]["data"]["info"][index]["unit"]

    if columnSplit[2] == "consumption":
        resultJson = {
            "requested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "read_at": data[0][0].strftime("%Y-%m-%d %H:%M:%S.%f"),
            "value": result["consumption_w"],
            "unit": unit
        }

        return resultJson

    elif columnSplit[2] == "state":
        resultJson = {
            "requested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "read_at": data[0][0].strftime("%Y-%m-%d %H:%M:%S.%f"),
            "value": result["state"],
            "unit": unit
        }

        return resultJson


async def verifyStateOfAC(result):
    if result >= 0 and result < 100:
        return {"consumption_w": result, "state": "standby"}
    elif result >= 100:
        return {"consumption_w": result, "state": "on"}


async def validateDates(startDate, startTime, endDate, endTime):
    DATETIMEFORMAT = config.get('DATETIMEFORMAT')
    try:
        start_date = datetime.strptime((startDate + " " + startTime), DATETIMEFORMAT)
    except:
        return False
    try:
        end_date = datetime.strptime((endDate + " " + endTime), DATETIMEFORMAT)
    except:
        return False
    if end_date < start_date:
        return False
    return True


async def getRawDataController(tableName, column, startDate, startTime, endDate, endTime):
    tables = config.get('tables').get('db')
    for t in tables:
        if t.get('name') == tableName:
            tablesWithData = t.get('data')
            break
    infoData = tablesWithData.get('info')
    if column is not None:
        for info in infoData:
            if info.get('info') == column:
                tables = info.get('tables')
                unit = info.get('unit')
                break

        df_column = await getRawDataForColumn(tables, tableName, startDate, startTime, endDate, endTime, column, unit)

        resp = []
        if not df_column.empty:
            df_column['read_at'] = df_column['read_at'].dt.strftime('%Y-%m-%d %H:%M:%S')

            for _, row in df_column.iterrows():
                resp.append(await generate_structureRaw(row))

        return resp
    else:
        result_df = pd.DataFrame()
        columns_df = []
        dataframes = []
        units_ = []
        for info in infoData:
            column = info.get('info')
            tables = info.get('tables')
            unit = info.get('unit')
            units_.append(unit)
            columns_df.append(column)
            df_column = await getRawDataForColumn(tables, tableName, startDate, startTime, endDate, endTime, column,
                                                  unit)
            df_column = df_column.rename(columns={"unit": "unit" + column})
            dataframes.append(df_column)
        valueToSubstitute = 0
        empty_df = []
        for idx, df in enumerate(dataframes):
            valueToSubstitute = df_column['read_at'][0].microsecond
            break

        for idx, df in enumerate(dataframes):
            if df.empty:
                columns_list = df.columns.tolist()
                columns_list.remove('read_at')
                for c in columns_list:
                    empty_df.append(c)
            else:
                df['read_at'] = df['read_at'].apply(lambda x: x.replace(microsecond=valueToSubstitute))
                df['read_at'] = pd.to_datetime(df['read_at'])
                df['read_at'] = df['read_at'].dt.round('1s')
                if idx == 0:
                    if result_df.empty:
                        result_df = df.copy()
                else:
                    if result_df.empty:
                        result_df = df.copy()
                    else:
                        result_df = pd.merge(result_df, df, how='outer', on=['read_at'])
                        result_df = result_df.sort_values(by='read_at')
        if len(dataframes) == len(empty_df):
            return []
        for col in empty_df:
            result_df[col] = np.nan
        result_df['read_at'] = result_df['read_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        resp = []
        for _, row in result_df.iterrows():
            row_resp = {}
            for c, u in zip(columns_df, units_):
                val_ = row.loc[c]
                unit_ = row.loc['unit' + c]
                try:
                    if np.isnan(unit_):
                        unit_ = u
                except:
                    pass
                row_c = [row[0], val_, unit_]
                res_column = await generate_structureRaw(row_c)
                row_resp.update({c: res_column})
            resp.append(row_resp)
        return resp


async def getRawDataForColumn(tables, tableName, startDate, startTime, endDate, endTime, columnRequest, unit):
    df_column = pd.DataFrame(columns=['read_at', columnRequest, "unit"])

    for table in tables:
        try:
            listSum = []
            sumList = table.get('sum')
            for row in sumList:
                query = config.get('queryRawData')
                query_replaces = config.get('queryRawData_replace')
                targetColumn = row.get('column')
                tableName = row.get('name')
                datetime_column = row.get('datetime_column')
                try:
                    multiplier = row.get('multiplier')
                except:
                    multiplier = 1
                query = await replaceQueryRawData(query, query_replaces, tableName, targetColumn, datetime_column,
                                                  startDate, startTime, endDate, endTime)
                data = await Repo.Repo.executeQuery(query)
                listSum.append(data)

            listWithoutNull = [lista for lista in listSum if lista]

            for i in range(len(listWithoutNull[0])):
                soma = 0
                for lista in listWithoutNull:
                    soma += lista[i][1]

                df_column = pd.concat([df_column, pd.DataFrame(
                    {'read_at': [listWithoutNull[0][i][0]], columnRequest: [soma * multiplier], "unit": unit})],
                                      ignore_index=True)

        except:
            query = config.get('queryRawData')
            query_replaces = config.get('queryRawData_replace')
            targetColumn = table.get('column')
            tableName = table.get('name')
            datetime_column = table.get('datetime_column')
            try:
                multiplier = table.get('multiplier')
            except:
                multiplier = 1

            query = await replaceQueryRawData(query, query_replaces, tableName, targetColumn, datetime_column,
                                              startDate, startTime, endDate, endTime)
            data = await Repo.Repo.executeQuery(query)

            for date_column, val in data:
                df_column = pd.concat([df_column, pd.DataFrame(
                    {'read_at': [date_column], columnRequest: [val * multiplier], "unit": unit})], ignore_index=True)

    if df_column.empty:
        return df_column
        # datetimeFormat='%Y-%m-%d %H:%M:%S'
        # start_date = datetime.strptime((startDate+" "+startTime+":00"), datetimeFormat)
        # end_date = datetime.strptime((endDate+" "+endTime+":00"), datetimeFormat)
        # # Define the frequency in seconds
        # date_range_init = [start_date + i * timedelta(seconds=10) for i in range(int(((end_date - start_date).total_seconds() / 60) / (10/60)) + 1)]
        # date_range_final = [date + timedelta(seconds=10) - timedelta(seconds=1) for date in date_range_init]

        # # Create a dataframe using the date range
        # df_column = pd.DataFrame({'read_at': date_range_init,columnRequest:0,"unit":unit})

    df_column = df_column.sort_values(by='read_at')
    return df_column


async def replaceQueryRawData(query, query_replaces, tableName, targetColumn, datetime_column, startDate, startTime,
                              endDate, endTime):
    for k, v in query_replaces.items():
        if v == 'datetime_column':
            query = query.replace(k, datetime_column)
        elif v == 'tableName':
            query = query.replace(k, tableName)
        elif v == 'targetColumn':
            query = query.replace(k, targetColumn)
        elif v == 'startDate':
            query = query.replace(k, startDate)
        elif v == 'startTime':
            query = query.replace(k, startTime)
        elif v == 'endDate':
            query = query.replace(k, endDate)
        elif v == 'endTime':
            query = query.replace(k, endTime)

    return query


async def generate_structureRaw(row):
    return {
        "read_at": row[0],
        "value": row[1],
        "unit": row[2],
    }


async def getAnalyticDataDB(method, interval, tableName, column, startDate, startTime, endDate, endTime):
    # start all with none to verify the variables in "if"
    if column is not None:
        valid_columns = config.get('json_path_columns')
        jsonpath_expression = parse(valid_columns)
        matches = [match.value for match in jsonpath_expression.find(config)]
        if column not in matches:
            result = {"Error": "column name is not valid!"}
            return jsonify(result)

    results = await getAnalyticDataController(tableName, column, method, interval, startDate, startTime, endDate,
                                              endTime)

    return jsonify(results)


async def getAnalyticDataEmul(method, interval, tableName, column, startDate, startTime, endDate, endTime):
    # start all with none to verify the variables in "if"
    if column is not None:
        valid_columns = config.get('json_path_columns_emul')
        jsonpath_expression = parse(valid_columns)
        matches = [match.value for match in jsonpath_expression.find(config)]
        if column not in matches:
            result = {"Error": "column name is not valid!"}
            return jsonify(result)
    results = await getAnalyticDataControllerEmul(tableName, column, method, interval, startDate, startTime, endDate,
                                                  endTime, matches)

    return jsonify(results)


async def getAnalyticDataController(tableName, column, method, interval, startDate, startTime, endDate, endTime):
    tables = config.get('tables').get('db')
    for t in tables:
        if t.get('name') == tableName:
            tablesWithData = t.get('data')
            break
    infoData = tablesWithData.get('info')
    if column is not None:
        for info in infoData:
            if info.get('info') == column:
                tables = info.get('tables')
                unit = info.get('unit')
                break
        df_column = await getAnalyticsDataForColumn(tables, tableName, method, interval, startDate, startTime, endDate,
                                                    endTime, column, unit)
        resp = []
        if not df_column.empty:
            df_column['start_at'] = df_column['start_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_column['end_at'] = df_column['end_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_column = df_column.drop_duplicates(subset=['start_at'])
            df_column = await dataset.fix_dataset(df_column)
            for _, row in df_column.iterrows():
                resp.append(await generate_structure(row))
        return resp
    else:
        result_df = pd.DataFrame()
        columns_df = []
        dataframes = []
        units_ = []
        for info in infoData:
            column = info.get('info')
            tables = info.get('tables')
            unit = info.get('unit')
            units_.append(unit)
            columns_df.append(column)
            df_column = await getAnalyticsDataForColumn(tables, tableName, method, interval, startDate, startTime,
                                                        endDate, endTime, column, unit)
            df_column = df_column.rename(columns={"unit": "unit" + column})
            dataframes.append(df_column)
        empty_df = []
        for idx, df in enumerate(dataframes):
            if df.empty:
                columns_list = df.columns.tolist()
                columns_list.remove('start_at')
                columns_list.remove("end_at")
                for c in columns_list:
                    empty_df.append(c)
            else:
                if idx == 0:
                    # For the first DataFrame, directly set it as the result_df
                    result_df = df.copy()
                else:
                    if result_df.empty:
                        result_df = df.copy()
                    # For subsequent DataFrames, perform an inner join using the common column
                    else:
                        result_df = pd.merge(result_df, df, how='inner', on=['start_at', 'end_at'])
        if len(dataframes) == len(empty_df):
            return []
        for col in empty_df:
            result_df[col] = np.nan
        result_df['start_at'] = result_df['start_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        result_df['end_at'] = result_df['end_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        result_df = result_df.drop_duplicates(subset=['start_at'])
        result_df = await dataset.fix_dataset(result_df)
        resp = []
        for _, row in result_df.iterrows():
            row_resp = {}
            for c, u in zip(columns_df, units_):
                val_ = row.loc[c]
                unit_ = row.loc['unit' + c]
                try:
                    if np.isnan(unit_):
                        unit_ = u
                except:
                    pass
                row_c = [row[0], row[1], val_, unit_]
                res_column = await generate_structure(row_c)
                row_resp.update({c: res_column})
            resp.append(row_resp)
        return resp


async def getAnalyticDataControllerEmul(tableName, column, method, interval, startDate, startTime, endDate, endTime,
                                        matches):
    tables = config.get('tables').get('emul')
    for t in tables:
        if t.get('name') == tableName:
            method_ = config.get('methods').get(method)
            datetime_column = t.get('datetime_column')
            break
    index = matches.index(column)
    try:
        totalConsumptionsTableDb = config["tables"]["emul"][0]["data"]["info"][index]["totalConsumptionsTableDb"]
    except:
        totalConsumptionsTableDb = None
    if totalConsumptionsTableDb is not None:
        df_column = await IndividualConsumptionInRoomsEmulToDB(column, method_, startDate, startTime,
                                                               endDate,
                                                               endTime, datetime_column, interval, matches)
    else:
        table_ = config["tables"]["emul"][0]["data"]["info"][index]["tableDb"]
        column_ = config["tables"]["emul"][0]["data"]["info"][index]["columnDb"]
        unit = config["tables"]["emul"][0]["data"]["info"][index]["unit"]
        df_column = await getAnalyticsDataForColumnEmul(table_, column_, method_, interval, startDate, startTime,
                                                        endDate, endTime,
                                                        column, unit, datetime_column)

    resp = []
    if not df_column.empty:
        df_column['start_at'] = df_column['start_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_column['end_at'] = df_column['end_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_column = df_column.drop_duplicates(subset=['start_at'])
        df_column = await dataset.fix_dataset(df_column)
        for _, row in df_column.iterrows():
            resp.append(await generate_structure(row))
    return resp


async def getAnalyticsDataForColumnEmul(tableName, targetColumn, method, interval, startDate, startTime, endDate,
                                        endTime,
                                        columnRequest, unit, datetime_column):
    df_column = pd.DataFrame(columns=['start_at', "end_at", columnRequest, "unit"])
    query = config.get('queryAnalyticsData')
    query_replaces = config.get('query_replace')
    op = method + '(' + targetColumn + ')'
    query = await replaceQueryAnalyticsData(query, query_replaces, tableName, datetime_column, op, interval,
                                            startDate, startTime, endDate, endTime)
    data = await Repo.Repo.executeQuery(query)
    for date_column, val in data:
        end_date_obj = date_column + timedelta(minutes=interval) - timedelta(seconds=1)
        split_column_requested_name = columnRequest.split ('_')
        if unit == "" and split_column_requested_name[1] =='ac' and split_column_requested_name[2]=='state':
            if 0 <= val < 100:
                val = "standby"
            elif val >= 100:
                val = "on"
        df_column = pd.concat([df_column, pd.DataFrame(
            {'start_at': [date_column], "end_at": [end_date_obj], columnRequest: [val],
             "unit": unit})], ignore_index=True)
    if df_column.empty:
        return df_column
    df_column = df_column.sort_values(by='start_at')
    datetimeFormat = '%Y-%m-%d %H:%M:%S'
    start_date_og = datetime.strptime((startDate + " " + startTime + ":00"), datetimeFormat)
    end_date_og = datetime.strptime((endDate + " " + endTime + ":00"), datetimeFormat)
    start_date = datetime.strptime((startDate + " " + "00:00:00"), datetimeFormat)
    end_date = datetime.strptime((endDate + " " + "23:59:00"), datetimeFormat)
    df_column = await check_df_analytics(df_column, start_date, end_date, start_date_og, end_date_og, interval,
                                         columnRequest, unit)
    return df_column


async def getAnalyticsDataForColumn(tables, tableName, method, interval, startDate, startTime, endDate, endTime,
                                    columnRequest, unit):
    df_column = pd.DataFrame(columns=['start_at', "end_at", columnRequest, "unit"])
    for table in tables:
        query = config.get('queryAnalyticsData')
        query_replaces = config.get('query_replace')
        currentKey = next(iter(table.keys()))
        if currentKey == 'name':
            tableName = table.get('name')
            method_ = config.get('methods').get(method)
            targetColumn = table.get('column')
            datetime_column = table.get('datetime_column')
            try:
                multiplier = table.get('multiplier')
            except:
                multiplier = 1
            if unit == 'binary':
                op = targetColumn
                query = config.get('queryAnalyticsData_binary')
            else:
                op = method_ + '(' + targetColumn + ')'
                query = config.get('queryAnalyticsData')
            query = await replaceQueryAnalyticsData(query, query_replaces, tableName, datetime_column, op, interval,
                                                    startDate, startTime, endDate, endTime)
            data = await Repo.Repo.executeQuery(query)
            for date_column, val in data:
                end_date_obj = date_column + timedelta(minutes=interval) - timedelta(seconds=1)
                df_column = pd.concat([df_column, pd.DataFrame(
                    {'start_at': [date_column], "end_at": [end_date_obj], columnRequest: [val * multiplier],
                     "unit": unit})], ignore_index=True)

        else:
            dataframes = []
            method_agg_column = currentKey
            for table_agg in table.get(currentKey):
                tableName = table_agg.get('name')
                method_ = config.get('methods').get(method)
                targetColumn = table_agg.get('column')
                datetime_column = table_agg.get('datetime_column')
                try:
                    multiplier = table_agg.get('multiplier')
                except:
                    multiplier = 1
                if unit == 'binary':
                    op = targetColumn
                    query = config.get('queryAnalyticsData_binary')
                else:
                    op = method_ + '(' + targetColumn + ')'
                    query = config.get('queryAnalyticsData')
                query = await replaceQueryAnalyticsData(query, query_replaces, tableName, datetime_column, op, interval,
                                                        startDate, startTime, endDate, endTime)
                data = await Repo.Repo.executeQuery(query)
                df = pd.DataFrame(columns=['start_at', "end_at", columnRequest, "unit"])
                if not len(data) == 0:
                    for date_column, val in data:
                        end_date_obj = date_column + timedelta(minutes=interval) - timedelta(seconds=1)
                        df = pd.concat([df, pd.DataFrame(
                            {'start_at': [date_column], "end_at": [end_date_obj], columnRequest: [val * multiplier],
                             "unit": unit})], ignore_index=True)
                    dataframes.append(df)
            for idx, df in enumerate(dataframes):
                if idx == 0:
                    # For the first DataFrame, directly set it as the df_column
                    df_column = df.copy()
                else:
                    combined_df = pd.concat([df_column, df], ignore_index=True)
                    # Convert columnRequest column to numeric (in case it's not already)
                    combined_df[columnRequest] = pd.to_numeric(combined_df[columnRequest], errors='coerce')
                    # print(combined_df)
                    # Group by 'Date' and calculate the sum/average
                    if method_agg_column == 'sum':
                        df_column = combined_df.groupby(['start_at', 'end_at', 'unit'])[
                            columnRequest].sum().reset_index()
                    elif method_agg_column == 'average':
                        df_column = combined_df.groupby(['start_at', 'end_at', 'unit'])[
                            columnRequest].mean().reset_index()
    if df_column.empty:
        return df_column
    df_column = df_column.sort_values(by='start_at')
    datetimeFormat = '%Y-%m-%d %H:%M:%S'
    start_date_og = datetime.strptime((startDate + " " + startTime + ":00"), datetimeFormat)
    end_date_og = datetime.strptime((endDate + " " + endTime + ":00"), datetimeFormat)
    start_date = datetime.strptime((startDate + " " + "00:00:00"), datetimeFormat)
    end_date = datetime.strptime((endDate + " " + "23:59:00"), datetimeFormat)
    df_column = await check_df_analytics(df_column, start_date, end_date, start_date_og, end_date_og, interval,
                                         columnRequest, unit)
    return df_column


async def replaceQueryAnalyticsData(query, query_replaces, tableName, datetime_column, op, interval, startDate,
                                    startTime, endDate, endTime):
    for k, v in query_replaces.items():
        if v == 'interval':
            query = query.replace(k, str(interval))
        elif v == 'datetime_column':
            query = query.replace(k, datetime_column)
        elif v == 'operation':
            query = query.replace(k, op)
        elif v == 'tableName':
            query = query.replace(k, tableName)
        elif v == 'startDate':
            query = query.replace(k, startDate)
        elif v == 'startTime':
            query = query.replace(k, startTime)
        elif v == 'endDate':
            query = query.replace(k, endDate)
        elif v == 'endTime':
            query = query.replace(k, endTime)

    return query


async def check_df_analytics(df, start_date, end_date, start_date_og, end_date_og, interval, column, unit):
    date_range_init = [start_date + i * timedelta(minutes=interval) for i in
                       range(int(((end_date - start_date).total_seconds() / 60) / interval) + 1)]
    date_range_final = [date + timedelta(minutes=interval) - timedelta(seconds=1) for date in date_range_init]

    # Create a dataframe using the date range
    df2 = pd.DataFrame({'start_at': date_range_init, "end_at": date_range_final, column: np.nan, "unit": unit})
    df2 = df2[(df2['start_at'] >= start_date_og) & (df2['end_at'] <= end_date_og)]
    df3 = pd.merge(df2, df, on=['start_at', 'end_at'], how='outer')
    df3 = df3.sort_values('start_at')
    df3 = df3.reset_index(drop=True)
    unit_x_merge = 'unit_x'
    unit_y_merge = 'unit_y'
    df3['unit'] = unit
    df3 = df3.drop(columns=unit_x_merge)
    df3 = df3.drop(columns=unit_y_merge)
    x_merge = column + '_x'
    y_merge = column + '_y'
    df3[column] = df3[y_merge]
    df3 = df3.drop(columns=x_merge)
    df3 = df3.drop(columns=y_merge)
    # posicionamento
    df3 = df3.drop(columns='unit')
    df3['unit'] = unit
    # Fill missing values in the 'column' column with NaN
    df3[column].fillna(np.nan, inplace=True)
    if unit == '%':
        df3[column] = await apply_cap_value(df3[column])

    return df3


async def apply_cap_value(column):
    new_column = []
    for value in column:
        new_value = await cap_value(value)
        new_column.append(new_value)
    return new_column


async def generate_structure(row):
    if isinstance(row, list):
        return {
            "start_at": row[0],
            "end_at": row[1],
            "value": row[2],
            "unit": row[3]
        }
    return {
        "start_at": row.iloc[0],
        "end_at": row.iloc[1],
        "value": row.iloc[2],
        "unit": row.iloc[3]
    }


async def ActivityCalucation(request, time, column, startDate, endDate, startTime, endTime, matches):
    if startDate == None:
        result = {"Error": "startDate is not entered and is required"}
        return jsonify(result)

    if endDate == None:
        result = {"Error": "endDate is not entered and is required"}
        return jsonify(result)

    if startTime == None:
        result = {"Error": "startTime is not entered and is required"}
        return jsonify(result)

    if endTime == None:
        result = {"Error": "endTime is not entered and is required"}
        return jsonify(result)

    time = int(time)
    initialDate = await formatDate(startDate, startTime)
    finishDate = await formatDate(endDate, endTime)

    if await is_valid_interval(initialDate, finishDate, time) == False:
        result = {"Error": "The interval is not valid with the dates entered"}
        return jsonify(result)
    if initialDate > finishDate:
        result = {"Error": "Invalid Dates. End Date must be higher or equal to Start Date."}
        return jsonify(result)

    # elif initialDate > datetime.now() or finishDate > datetime.now():
    #     date_range_init = [initialDate + i * timedelta(minutes=time) for i in
    #                        range(int(((finishDate - initialDate).total_seconds() / 60) / time) + 1)]
    #     for date_ in reversed(date_range_init):
    #         if date_ > datetime.now():
    #
    #     result = {"Error": "Invalid Dates. Your dates are higher than today date."}
    #     return jsonify(result)

    elif initialDate == finishDate:
        startDate = await formatDate(startDate, startTime)

        listOfDateTimes = []
        listOfDateTimes.append((startDate.strftime('%Y-%m-%d'), startDate.strftime('%H:%M')))

        finalJson = await getResultJsonForActivity(listOfDateTimes, time, column, matches)

        return jsonify(finalJson)

    else:

        minuteOfStartTime = startTime.split(':')[1]
        hourOfStartTime = startTime.split(':')[0]
        minuteOfEndTime = endTime.split(':')[1]

        if time == 30:
            waitingTimeList = np.arange(00, 60, 15).tolist()
        else:
            waitingTimeList = np.arange(00, 60, time).tolist()

        if time != 60 and (
                (int(minuteOfEndTime) not in waitingTimeList and int(minuteOfStartTime) not in waitingTimeList) or int(
                minuteOfStartTime) not in waitingTimeList):
            next15Mins = await calculateAMinuteForNext15Mins(minuteOfStartTime)

            startTime = await verifyStartTime(startTime, next15Mins, hourOfStartTime, minuteOfStartTime)

            startDate = await formatDate(startDate, startTime)
            endDate = await formatDate(endDate, endTime)
            listOfDateTimes = []
            async for dt in datetime_range(startDate, endDate, timedelta(minutes=time)):
                listOfDateTimes.append((dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M')))

            listOfDateTimes.append((endDate.strftime('%Y-%m-%d'), endDate.strftime('%H:%M')))

        else:
            startDate = await formatDate(startDate, startTime)
            endDate = await formatDate(endDate, endTime)
            listOfDateTimes = []

            async for dt in datetime_range(startDate + relativedelta(minutes=time), endDate, timedelta(minutes=time)):
                listOfDateTimes.append((dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M')))

            listOfDateTimes.append((endDate.strftime('%Y-%m-%d'), endDate.strftime('%H:%M')))

        finalJson = await getResultJsonForActivity(listOfDateTimes, time, column, matches)

    return jsonify(finalJson)


async def formatDate(date, time):
    finalDate = str(date) + " " + str(time)

    finalDate = datetime.strptime(finalDate, '%Y-%m-%d %H:%M')
    return finalDate


async def is_valid_interval(start_date, end_date, interval_minutes):
    # Calculate the difference in minutes between the two dates
    time_difference = (end_date - start_date).total_seconds() / 60
    if start_date == end_date:
        return True

    if time_difference - interval_minutes >= 0:
        return True
    else:
        return False


async def calculateAMinuteForNext15Mins(timeByMinute):
    if timeByMinute >= "45" and timeByMinute <= "59":
        timeByMinute = "00"

    elif timeByMinute >= "00" and timeByMinute < "15":
        timeByMinute = "15"

    elif timeByMinute >= "15" and timeByMinute < "30":
        timeByMinute = "30"

    elif timeByMinute >= "30" and timeByMinute < "45":
        timeByMinute = "45"

    return timeByMinute


async def verifyStartTime(startTime, next15Mins, hourOfStartTime, minuteOfStartTime):
    if next15Mins == "00":
        if hourOfStartTime == "23":
            nexthourOfStartTime = 00
        else:
            nexthourOfStartTime = int(hourOfStartTime) + 1

        startTime = startTime.replace(hourOfStartTime, str(nexthourOfStartTime))

    startTime = startTime.replace(minuteOfStartTime, next15Mins)
    return startTime


async def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


async def getResultJsonForActivity(listOfDateTimes, time, column, matches):
    try:
        finalJson = []
        penultimateDate = listOfDateTimes[len(listOfDateTimes) - 2]
        penultimateDate = await formatDate(penultimateDate[0], penultimateDate[1])
        count = 0
        index = matches.index(column)
        unit = config["tables"]["emul"][0]["data"]["info"][index]["unit"]
        if len(listOfDateTimes) == 1:
            tempDate = await formatDate(listOfDateTimes[0][0], listOfDateTimes[0][1])
            intervalDate = tempDate - relativedelta(minutes=1)
            dateOfIntervalDate = intervalDate.strftime('%Y-%m-%d')
            hourOfIntervaleDate = (intervalDate.strftime('%H:%M')).split(":")[0]
            resultJson = [{
                    "start_at": (tempDate - timedelta(minutes=time)).isoformat(sep=' '),
                    "end_at": (tempDate - timedelta(seconds=1)).isoformat(sep=' '),
                    "value": await calculateProbActivity(hourOfIntervaleDate, dateOfIntervalDate),
                    "unit": unit
                }]
            return resultJson
        else:
            for dateTimes in listOfDateTimes:

                tempDate = await formatDate(dateTimes[0], dateTimes[1])
                intervalDate = tempDate - relativedelta(minutes=1)
                dateOfIntervalDate = intervalDate.strftime('%Y-%m-%d')
                hourOfIntervaleDate = (intervalDate.strftime('%H:%M')).split(":")[0]

                if count == len(listOfDateTimes) - 1:
                    finalJson.append(await verifyActivityFinal(hourOfIntervaleDate, tempDate, time, dateOfIntervalDate,
                                                               penultimateDate, unit))
                else:
                    finalJson.append(
                        await verifyActivity(hourOfIntervaleDate, tempDate, time, dateOfIntervalDate, unit))
                count = count + 1

            return finalJson
    except Exception as e:
        print("erro", e)


async def verifyActivityFinal(hourOfIntervaleDate, tempDate, time, dateOfIntervalDate, penultimateDate, unit):
    
    resultJson= { 
                "start_at":(penultimateDate).isoformat(sep=' '),
                "end_at":(tempDate - timedelta(seconds=1)).isoformat(sep=' '),
                "value":await calculateProbActivity(hourOfIntervaleDate, dateOfIntervalDate),
                "unit":unit
            }

    return resultJson    


async def verifyActivity(hourOfIntervaleDate, tempDate, time, dateOfIntervalDate, unit):
    resultJson= {
                "start_at":(tempDate - timedelta(minutes=time)).isoformat(sep=' '),
                "end_at":(tempDate - timedelta(seconds=1)).isoformat(sep=' '),
                "value":await calculateProbActivity(hourOfIntervaleDate, dateOfIntervalDate),
                "unit":unit
            }

    return resultJson
