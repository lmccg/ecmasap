from quart import Quart, request, render_template, jsonify
from quart_cors import cors
import os, json
from controller import *
from jsonpath_ng.ext import parse 
from datetime import datetime

DB_PASSWORD = os.getenv('DB_PASSWORD')
app = Quart(__name__)
app = cors(app, allow_origin="*")
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET'])
async def startup():
    try:
        return await render_template('home.html')
    except Exception as e:
        return await internal_server_error(e)

@app.route('/documentation/', methods=['GET'])
async def documentation():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE_DIR, 'config.json'),encoding='utf-8') as f:
            config = json.load(f)
        return await render_template('documentation.html',  config= config)
    except Exception as e:
        return await internal_server_error(e)

@app.route('/api/resources', methods=['GET'])
async def getAllTables():
    try:
        #make an request
        result = await getAllTablesController() 
        return jsonify(result) 
    except Exception as e: 
        return await internal_server_error(e)

@app.route('/api/resource/<table>', methods=['GET'])
async def getRtDataFromATable(table):
    try:
        tables = config.get('json_path_tables_db')
        jsonpath_expr = parse(tables)
        matchesTable = [match.value for match in jsonpath_expr.find(config)]
        matchesTable.append('emul')
        # Verify if the table is in config.json
        if table not in matchesTable:
            result = {"Error": "Table name is not valid!"}
            return jsonify(result), 500  

        # Create a formatted JSON
        resultJson = []
        
        if table != 'emul':
            # Get index of table in config.json
            idx = matchesTable.index(table)

            # Get jsonFinal
            listOfcolumnsInTable = config["tables"]["db"][idx]["data"]["info"]
            requestedAt = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            resultJson = await formatJsonForATableInDb(listOfcolumnsInTable, idx, requestedAt, resultJson)
        else:
            listOfcolumnsInTable = config["tables"]["emul"][0]["data"]["info"]
            resultJson = await formatJsonForATableInEmul(listOfcolumnsInTable, resultJson)  

        return jsonify(resultJson)
    except Exception as e:
        return await internal_server_error(e)
    
@app.route('/api/resource/<table>/<column>', methods=['GET'])
async def getResouceData(table, column):
    try:
        tables_db=config.get('json_path_tables_db')
        jsonpath_expression = parse(tables_db)
        matches_db=[match.value for match in jsonpath_expression.find(config)]
        tables_emul=config.get('json_path_tables_emul')
        jsonpath_expression = parse(tables_emul)
        matches_emul=[match.value for match in jsonpath_expression.find(config)]
        if table in matches_db:
            return await getRtDataFromAColumn(table,matches_db,column)
        elif table in matches_emul:
            info=config.get('tables').get('emul')
            for i in info:
                if i.get('name')==table:
                    info_table=i.get('data').get('info')
            return await getRtDataEmul(column,info_table)
        else:
            result = {"Error": "table name is not valid!"}   
            return jsonify(result), 500  
    except Exception as e:
        return await internal_server_error(e)    

@app.route('/api/raw/<tableName>/<column>/', methods=['GET'])
@app.route('/api/raw/<tableName>/', methods=['GET'])
async def getRawData(tableName,column=None):
    try:
        startDate = None
        endDate = None
        startTime = None
        endTime = None
        startDate = request.args.get('startDate')
        endDate = request.args.get('endDate')
        startTime = request.args.get('startTime')
        endTime = request.args.get('endTime') 

        if startDate is None or endDate is None:
            result = {"Error": "set the start and end dates!"}   
            return jsonify(result), 500  
        if startTime is None:
            startTime=config.get('start_time')
        if endTime is None:
            endTime=config.get('end_time')
        valid= await validateDates(startDate, startTime, endDate, endTime)  
        if valid==False:
            result = {"Error": "dates are not valid!"}   
            return jsonify(result), 500  
        tables=config.get('json_path_tables_db')
        jsonpath_expression = parse(tables)
        matches=[match.value for match in jsonpath_expression.find(config)]
        if tableName not in matches:
            result = {"Error": "table name is not valid!"}   
            return jsonify(result), 500  
        if column is not None:
            valid_columns=config.get('json_path_columns')
            jsonpath_expression = parse(valid_columns)
            matches=[match.value for match in jsonpath_expression.find(config)]
            if column not in matches:
                result = {"Error": "column name is not valid!"}   
                return jsonify(result), 500  
            
        dataframe=await getRawDataController(tableName, column, startDate, startTime,endDate, endTime)

        return jsonify(dataframe)
    except Exception as e:
        return await internal_server_error(e) 

@app.route('/api/analytics/<method>/<interval>min/<tableName>/<column>/', methods=['GET'])
@app.route('/api/analytics/<method>/<interval>min/<tableName>/', methods=['GET'])
async def getAnalyticData(method, interval, tableName,column=None):
    print('here')
    try:
        startDate = None
        endDate = None
        startTime = None
        endTime = None
        startDate = request.args.get('startDate')
        endDate = request.args.get('endDate')
        startTime = request.args.get('startTime')
        endTime = request.args.get('endTime') 

        if startDate is None or endDate is None:
            result = {"Error": "set the start and end dates!"}   
            return jsonify(result), 500  
        if startTime is None:
            startTime=config.get('start_time')
        if endTime is None:
            endTime=config.get('end_time')
        interval=int(interval) 
        valid= await validateDates(startDate, startTime, endDate, endTime)  
        if valid==False:
            result = {"Error": "dates are not valid!"}   
            return jsonify(result), 500  
        valid_methods=list(config.get('methods').keys())
        if method not in valid_methods:
            result = {"Error": "method is not valid!"}   
            return jsonify(result), 500  
        valid_intervals=config.get('intervals')
        if interval not in valid_intervals:
            result = {"Error": "interval is not valid!"}   
            return jsonify(result), 500  
        
        tables_db=config.get('json_path_tables_db')
        jsonpath_expression = parse(tables_db)
        matches_db=[match.value for match in jsonpath_expression.find(config)]
        tables_emul=config.get('json_path_tables_emul')
        jsonpath_expression = parse(tables_emul)
        matches_emul=[match.value for match in jsonpath_expression.find(config)]
        if tableName in matches_db:
            return await getAnalyticDataDB(method, interval, tableName,column,startDate, startTime,endDate, endTime)
        elif tableName in matches_emul:
            valid_columns=config.get('json_path_columns_emul')
            jsonpath_expression = parse(valid_columns)
            matches=[match.value for match in jsonpath_expression.find(config)]
            if column is not None:
                if column not in matches:
                    print('not in matches')
                    result = {"Error": "column name is not valid!"}   
                    return jsonify(result), 500  
            unit_column = config.get('json_path_column_unit')
            unit_column = unit_column.replace('COLUMN', str(column))
            jsonpath_expression_unit = parse(unit_column)
            matches_unit = [match.value for match in jsonpath_expression_unit.find(config)]
            unit_column = matches_unit[0]
            if unit_column == 'activity':
                return await ActivityCalucation(request, interval, column,startDate,endDate,startTime,endTime, matches)
            else:
                return await getAnalyticDataEmul(method, interval, tableName,column,startDate, startTime,endDate, endTime)
        else:
            result = {"Error": "table name is not valid!"}   
            return jsonify(result), 500  
    except Exception as e:
        return await internal_server_error(e)     

if __name__ == '__main__':
    app.run(debug=True)