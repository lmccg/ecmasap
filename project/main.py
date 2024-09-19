# from quart import Quart, jsonify, request
# from quart_cors import cors
# from ecmasap.project.agents.utils_package import utils, repository, globals
# import json
#
# app = Quart(__name__, template_folder='templates')
# app = cors(app, allow_origin="*")
#
#
# @app.route('/ping/', methods=['GET'])
# async def get_pong_view():
#     with open('./config_settings.json') as f:
#         config_settings = json.load(f)
#     if not globals.get_app_running():
#         with open('./responses.json') as f:
#             responses = json.load(f)
#         return jsonify({'Info': responses['app_not_running']
#                         }), config_settings.get('errors').get('SERVER_ERROR')
#     return jsonify('pong', 200)
#
#
# @app.route('/stop/', methods=['GET'])
# async def stop_app():
#     if globals.get_app_running():
#         globals.change_app_running(False)
#         # await threadTraining.stop_training()
#         return jsonify({'info': 'Request to stop the app accepted! As soon as possible the app will be stopped.'}), 200
#     return jsonify({'info': 'The app already has a request to stop!'
#                     }), 404
#
#
# @app.route('/restart/', methods=['GET'])
# async def restart_app():
#     if globals.get_training_ongoing() and not globals.get_app_running():
#         return jsonify({'info': 'The app is still training some models, with a request to stop the app yet to perform.'
#                                 'Please try again later!'
#                         }), 404
#     elif globals.get_app_running():
#         return jsonify({'info': 'The app is already running!'
#                         }), 404
#     globals.reset_shutdown_event()
#     training_ongoing = globals.get_training_ongoing()
#     globals.change_app_running(True)
#     if not training_ongoing:
#         await utils.initialize_app()
#     return jsonify({'info': 'App restarted'}), 200
#
#
# @app.before_serving
# async def start_app():
#     repository.create_database_if_not_exists()
#     repository.initialize_database()
#     print(f"[{str(utils.timestamp_with_time_zone())}] [ecmasap] #################STARTED#######################")
#     training_ongoing = globals.get_training_ongoing()
#     globals.change_app_running(True)
#     if not training_ongoing:
#         await utils.initialize_app()
#
#
# @app.route('/predict/', methods=['GET'])
# async def get_predictions_view():
#     if request.method == 'GET':
#         try:
#             # await globals.increment_requests()
#             # async def after_response():
#             #     await globals.decrement_requests()
#             with open('../utils_package/config_settings.json') as f:
#                 config_settings = json.load(f)
#             training_ongoing = globals.get_training_ongoing()
#             if training_ongoing:
#                 with open('./responses.json') as f:
#                     responses = json.load(f)
#                 # asyncio.create_task(after_response())
#                 return {'Info':
#                         responses['info_train_predict']
#                         }, config_settings.get('errors').get('EMPTY_ERROR')
#             if not globals.get_app_running():
#                 with open('../utils_package/responses.json') as f:
#                     responses = json.load(f)
#                 # asyncio.create_task(after_response())
#                 return {'Error':
#                         responses['app_not_running']
#                         }, config_settings.get('errors').get('SERVER_ERROR')
#             try:
#                 data = await request.json
#                 if data is None:
#                     data_request = await request.form
#                     data = data_request.copy().to_dict()
#                 if isinstance(data, str):
#                     data = json.loads(data)
#             except Exception as ex:
#                 data = request.form.copy().to_dict()
#             output, status_code = await utils.process_request(data, 'predict')
#             if status_code == config_settings.get('errors').get('SERVER_ERROR') or status_code == config_settings.get(
#                     'errors').get('INPUT_ERROR') or status_code == config_settings.get('errors').get('FILE_ERROR'):
#                 # asyncio.create_task(after_response())
#                 return output, status_code
#             # asyncio.create_task(after_response())
#             return output, status_code
#         except Exception as ex:
#             e = str(ex)
#             # try:
#             #     asyncio.create_task(after_response())
#             # except:
#             #     pass
#             return jsonify({'error': e}), config_settings.get('errors').get('INPUT_ERROR')
#
#
# @app.route('/train/', methods=['POST'])
# async def get_train_view():
#     if request.method == 'GET':
#         try:
#             # await globals.increment_requests()
#             # async def after_response():
#             #     await globals.decrement_requests()
#             with open('../utils_package/config_settings.json') as f:
#                 config_settings = json.load(f)
#             training_ongoing = globals.get_training_ongoing()
#             if training_ongoing:
#                 with open('./responses.json') as f:
#                     responses = json.load(f)
#                 # asyncio.create_task(after_response())
#                 return {'Info':
#                         responses['info_train_predict']
#                         }, config_settings.get('errors').get('EMPTY_ERROR')
#             if not globals.get_app_running():
#                 with open('../utils_package/responses.json') as f:
#                     responses = json.load(f)
#                 # asyncio.create_task(after_response())
#                 return {'Error':
#                         responses['app_not_running']
#                         }, config_settings.get('errors').get('SERVER_ERROR')
#             try:
#                 data = await request.json
#                 if data is None:
#                     data_request = await request.form
#                     data = data_request.copy().to_dict()
#                 if isinstance(data, str):
#                     data = json.loads(data)
#             except Exception as ex:
#                 data = request.form.copy().to_dict()
#             output, status_code = await utils.process_request(data, 'train')
#             if status_code == config_settings.get('errors').get('SERVER_ERROR') or status_code == config_settings.get(
#                     'errors').get('INPUT_ERROR') or status_code == config_settings.get('errors').get('FILE_ERROR'):
#                 # asyncio.create_task(after_response())
#                 return output, status_code
#             # asyncio.create_task(after_response())
#             return output, status_code
#         except Exception as ex:
#             e = str(ex)
#             # try:
#             #     asyncio.create_task(after_response())
#             # except:
#             #     pass
#             return jsonify({'error': e}), config_settings.get('errors').get('INPUT_ERROR')
#
#
# @app.route('/retrain/', methods=['POST'])
# async def get_retrain_view():
#     if request.method == 'GET':
#         try:
#             # await globals.increment_requests()
#             # async def after_response():
#             #     await globals.decrement_requests()
#             with open('../utils_package/config_settings.json') as f:
#                 config_settings = json.load(f)
#             training_ongoing = globals.get_training_ongoing()
#             if training_ongoing:
#                 with open('./responses.json') as f:
#                     responses = json.load(f)
#                 # asyncio.create_task(after_response())
#                 return {'Info':
#                             responses['info_train_predict']
#                         }, config_settings.get('errors').get('EMPTY_ERROR')
#             if not globals.get_app_running():
#                 with open('../utils_package/responses.json') as f:
#                     responses = json.load(f)
#                 # asyncio.create_task(after_response())
#                 return {'Error':
#                             responses['app_not_running']
#                         }, config_settings.get('errors').get('SERVER_ERROR')
#             try:
#                 data = await request.json
#                 if data is None:
#                     data_request = await request.form
#                     data = data_request.copy().to_dict()
#                 if isinstance(data, str):
#                     data = json.loads(data)
#             except Exception as ex:
#                 data = request.form.copy().to_dict()
#             output, status_code = await utils.process_request(data, 'retrain')
#             if status_code == config_settings.get('errors').get('SERVER_ERROR') or status_code == config_settings.get(
#                     'errors').get('INPUT_ERROR') or status_code == config_settings.get('errors').get('FILE_ERROR'):
#                 # asyncio.create_task(after_response())
#                 return output, status_code
#             # asyncio.create_task(after_response())
#             return output, status_code
#         except Exception as ex:
#             e = str(ex)
#             # try:
#             #     asyncio.create_task(after_response())
#             # except:
#             #     pass
#             return jsonify({'error': e}), config_settings.get('errors').get('INPUT_ERROR')
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8086, debug=False)
