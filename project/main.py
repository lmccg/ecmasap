from quart import Quart, jsonify
from quart_cors import cors
from utils_package import repository, utils, globals
import json
from agents import MainAgent
app = Quart(__name__, template_folder='templates')
app = cors(app, allow_origin="*")


@app.route('/ping/', methods=['GET'])
async def get_pong_view():
    with open('./config_settings.json') as f:
        config_settings = json.load(f)
    if not globals.get_app_running():
        with open('./responses.json') as f:
            responses = json.load(f)
        return jsonify({'Info': responses['app_not_running']
                        }), config_settings.get('errors').get('SERVER_ERROR')
    return jsonify('pong', 200)


@app.route('/stop/', methods=['GET'])
async def stop_app():
    if globals.get_app_running():
        globals.change_app_running(False)
        # await threadTraining.stop_training()
        return jsonify({'info': 'Request to stop the app accepted! As soon as possible the app will be stopped.'}), 200
    return jsonify({'info': 'The app already has a request to stop!'
                    }), 404


@app.route('/restart/', methods=['GET'])
async def restart_app():
    if globals.get_training_ongoing() and not globals.get_app_running():
        return jsonify({'info': 'The app is still training some models, with a request to stop the app yet to perform.'
                                'Please try again later!'
                        }), 404
    elif globals.get_app_running():
        return jsonify({'info': 'The app is already running!'
                        }), 404
    globals.reset_shutdown_event()
    training_ongoing = globals.get_training_ongoing()
    globals.change_app_running(True)
    if not training_ongoing:
        # todo
        print('vamos chamar o treino de modelos')
        # output, status_code = await threadTraining.trainer()
    return jsonify({'info': 'App restarted'}), 200


@app.before_serving
async def start_app():
    repository.create_database_if_not_exists()
    repository.initialize_database()
    print(f"[{str(utils.timestamp_with_time_zone())}] [ecmasap] #################STARTED#######################")
    training_ongoing = globals.get_training_ongoing()
    globals.change_app_running(True)
    if not training_ongoing:
        # todo
        print('vamos treinar modelos ou então o main agent')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8086, debug=False)
