import asyncio

from peak import Agent, OneShotBehaviour, CyclicBehaviour, PeriodicBehaviour, Message
from spade.template import Template
import aiohttp_cors
import time
import json
training_on_going = False

class MainAgent(Agent):
    class GetDataFromAgent(OneShotBehaviour):
        async def run(self):
            jid = self.get('selected_agent')
            msg = Message()
            msg.to = jid
            msg_body = ''
            if self.get('request_type'):
                msg_body += self.get('request_type') + "|"
            if self.get('data'):
                if self.get('data').get('target'):
                    msg_body += self.get('data').get('target') + "|"
                msg_body += json.dumps(self.get('data'))
            msg.set_metadata("performative", "request")
            msg.body = msg_body
            await self.send(msg)
            response = await self.receive()
            try:
                self.set('agent_target', json.loads(response.body))
            except:
                self.set('agent_target', response.body)

    # class StartApp (OneShotBehaviour):
    #     def __init__(self, jid):
    #         self.jid = jid
    #     async def run(self):
    #         print('checking models to train')
    #         with open('utils_package/config_agents.json') as f:
    #             config_agents = json.load(f)
    #         target_agent = config_agents['target_agent']
    #         jid = f'{target_agent}@{self.jid}'
    #         self.set('selected_agent', jid)
    #         self.set('request_type', 'needed_models_to_train')
    #         behaviour = self.GetDataFromAgent()
    #         self.add_behaviour(behaviour)
    #         await behaviour.join()
    #         self.set('selected_agent', None)
    #         data = self.get('agent_target')
    #         try:
    #             data = json.loads(data)
    #         except:
    #             data = None
    #         print(data)
    #         if data is not None:
    #             global training_on_going
    #             training_on_going = True
    #             for entry in data:
    #                 self.train_model(entry)
    #                 await asyncio.sleep(10)
    #             training_on_going = False
    class StartApp(OneShotBehaviour):
        def __init__(self, jid):
            super().__init__()  # Call the constructor of OneShotBehaviour
            self.jid = jid

        async def run(self):
            await asyncio.sleep(10)
            with open('utils_package/config_agents.json') as f:
                config_agents = json.load(f)
            target_agent = config_agents['target_agent']
            jid = f'{target_agent}@{self.jid}'
            self.agent.set('selected_agent', jid)  # Use self.agent to access the agent instance
            self.agent.set('request_type', 'needed_models_to_train')
            behaviour = self.agent.GetDataFromAgent()
            self.agent.add_behaviour(behaviour)
            await behaviour.join()
            self.agent.set('selected_agent', None)
            data = self.agent.get('agent_target')
            print(data)
            if data is not None:
                for entry in data:
                    print('sending', entry)
                    await self.agent.train_model(entry)
                    await asyncio.sleep(10)
                    print('okkk')

    async def get_real_data_view(self, request):
        with open('utils_package/config_agents.json') as f:
            config_settings = json.load(f)
        try:
            data = await request.json()
            if data is None:
                data_request = await request.form
                data = data_request.copy().to_dict()
            if isinstance(data, str):
                data = json.loads(data)
        except Exception as ex:
            data = request.form.copy().to_dict()
        if 'target' not in data:
            return 'Missing target entry!'
        if 'target_table' not in data:
            return 'Missing target table entry!'
        if 'frequency' not in data:
            return 'Missing frequency entry!'
        if 'start_date' not in data:
            return 'Missing start date entry!'
        if 'start_time' not in data:
            return 'Missing start time entry!'
        if 'end_date' not in data:
            return 'Missing end date entry!'
        if 'end_time' not in data:
            return 'Missing end time entry!'
        target_agent = config_settings['target_agent']
        jid = f'{target_agent}@{self.jid.domain}'
        self.set('selected_agent', jid)
        self.set('data', data)
        self.set('request_type', 'get_real_data')
        self.set('target', data.get('target'))
        behaviour = self.GetDataFromAgent()
        self.add_behaviour(behaviour)
        await behaviour.join()

        self.set('selected_agent', None)
        data = self.get('agent_target')
        return data

    async def get_predictions_view(self, request):
        global training_on_going
        if not training_on_going:
            with open('utils_package/config_agents.json') as f:
                config_settings = json.load(f)
            try:
                data = await request.json()
                if data is None:
                    data_request = await request.form
                    data = data_request.copy().to_dict()
                if isinstance(data, str):
                    data = json.loads(data)
            except Exception as ex:
                data = request.form.copy().to_dict()
            if 'target' not in data:
                return 'Missing target entry!'
            if 'target_table' not in data:
                return 'Missing target table entry!'
            if 'frequency' not in data:
                return 'Missing frequency entry!'
            if 'start_date' not in data:
                return 'Missing start date entry!'
            if 'start_time' not in data:
                return 'Missing start time entry!'
            target_agent = config_settings['target_agent']
            jid = f'{target_agent}@{self.jid.domain}'
            self.set('selected_agent', jid)
            self.set('data', data)
            self.set('request_type', 'predict')
            self.set('target', data.get('target'))
            behaviour = self.GetDataFromAgent()
            self.add_behaviour(behaviour)
            await behaviour.join()

            self.set('selected_agent', None)
            data = self.get('agent_target')
            print('from predictions', data)
            return data
        else:
            return 'There is a training going on, please wait and try again!'

    async def get_train_view(self, request):
        global training_on_going
        if not training_on_going:
            training_on_going = True
            with open('utils_package/config_agents.json') as f:
                config_settings = json.load(f)
            try:
                data = await request.json()
                if data is None:
                    data_request = await request.form
                    data = data_request.copy().to_dict()
                if isinstance(data, str):
                    data = json.loads(data)
            except Exception as ex:
                data = request.form.copy().to_dict()
            if 'target' not in data:
                return 'Missing target entry!'
            if 'target_table' not in data:
                return 'Missing target table entry!'
            if 'frequency' not in data:
                return 'Missing frequency entry!'
            if 'end_date' not in data:
                return 'Missing end date entry!'
            if 'end_time' not in data:
                return 'Missing end time entry!'
            target_agent = config_settings['target_agent']
            jid = f'{target_agent}@{self.jid.domain}'
            self.set('selected_agent', jid)
            self.set('data', data)
            self.set('request_type', 'train')
            self.set('target', data.get('target'))
            behaviour = self.GetDataFromAgent()
            self.add_behaviour(behaviour)
            await behaviour.join()

            self.set('selected_agent', None)
            data = self.get('agent_target')
            training_on_going = False
            return data
        else:
            return 'There is a training going on, please wait and try again!'
    async def train_model(self, data):
        global training_on_going
        if not training_on_going:
            training_on_going = True
            with open('utils_package/config_agents.json') as f:
                config_settings = json.load(f)
            if 'target' not in data:
                return 'Missing target entry!'
            if 'target_table' not in data:
                return 'Missing target table entry!'
            if 'frequency' not in data:
                return 'Missing frequency entry!'
            if 'end_date' not in data:
                return 'Missing end date entry!'
            if 'end_time' not in data:
                return 'Missing end time entry!'
            target_agent = config_settings['target_agent']
            jid = f'{target_agent}@{self.jid.domain}'
            self.set('selected_agent', jid)
            self.set('data', data)
            self.set('request_type', 'train')
            self.set('target', data.get('target'))
            behaviour = self.GetDataFromAgent()
            self.add_behaviour(behaviour)
            await behaviour.join()

            self.set('selected_agent', None)
            data = self.get('agent_target')
            training_on_going = False
            print('data', data)
            return data
        else:
            return 'There is a training going on, please wait and try again!'

    async def get_retrain_view(self, request):
        global training_on_going
        if not training_on_going:
            training_on_going = True
            with open('utils_package/config_agents.json') as f:
                config_settings = json.load(f)
            try:
                data = await request.json()
                if data is None:
                    data_request = await request.form
                    data = data_request.copy().to_dict()
                if isinstance(data, str):
                    data = json.loads(data)
            except Exception as ex:
                data = request.form.copy().to_dict()
            if 'target' not in data:
                return 'Missing target entry!'
            if 'target_table' not in data:
                return 'Missing target table entry!'
            if 'frequency' not in data:
                return 'Missing frequency entry!'
            if 'end_date' not in data:
                return 'Missing end date entry!'
            if 'end_time' not in data:
                return 'Missing end time entry!'
            target_agent = config_settings['target_agent']
            jid = f'{target_agent}@{self.jid.domain}'
            self.set('selected_agent', jid)
            self.set('data', data)
            self.set('request_type', 'retrain')
            self.set('target', data.get('target'))
            behaviour = self.GetDataFromAgent()
            self.add_behaviour(behaviour)
            await behaviour.join()

            self.set('selected_agent', None)
            data = self.get('agent_target')
            training_on_going = False
            return data
        else:
            return 'There is a training going on, please wait and try again!'


    async def setup(self):
        self.web.add_post('/predict', self.get_predictions_view, None)
        self.web.add_post('/train', self.get_train_view, None)
        self.web.add_post('/retrain', self.get_retrain_view, None)
        self.web.add_post('/get_real_data', self.get_real_data_view, None)
        # Configure default CORS settings.
        cors = aiohttp_cors.setup(
            self.web.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
            },
        )

        # Configure CORS on all routes.
        for route in list(self.web.app.router.routes()):
            cors.add(route)
        self.web.start('0.0.0.0', 8080)
        b = self.StartApp(self.jid.domain)
        self.add_behaviour(b)

    # class SendMessage(CyclicBehaviour):
    #     def __init__(self, agent_name, slash):
    #         super().__init__()
    #         self.agent_name = agent_name
    #         self.slash = slash
    #
    #     async def run(self):
    #         start_msg = await self.receive(timeout=10)
    #         if start_msg:
    #             parts_of_msg = start_msg.body.split("|")
    #             request_type = parts_of_msg[0]
    #             request_data = parts_of_msg[2]
    #             target = parts_of_msg[1]
    #             msg = Message(to=f"{self.agent_name}@{self.agent.jid.domain}/{self.slash}")
    #             msg.set_metadata("performative", "request")  # Set the "inform" FIPA performative
    #             if isinstance(request_data, dict):
    #                 request_data = json.dumps(request_data)
    #             msg.body = request_type + "|" + target + "|" + request_data
    #             await self.send(msg)
    #
    #             # Wait for a response
    #             response = await self.receive()
    #             if response:
    #                 if response.get_metadata("performative") == "inform":
    #                     print(f"Response received: {response.body}")
    #                 elif response.get_metadata("performative") == "failure":
    #                     print(f"Request failed: {response.body}")
    #             else:
    #                 print("No response received within timeout.")
    #
    # # Setup function for the main agent
    # async def setup(self):
    #     print(f"Agent {self.jid} starting...")
    #     request_training = True
    #     print('in handle', request_training)
    #     time.sleep(10)
    #     request_training = True
    #     request_predict = False
    #     with open('../utils_package/config_agents.json') as config_file:
    #         config = json.load(config_file)
    #     agent = config['target_agent']
    #     if request_training:
    #         request_type = 'train'
    #     elif request_predict:
    #         request_type = 'predict'
    #     b = self.SendMessage(agent_name=agent, slash=agent)
    #     self.add_behaviour(b)
