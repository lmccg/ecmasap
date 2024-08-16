from peak import Agent, OneShotBehaviour, CyclicBehaviour, PeriodicBehaviour, Message
from peak.behaviours import OneShotBehaviour
from spade.template import Template
import time
import json


class TargetAgent(Agent):
    class requestAction(CyclicBehaviour):
        def __init__(self, agent_name, slash, model, target, request_type, request_data):
            super().__init__()
            self.agent_name = agent_name
            self.slash = slash
            self.model = model
            self.target = target
            self.request_type = request_type
            self.request_data = request_data

        async def run(self):
            msg = await self.receive(10)
            if msg:
                msg = Message(to=f"{self.agent_name}@{self.agent.jid.domain}/{self.slash}")
                msg.set_metadata("performative", "request")  # Set the "inform" FIPA performative
                if isinstance(self.request_data, dict):
                    self.request_data = json.dumps(self.request_data)
                msg.body = self.request_type + "|" + self.model + "|" + self.request_data
                await self.send(msg)

    class ReceiveMsg(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"TargetAgent: {msg.sender} sent me a message: '{msg.body}'")
                parts_of_msg = msg.body.split("|")
                self.request_type = parts_of_msg[0]
                self.target = parts_of_msg[1]
                self.model = parts_of_msg[2]
                self.request_data = parts_of_msg[3]

                print(parts_of_msg)
                with open('../utils_package/config_data.json') as config_file:
                   config_data = json.load(config_file)
                model_type = config_data.get("model_type").get(self.target)
                with open('../utils_package/config_agents.json') as config_file:
                    config_agents = json.load(config_file)
                print('lets send some requests')
                agents = config_agents['ml_model_agents'][model_type]
                for agent, model in agents.items():
                    print(agent, model)

                response_msg = msg.make_reply()
                response_msg.set_metadata("performative", "inform")
                response_msg.body = str(scores)
                await self.send(response_msg)

        async def on_def(self):
            await self.agent.stop()

    # Setup function for the target agent
    async def setup(self):
        print(f"Agent {self.jid} starting...")
        request_training = True
        print('in handle', request_training)
        time.sleep(10)
        request_training = True
        request_predict = False
        with open('../utils_package/config_agents.json') as config_file:
            config = json.load(config_file)
        agents = config['agents']
        for agent, model in agents.items():
            print(agent, model)
            if request_training:
                request_type = 'train'
            elif request_predict:
                request_type = 'predict'
            b = self.SendMessage(agent_name=agent, slash=agent, model=model, request_type=request_type, request_data=None)
            self.add_behaviour(b)