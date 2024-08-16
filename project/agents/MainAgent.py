from peak import Agent, OneShotBehaviour, CyclicBehaviour, PeriodicBehaviour, Message
from peak.behaviours import OneShotBehaviour
from spade.template import Template
import time
import json


class MainAgent(Agent):
    class SendMessage(CyclicBehaviour):
        def __init__(self, agent_name, slash, target, request_type, request_data):
            super().__init__()
            self.agent_name = agent_name
            self.slash = slash
            self.target = target
            self.request_type = request_type
            self.request_data = request_data

        async def run(self):
            msg = Message(to=f"{self.agent_name}@{self.agent.jid.domain}/{self.slash}")
            msg.set_metadata("performative", "request")  # Set the "inform" FIPA performative
            if isinstance(self.request_data, dict):
                self.request_data = json.dumps(self.request_data)
            msg.body = self.request_type + "|" + self.target + "|" + self.request_data
            await self.send(msg)

            # Wait for a response
            response = await self.receive(timeout=60)
            if response:
                if response.get_metadata("performative") == "inform":
                    print(f"Response received: {response.body}")
                elif response.get_metadata("performative") == "failure":
                    print(f"Request failed: {response.body}")
            else:
                print("No response received within timeout.")

    # Setup function for the main agent
    async def setup(self):
        print(f"Agent {self.jid} starting...")
        request_training = True
        print('in handle', request_training)
        time.sleep(10)
        request_training = True
        request_predict = False
        with open('../utils_package/config_agents.json') as config_file:
            config = json.load(config_file)
        agent = config['target_agent']
        if request_training:
            request_type = 'train'
        elif request_predict:
            request_type = 'predict'
        b = self.SendMessage(agent_name=agent, slash=agent, target=None, request_type=request_type, request_data=None)
        self.add_behaviour(b)
