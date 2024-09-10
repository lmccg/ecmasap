from peak import Agent, Message, PeriodicBehaviour, CyclicBehaviour
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from sklearn.metrics.pairwise import cosine_similarity
import json

class RLSelectorAgent(Agent):
    class ReceiveMsg(PeriodicBehaviour):
        async def run(self):
            msg = await self.receive()
            if msg:
                self.similarity_threshold = 0.85
                self.models = []
                self.target_characteristics = {}
                self.model = None
                self.env = None
                self.error_metric = None

                print(f"model: {msg.sender} sent me a message: '{msg.body}'")
                request_data = msg.body
                request_data = json.loads(request_data)
                with open('utils_package/config_settings.json') as config_file:
                    config_settings = json.load(config_file)
                similarity_threshold = config_settings['similarity_threshold']  # could be 0.85
                with open('utils_package/config_agents.json') as config_file:
                    config_agents = json.load(config_file)
                database_agent = config_agents["database_agent"]
                request_models = Message(to=f"{database_agent}@{self.agent.jid.domain}/{database_agent}")
                request_models.set_metadata("performative", "request")
                request_models.body = 'get_models'
                await self.send(request_models)
                response = await self.receive()
                if response:
                    if response.get_metadata("performative") == "inform":
                        models = response.body
                        models = json.loads(models)
                    else:
                        models = []
                else:
                    models = []
                self.target_characteristics = request_data
                self.models = models
                self.similarity_threshold = similarity_threshold
                # Setup environment and model
                self.setup_environment()
                self.train_model(total_timesteps=10000)

                # Evaluate and print results
                similar_models = self.get_similar_models()
                # REPLY BACK
                response_msg = msg.make_reply()
                response_msg.set_metadata("performative", "inform")
                response_msg.body = json.dumps(similar_models)
                await self.send(response_msg)


        def setup_environment(self):
            """
            Setup the environment and PPO model.
            """
            self.env = self.create_environment()
            self.model = PPO("MlpPolicy", self.env, verbose=1)

        def create_environment(self):
            """
            Create the gym environment.
            """

            class ModelSelectionEnv(gym.Env):
                def __init__(self, agent):
                    super(ModelSelectionEnv, self).__init__()
                    self.agent = agent
                    self.action_space = spaces.Discrete(len(agent.models))
                    self.observation_space = spaces.Box(
                        low=0, high=1, shape=(len(agent.target_characteristics),), dtype=np.float32
                    )
                    self.current_step = 0

                def reset(self):
                    self.current_step = 0
                    return self._get_observation()

                def _get_observation(self):
                    model = self.agent.models[self.current_step]
                    return np.array(list(model['characteristics'].values()), dtype=np.float32)

                def step(self, action):
                    selected_model = self.agent.models[action]
                    model_characteristics = np.array(list(selected_model['characteristics'].values())).reshape(1, -1)
                    target_values = np.array(list(self.agent.target_characteristics.values())).reshape(1, -1)

                    similarity = cosine_similarity(model_characteristics, target_values)[0][0]

                    if similarity >= self.agent.similarity_threshold:
                        r2 = selected_model['errors'][selected_model['default_metric']]
                        reward = similarity + r2
                    else:
                        reward = -1

                    done = True
                    self.current_step = (self.current_step + 1) % len(self.agent.models)
                    return self._get_observation(), reward, done, {}

                def render(self, mode='human'):
                    pass

                def close(self):
                    pass

            return ModelSelectionEnv(self)

        def train_model(self, total_timesteps=10000):
            """
            Train the PPO model.
            """
            self.model.learn(total_timesteps=total_timesteps)

        def get_similar_models(self):
            """
            Get models that are similar to the target characteristics and sort them by similarity then by metric.
            """
            similar_models = []
            for mdl in self.models:
                model_characteristics = np.array(list(mdl['characteristics'].values())).reshape(1, -1)
                target_values = np.array(list(self.target_characteristics.values())).reshape(1, -1)

                similarity = cosine_similarity(model_characteristics, target_values)[0][0]

                if similarity >= self.similarity_threshold:
                    similar_models.append((mdl['id'], mdl['ml_model'], mdl['model_type'], similarity, mdl['errors'][mdl['default_metric']]))

            similar_models.sort(key=lambda x: (x[3], x[4]), reverse=True)
            return similar_models
    async def setup(self):
        self.add_behaviour(self.ReceiveMsg(period=1))