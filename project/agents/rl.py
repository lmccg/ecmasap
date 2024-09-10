from peak import Agent, PeriodicBehaviour, Message
from datetime import datetime, timedelta
import numpy as np
import random


class RetrainAgent(Agent):
    class ReceiveMsg(PeriodicBehaviour):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.q_table = {}  # Store Q-values for state-action pairs
            self.learning_rate = 0.1
            self.discount_factor = 0.95
            self.epsilon = 0.1  # For epsilon-greedy action selection
            self.retraining_count = 0

        async def run(self):
            msg = await self.receive()
            if msg:
                models = self.getAllModels()

                for model in models:
                    state = self.get_state(model)
                    action = self.choose_action(state)

                    if action == 1:
                        self.retrain_model(model)

                    reward = self.calculate_reward(model, action)
                    next_state = self.get_state(model)

                    # Update the Q-table with the observed reward and next state
                    self.update_q_table(state, action, reward, next_state)

        def get_state(self, model):
            """
            Generate a state representation for the model.
            The state is defined by:
            - R² degradation.
            - Feature deviations.
            - Time since last retraining.
            """
            r2_degradation = 1 - model['errors']['r2']  # Higher degradation = worse model
            feature_deviation = np.mean(
                [abs(model['current_feature_values'][f] - std) for f, std in model['feature_stds'].items()])
            time_since_training = (datetime.now() - model['training_date']).days / 90  # Normalize to 0-1 range

            return (round(r2_degradation, 2), round(feature_deviation, 2), round(time_since_training, 2))

        def choose_action(self, state):
            """
            Epsilon-greedy policy to choose whether to retrain (action=1) or not (action=0).
            """
            if random.uniform(0, 1) < self.epsilon:
                return random.choice([0, 1])  # Explore: random action
            return np.argmax(self.q_table.get(state, [0, 0]))  # Exploit: pick action with highest Q-value

        def update_q_table(self, state, action, reward, next_state):
            """
            Update the Q-table based on the agent's experience.
            """
            old_q_value = self.q_table.get(state, [0, 0])[action]
            next_max_q = max(self.q_table.get(next_state, [0, 0]))

            # Q-learning formula
            new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * next_max_q - old_q_value)

            # Update the Q-table
            if state not in self.q_table:
                self.q_table[state] = [0, 0]  # Initialize actions [don't retrain, retrain]
            self.q_table[state][action] = new_q_value

        def calculate_reward(self, model, action):
            """
            Define a reward function:
            - Positive reward if retraining improves performance.
            - Negative reward if retraining is unnecessary or fails to improve.
            """
            initial_r2 = model['errors']['r2']

            if action == 1:  # Retrain
                # Simulate retraining (in practice, call your actual retraining logic)
                new_r2 = min(1.0, initial_r2 + random.uniform(0, 0.1))  # Simulate improvement
                model['errors']['r2'] = new_r2
                self.retraining_count += 1

                # Reward is based on improvement in R²
                reward = (new_r2 - initial_r2) * 100  # Scale the reward for easier tuning
            else:  # Do nothing
                reward = -1 if initial_r2 < 0.8 else 0  # Penalize not retraining if R² is low

            return reward

        def retrain_model(self, model):
            print(f"Retraining model: {model['name']}...")
    async def setup(self):
        b = self.ReceiveMsg(period=1)
        self.add_behaviour(b)

    def getAllModels(self):
        # Dummy implementation - replace with actual model retrieval logic
        return [
            {
                'name': 'Model A',
                'errors': {'r2': 0.75},
                'feature_stds': {'feature1': 0.2, 'feature2': 0.1},
                'current_feature_values': {'feature1': 0.3, 'feature2': 0.15},
                'training_date': datetime.now() - timedelta(days=100)
            },
            {
                'name': 'Model B',
                'errors': {'r2': 0.85},
                'feature_stds': {'feature1': 0.1, 'feature2': 0.2},
                'current_feature_values': {'feature1': 0.15, 'feature2': 0.25},
                'training_date': datetime.now() - timedelta(days=50)
            },
            # Add more models here...
        ]
