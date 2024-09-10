from peak import Agent, OneShotBehaviour, CyclicBehaviour, PeriodicBehaviour, Message
import random
from paho.mqtt import client as mqtt_client

broker = 'broker.emqx.io'
port = 1883
client_id_base = 'subscribe-'
subscribed_topics = set()


class SubscriberAgent(Agent):
    class SubscribeResults(CyclicBehaviour):
        async def connect_mqtt(self, client_id):
            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    print(f"Connected to MQTT Broker with client ID: {client_id}!")
                else:
                    print(f"Failed to connect, return code {rc}")

            client = mqtt_client.Client(client_id)
            client.on_connect = on_connect
            client.connect(broker, port)
            return client

        async def subscribe(self, client: mqtt_client.Client, topic: str):
            def on_message(client, userdata, msg):
                print(f"Received `{msg.payload.decode()}` from topic `{msg.topic}`")

            if topic not in subscribed_topics:
                client.subscribe(topic)
                client.on_message = on_message
                subscribed_topics.add(topic)
                print(f"Subscribed to topic: {topic}")
            else:
                print(f"Already subscribed to topic: {topic}")

        async def run(self):
            topic = "report/results"  # Topic to subscribe to

            client_id = client_id_base + str(random.randint(0, 1000))
            client = await self.connect_mqtt(client_id)
            await self.subscribe(client, topic)
            client.loop_forever()
    async def setup(self):
        b = self.SubscribeResults()
        self.add_behaviour(b)


