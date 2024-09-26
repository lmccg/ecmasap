from peak import Agent, OneShotBehaviour, CyclicBehaviour, Message
import random
from paho.mqtt import client as mqtt_client
from utils_package.utils import timestamp_with_time_zone

broker = 'broker.emqx.io'
port = 1883
client_id_base = 'subscribe-'
subscribed_topics = set()
processed_messages_id = set()


class SubscriberAgent(Agent):
    class SubscribeResults(CyclicBehaviour):
        async def connect_mqtt(self, client_id):
            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    print(f"{timestamp_with_time_zone()} Connected to MQTT Broker with client ID: {client_id}!")
                else:
                    print(f"{timestamp_with_time_zone()} Failed to connect, return code {rc}")

            def on_disconnect(client, userdata, flags, rc):
                if rc == 0:
                    print(f"{timestamp_with_time_zone()} Disconnected!")
                else:
                    print(f"{timestamp_with_time_zone()} Failed to disconnect, return code {rc}")

            client = mqtt_client.Client(client_id)
            client.on_connect = on_connect
            client.on_disconnect = on_disconnect
            client.connect(broker, port)
            return client

        async def subscribe(self, client: mqtt_client.Client, topic: str):
            def on_message(client, userdata, msg):
                msg_received = msg.payload.decode()
                msg_received = msg_received.split('|')
                id_msg = msg_received[0]
                if id_msg not in processed_messages_id:
                    processed_messages_id.add(id_msg)
                    print(f"{timestamp_with_time_zone()} Received `{msg_received[1]}` from topic `{msg.topic}`")

            if topic not in subscribed_topics:
                client.subscribe(topic)
                client.on_message = on_message
                subscribed_topics.add(topic)
                print(f"{timestamp_with_time_zone()} Subscribed to topic: {topic}")
            else:
                print(f"{timestamp_with_time_zone()} Already subscribed to topic: {topic}")

        async def run(self):
            topic = "report/results"  # Topic to subscribe to
            client_id = client_id_base + str(random.randint(0, 1000))
            client = await self.connect_mqtt(client_id)
            try:
                await self.subscribe(client, topic)
                client.loop_forever()
            except KeyboardInterrupt:
                print('keyboard stop')
                client.disconnect()
                client.loop_stop()
    async def setup(self):
        b = self.SubscribeResults()
        self.add_behaviour(b)


