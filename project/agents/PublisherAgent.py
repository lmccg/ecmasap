# python 3.6
from peak import Agent, OneShotBehaviour, CyclicBehaviour, PeriodicBehaviour, Message
import random
import time
from datetime import datetime, timedelta
from paho.mqtt import client as mqtt_client
import concurrent.futures
from threading import Thread

broker = 'broker.emqx.io'
port = 1883
# Generate a Client ID with the publish prefix.
client_id_base = 'publish-'
# username = 'emqx'
# password = 'public'


def connect_mqtt(client_id):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)

        client = mqtt_client.Client(client_id)
        # client.username_pw_set(username, password)
        client.on_connect = on_connect
        client.connect(broker, port)
        return client

class PublisherAgent(Agent):
    class PublishResults(PeriodicBehaviour):
        async def run(self):
            msg = await self.receive(10)
            if msg:
                publishers = []
                topics = ["report/predictions"]
                client_id = client_id_base + str(random.randint(0, 1000))
                client = connect_mqtt(client_id)
                client.loop_start()
                content = msg.body
                for topic in topics:
                    publisher = self.MQTTPublisher(client_id, topic, content)
                    publishers.append(publisher)

                threads = []
                for publisher in publishers:
                    thread = Thread(target=publisher.run, args=(client,))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()


        class MQTTPublisher:
            def __init__(self, client_id, topic, content):
                self.client_id = client_id
                self.topic = topic
                self.content = content
                self.datetime_start = datetime.now()

            def publish(self, client):
                # msg_count = 1
                while True:
                    time.sleep(1)
                    result = client.publish(self.topic, self.content)
                    # result: [0, 1]
                    status = result[0]
                    if status == 0:
                        print(f"Send `{self.content}` to topic `{self.topic}` using client ID: {self.client_id}")
                    else:
                        print(f"Failed to send message to topic {self.topic} using client ID: {self.client_id}")
                    #comentar esta zona
                    # msg_count += 1
                    # if msg_count > 5:
                    #     break

            def run(self, client):
                self.publish(client)
                client.loop_stop()

    async def setup(self):
        print(f"Agent {self.jid} starting...")
        b = self.PublishResults()
        self.add_behaviour(b)