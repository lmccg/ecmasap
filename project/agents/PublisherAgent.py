# # python 3.6
# from peak import Agent, OneShotBehaviour, CyclicBehaviour, PeriodicBehaviour, Message
# import random
# import time
# from datetime import datetime, timedelta
# from paho.mqtt import client as mqtt_client
# import concurrent.futures
# from threading import Thread
#
# broker = 'broker.emqx.io'
# port = 1883
# # Generate a Client ID with the publish prefix.
# client_id_base = 'publish-'
# # username = 'emqx'
# # password = 'public'
#
# publish_interval = timedelta(minutes=5)
#
# published_topics = {}
#
#
# def connect_mqtt(client_id):
#         def on_connect(client, userdata, flags, rc):
#             if rc == 0:
#                 print("Connected to MQTT Broker!")
#             else:
#                 print("Failed to connect, return code %d\n", rc)
#
#         client = mqtt_client.Client(client_id)
#         client.on_connect = on_connect
#         client.connect(broker, port, keepalive=60*60)
#         return client
#
# class PublisherAgent(Agent):
#     class PublishResults(PeriodicBehaviour):
#         async def run(self):
#             msg = await self.receive()
#             if msg:
#                 print(f'got a message {msg.body}')
#                 publishers = []
#                 topics = ["report/results"]
#                 content = msg.body
#                 for topic in topics:
#                     # Check if this topic and message were already published recently
#                     if topic in published_topics:
#                         last_published, last_content = published_topics[topic]
#                         if last_content == content and datetime.now() - last_published < publish_interval:
#                             print(f"Skipping publishing to `{topic}` as the message has been published recently.")
#                             continue
#
#                     client_id = client_id_base + str(random.randint(0, 1000))
#                     client = connect_mqtt(client_id)
#                     client.loop_start()
#                     print('body', content)
#                     # for topic in topics:
#                     print('topic', topic, client_id)
#                     publisher = self.MQTTPublisher(client_id, topic, content)
#                     thread = Thread(target=publisher.run, args=(client,))
#                     thread.start()
#                     thread.join()
#
#                     # Store the publication timestamp and content
#                     published_topics[topic] = (datetime.now(), content)
#                     # publishers.append(publisher)
#
#                 # threads = []
#                 # for publisher in publishers:
#                 #     thread = Thread(target=publisher.run, args=(client,))
#                 #     thread.start()
#                 #     threads.append(thread)
#                 #
#                 # for thread in threads:
#                 #     thread.join()
#
#
#         class MQTTPublisher:
#             def __init__(self, client_id, topic, content):
#                 self.client_id = client_id
#                 self.topic = topic
#                 self.content = content
#                 self.datetime_start = datetime.now()
#
#             def publish(self, client):
#                 # msg_count = 1
#                 # while True:
#                 #     time.sleep(1)
#                     result = client.publish(self.topic, self.content)
#                     # result: [0, 1]
#                     status = result[0]
#                     if status == 0:
#                         print(f"Send `{self.content}` to topic `{self.topic}` using client ID: {self.client_id}")
#                     else:
#                         print(f"Failed to send message to topic {self.topic} using client ID: {self.client_id}")
#                     #comentar esta zona
#                     # msg_count += 1
#                     # if msg_count > 5:
#                     #     break
#
#             def run(self, client):
#                 self.publish(client)
#                 client.loop_stop()
#
#     async def setup(self):
#         b = self.PublishResults(period=1)
#         self.add_behaviour(b)

from peak import Agent, PeriodicBehaviour
import time
from datetime import datetime, timedelta
from paho.mqtt import client as mqtt_client
from threading import Thread
import random
from utils_package.utils import timestamp_with_time_zone

broker = 'broker.emqx.io'

port = 1883

client_id_base = 'publish-'

publish_interval = timedelta(minutes=5)

published_topics = {}


def connect_mqtt(client_id, max_retries=5, retry_interval=5):
    """Connect to the MQTT broker and handle reconnection."""

    def on_connect(client, userdata, flags, rc):

        if rc == 0:

            print(f"{timestamp_with_time_zone()} Connected to MQTT Broker with client ID: {client_id}")

        else:

            print(f"{timestamp_with_time_zone()} Failed to connect, return code {rc}")

    client = mqtt_client.Client(client_id)

    client.on_connect = on_connect

    # Try to connect with retries

    retries = 0

    while retries < max_retries:

        try:

            client.connect(broker, port, keepalive=60 * 60)

            return client

        except Exception as e:

            retries += 1

            print(f"{timestamp_with_time_zone()} Connection failed, retrying {retries}/{max_retries}: {e}")

            time.sleep(retry_interval)

    raise Exception("Failed to connect to MQTT Broker after several retries.")


class PublisherAgent(Agent):
    class PublishResults(PeriodicBehaviour):

        async def run(self):

            msg = await self.receive()

            if msg:

                print(f"{timestamp_with_time_zone()} Got a message: {msg.body}")

                topics = ["report/results"]

                content = msg.body

                for topic in topics:

                    # Check if this topic and message were already published recently

                    if topic in published_topics:

                        last_published, last_content = published_topics[topic]

                        if last_content == content and datetime.now() - last_published < publish_interval:
                            print(f"{timestamp_with_time_zone()} Skipping publishing to `{topic}` as the message has been published recently.")

                            continue

                    client_id = client_id_base + str(random.randint(0, 1000))

                    client = connect_mqtt(client_id)

                    client.loop_start()

                    publisher = self.MQTTPublisher(client_id, topic, content)

                    thread = Thread(target=publisher.run, args=(client,))

                    thread.start()

                    thread.join()

                    # Store the publication timestamp and content

                    published_topics[topic] = (datetime.now(), content)

        class MQTTPublisher:

            def __init__(self, client_id, topic, content):

                self.client_id = client_id

                self.topic = topic

                self.content = content

            def publish(self, client):

                """Publish the message and check for success."""

                try:

                    result = client.publish(self.topic, self.content)

                    status = result[0]

                    if status == 0:

                        print(f"{timestamp_with_time_zone()} Sent `{self.content}` to topic `{self.topic}` using client ID: {self.client_id}")

                    else:

                        print(f"{timestamp_with_time_zone()} Failed to send message to topic {self.topic} using client ID: {self.client_id}")

                except Exception as e:

                    print(f"{timestamp_with_time_zone()} Error publishing message: {e}")

                    # Reconnect if necessary

                    print(timestamp_with_time_zone(), "Attempting to reconnect and publish...")

                    client.reconnect()

                    self.publish(client)

            def run(self, client):

                self.publish(client)

                client.loop_stop()

    async def setup(self):

        b = self.PublishResults(period=1)

        self.add_behaviour(b)

