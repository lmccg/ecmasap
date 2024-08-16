# python3.6

import random

from paho.mqtt import client as mqtt_client


broker = 'broker.emqx.io'
port = 1883
# Generate a Client ID with the subscribe prefix.
client_id = f'subscribe-{random.randint(0, 100)}'
# username = 'emqx'
# password = 'public'
# topic ="report/agent4"

# class MQTTSubscriber:
#     def __init__(self, topic):
#         self.topic = topic
#         self.client = self.connect_mqtt()

#     def connect_mqtt(self) -> mqtt_client:
#         def on_connect(client, userdata, flags, rc):
#             if rc == 0:
#                 print(f"Connected to MQTT Broker! Topic: {self.topic}")
#             else:
#                 print(f"Failed to connect, return code {rc} - Topic: {self.topic}")

#         client = mqtt_client.Client(client_id)
#         # client.username_pw_set(username, password)
#         client.on_connect = on_connect
#         client.connect(broker, port)
#         return client

#     def subscribe(self):
#         def on_message(client, userdata, msg):
#             print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic - Subscriber: {self.client._client_id.decode()}")

#         self.client.subscribe(self.topic)
#         self.client.on_message = on_message

#     def run(self):
#         self.subscribe()
#         self.client.loop_forever()

# if __name__ == '__main__':
#     subscriber = MQTTSubscriber(topic)
#     subscriber.run()

def on_connect(client, userdata, flags, rc):
    print('on_connect')
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print("Failed to connect, return code %d\n", rc)


def on_message(client, userdata, msg):
    print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")


def run():
    print('run')
    client = mqtt_client.Client()
    client.on_connect = on_connect
    client.connect(broker, port)
    # client.subscribe(topic)
    client.on_message = on_message

if __name__ == "__main__":
    print('main')
    run()