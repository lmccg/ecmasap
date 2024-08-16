from peak import Agent, Message, PeriodicBehaviour, CyclicBehaviour


class RLSelectorAgent(Agent):
    class ReceiveMsg(PeriodicBehaviour):
        async def run(self):
            msg = await self.receive(10)
            if msg:
                print(f"model: {msg.sender} sent me a message: '{msg.body}'")
                parts_of_msg = msg.body.split("|")
                dataset_type = parts_of_msg[0]
                target = parts_of_msg[1]
                request_data = parts_of_msg[2]

                # todo: process data

                # REPLY BACK
                response_msg = msg.make_reply()
                response_msg.set_metadata("performative", "inform")
                response_msg.body = "success"
                await self.send(response_msg)

    async def setup(self):
        self.add_behaviour(self.ReceiveMsg(period=1))
