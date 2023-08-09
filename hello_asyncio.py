# %%
import asyncio
import random

# %%
import asyncio
import random

class Agent:
    def __init__(self, id, server_semaphore, steps_per_round=5):
        self.event = asyncio.Event()
        self.server_semaphore = server_semaphore
        self.id = id
        self.steps_per_round = steps_per_round
        self.t = 0
        self.done = False

    async def step(self):
        print(f'Agent {self.id}: step {self.t}')
        self.t += 1

    async def run(self):
        while not self.done:
            for _ in range(self.steps_per_round):
                await asyncio.sleep(random.random())  # Simulate variable time per step
                await self.step()
            print(f'Agent {self.id} has done its work.')
            if self.t > 20:
                self.done = True
            # await self.server_semaphore.acquire()  # Send signal to the server
            self.server_semaphore.release()  # Release the semaphore
            await self.event.wait()
            # self.event.clear()

class Server:
    def __init__(self, agents, semaphore):
        self.agents = agents
        self.semaphore = semaphore
        self.event = asyncio.Event()

    async def run(self):
        while not all(agent.done for agent in self.agents): # Add this line
            print('Server: Waiting for agents...')
            for _ in range(len(self.agents)):
                await self.semaphore.acquire()
            print('Server: Waking up agents...')
            for agent in self.agents:
                agent.event.set()

async def main():
    num_agents = 2
    server_semaphore = asyncio.Semaphore(num_agents)  # Set the initial count to the number of agents
    agents = [Agent(i, server_semaphore) for i in range(num_agents)]
    server = Server(agents, server_semaphore)
    await asyncio.gather(server.run(), *(agent.run() for agent in agents))


# asyncio.run(main())


await main()



# asyncio.run(main()) # in Terminal


    
# %%
import asyncio
import random

class Factorial:
    def __init__(self, id, n, steps_per_round, server_semaphore):
        self.event = asyncio.Event()
        self.event_for_server = asyncio.Event()
        self.id = id
        self.n = n
        self.steps_per_round = steps_per_round
        self.steps = 0
        self.result = 1
        self.done = False
        self.server_semaphore = server_semaphore

    async def calculate(self, i):
        if i > self.n:
            self.done = True
            # self.server_semaphore.release()  # Release the semaphore
            return
        # await asyncio.sleep(random.random())
        self.result *= i
        self.steps += 1
        print(f'Agent {self.id}: step {self.steps}')
        if self.steps % self.steps_per_round == 0:
            print(f'Agent {self.id}: result {self.result}')
            # self.server_semaphore.release()  # Release the semaphore
            self.event_for_server.set()
            self.event.clear()  # Reset the event
            await self.event.wait()  # Yield control to the event loop
        await self.calculate(i + 1)

class Server:
    def __init__(self, agents, server_semaphore):
        self.agents = agents
        self.semaphore = server_semaphore
        self.condition = asyncio.Condition()

    async def run(self):
        while not all(agent.done for agent in self.agents):
            # await asyncio.sleep(0.1)  # Polling delay
            await self.condition.wait_for(lambda: all((agent.steps % agent.steps_per_round == 0 or agent.done) for agent in self.agents))
            # for _ in range(len(self.agents)):
            #     print(f'Server: Waiting for agent {_}...')
            #     await self.semaphore.acquire()
            if all((agent.steps % agent.steps_per_round == 0) for agent in self.agents):
                for agent in self.agents:
                    agent.result = 1  # Reset result
                    agent.event.set()
                print('Server: Waking up agents...')


async def main():
    num_agents = 2
    server_semaphore = asyncio.Semaphore(num_agents)  # Set the initial count to the number of agents
    agents = [Factorial(i, 10+i*10, 5, server_semaphore) for i in range(num_agents)]
    server = Server(agents, server_semaphore)
    await asyncio.gather(server.run(), *(agent.calculate(1) for agent in agents))

await main()

# %%
import asyncio
import random

class Agent:
    def __init__(self, id, n, steps_per_round, condition):
        self.id = id
        self.n = n
        self.steps_per_round = steps_per_round
        self.steps = 0
        self.result = 1
        self.condition = condition
        self.ready = False
        self.done = False

    async def calculate(self):
        async with self.condition:
            for i in range(1, self.n+1):
                self.result *= i
                self.steps += 1
                print(f'Agent {self.id}: step {self.steps}')
                if self.steps % self.steps_per_round == 0:
                    print(f'Agent {self.id}: result {self.result}')
                    self.ready = True  # Set ready status
                    self.condition.notify_all()  # Notify the server
                    await self.condition.wait_for(lambda: not self.ready)  # Wait for the server
            self.done = True
            self.ready = True
            self.condition.notify_all()  # Notify the server when done

class Server:
    def __init__(self, agents, condition):
        self.agents = agents
        self.condition = condition

    async def run(self):
        async with self.condition:
            while not all(agent.done for agent in self.agents):
                await self.condition.wait_for(lambda: all((agent.ready or agent.done) for agent in self.agents))  # Wait for all agents to reach a sync point
                print('Server: Waking up agents...')
                for agent in self.agents:  # Reset all agents' ready status
                    agent.result = 1  # Reset result
                    agent.ready = False
                self.condition.notify_all()  # Wake up all agents

async def main():
    num_agents = 3
    condition = asyncio.Condition()
    agents = [Agent(i, 10+i*5, 5, condition) for i in range(num_agents)]
    server = Server(agents, condition)
    await asyncio.gather(server.run(), *(agent.calculate() for agent in agents))
asyncio.run(main())



await main()








# %%

import asyncio
import random

class Agent:
    def __init__(self, id, n, steps_per_round, condition):
        self.id = id
        self.n = n
        self.steps_per_round = steps_per_round
        self.steps = 0
        self.result = 1
        self.condition = condition
        self.ready = False
        self.done = False

    async def calculate(self):
        async with self.condition:
            for i in range(1, self.n+1):
                self.result *= i
                self.steps += 1
                print(f'Agent {self.id}: step {self.steps}')
                if self.steps % self.steps_per_round == 0:
                    print(f'Agent {self.id}: result {self.result}')
                    self.ready = True  # Set ready status
                    self.condition.notify_all()  # Notify the server
                    await self.condition.wait_for(lambda: not self.ready)  # Wait for the server
            self.done = True
            self.ready = True
            self.condition.notify_all()  # Notify the server when done

class Server:
    def __init__(self, agents, condition):
        self.agents = agents
        self.condition = condition

    async def run(self):
        async with self.condition:
            while not all(agent.done for agent in self.agents):
                await self.condition.wait_for(lambda: all((agent.ready or agent.done) for agent in self.agents))  # Wait for all agents to reach a sync point
                print('Server: Waking up agents...')
                for agent in self.agents:  # Reset all agents' ready status
                    agent.result = 1  # Reset result
                    agent.ready = False
                self.condition.notify_all()  # Wake up all agents

async def main():
    num_agents = 3
    condition = asyncio.Condition()
    agents = [Agent(i, 10+i*5, 5, condition) for i in range(num_agents)]
    server = Server(agents, condition)
    await asyncio.gather(server.run(), *(agent.calculate() for agent in agents))
# asyncio.run(main())



await main()

# %%
import asyncio
import random

class Factorial:
    def __init__(self, id, n, steps_per_round, condition):
        self.condition = condition
        self.id = id
        self.n = n
        self.steps_per_round = steps_per_round
        self.steps = 0
        self.result = 1
        self.ready = False
        self.done = False

    async def calculate(self, i):
        if i > self.n:
            self.done = True
            self.ready = True
            async with self.condition:
                self.condition.notify_all()
        else:
            # await asyncio.sleep(random.random())
            self.result *= i
            self.steps += 1
            print(f'Agent {self.id}: step {self.steps}')
            if self.steps % self.steps_per_round == 0:
                print(f'Agent {self.id}: result {self.result}')
                async with self.condition:
                    self.ready = True  # Set ready status
                    self.condition.notify_all()  # Notify the server
                    await self.condition.wait_for(lambda: not self.ready)  # Wait for the server
            await self.calculate(i + 1)

class Server:
    def __init__(self, agents, condition):
        self.agents = agents
        self.condition = condition

    async def run(self):
        async with self.condition:
            while not all(agent.done for agent in self.agents):
                await self.condition.wait_for(lambda: all((agent.ready or agent.done) for agent in self.agents))  # Wait for all agents to reach a sync point
                print('Server: Waking up agents...')
                for agent in self.agents:  # Reset all agents' ready status
                    agent.result = 1  # Reset result
                    agent.ready = False
                self.condition.notify_all()  # Wake up all agents

async def main():
    num_agents = 3
    condition = asyncio.Condition()
    agents = [Factorial(i, 5+i*5, 5, condition) for i in range(num_agents)]
    server = Server(agents, condition)
    await asyncio.gather(server.run(), *(agent.calculate(1) for agent in agents))

asyncio.run(main())
