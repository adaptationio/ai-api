import requests
import numpy as np
from utilities import DataTransforms

class Local_Agent():
    def __init__(self, env ='moose'):
        self.love = "Ramona"
        self.server = 'http://localhost:5000/'
        self.tranformer = DataTransforms()
        self.score = None
        self.env = env

    def act(self, state, eps):
        payload = {'state': state, 'eps':eps} 
        r1 = requests.post(self.server+'action', data=payload)
        action = int(r1.text)
        action = np.int64(action)
        return action

    def step(self, state, action, reward, next_state, done):
        payload = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done} 
        r1 = requests.post(self.server+'step', data=payload)
        return r1.text

    def config(self):
        payload = {'state': [1,2,3,4,5]} 
        r1 = requests.post(self.server+'action', data=payload)


test_client = Local_Agent()

test_client.act([1,2,3,4,5,6], 1)