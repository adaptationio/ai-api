from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from utilities import DataTransforms
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
api = Api(app)
import numpy as np
import json
from agent import Agent
from flask import jsonify


state_size = 6
action_size = 2
parser = reqparse.RequestParser()
#parser.add_argument('data1', type=list, location='json')
parser.add_argument('state', action='append')
parser.add_argument('eps', type=int)
parser.add_argument('action', type=int)
parser.add_argument('reward', type=int)
parser.add_argument('next_state', action='append')
parser.add_argument('done', action='append')
parser.add_argument('state_size', type=int)
parser.add_argument('action_size', type=int)
parser.add_argument('seed', type=int)
parser.add_argument('network')
parser.add_argument('load_network', type=int)


response = {
    'action': 'data recorded',
    'setup': 'pretict action',
    'step': "step complete"
}

transformer = DataTransforms()
#eps=1
agent = Agent(state_size=state_size, action_size=action_size, seed=0, network="cnn")
# Action
#   receives an state array and returns a action
class Action(Resource):
    def get(self):
        return response['record']

    def post(self):
        args = parser.parse_args()
        state = args['state']
        eps = args['eps']
        state = transformer.toarray([state])
        action = agent.act(state, eps)
        action = int(action)
        transformer.tocsv_single(state,'test_data.csv')
        return action

class Step(Resource):
    def get(self):
        return response['preditc']

    def post(self):
        args = parser.parse_args()
        state = args['state']
        action = args['action']
        reward = args['reward']
        next_state = args['next_state']
        done = args['done']
        data = transformer.toarray([data])
        agent.step(state, action, reward, next_state, done)
        print(data)
        transformer.tocsv_single(data,'test_data.csv')
        return response['step']

class Setup(Resource):
    def get(self):
        return response['preditc']

    def post(self):
        args = parser.parse_args()
        state_size = args['state_size']
        action_size = args['action_size']
        seed = args['seed']
        network = args['network']
        load_network = args['load_network']
        agent = Agent(state_size=state_size, action_size=action_size, seed=0, network="cnn")
        return response['step']
##
## Actually setup the Api resource routing here
##
api.add_resource(Action, '/action')
api.add_resource(Step, '/step')
api.add_resource(Setup, '/setup')


if __name__ == '__main__':
    app.run(debug=True)