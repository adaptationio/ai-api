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


state_size = 10
action_size = 2
parser = reqparse.RequestParser()
#parser.add_argument('data1', type=list, location='json')
parser.add_argument('state', action='append')
parser.add_argument('action', action='append')
parser.add_argument('reward', action='append')
parser.add_argument('next_state', action='append')
parser.add_argument('done', action='append')

response = {
    'record': 'data recorded',
    'predict': 'pretict action'
}

transformer = DataTransforms()
eps=1
agent = Agent(state_size=state_size, action_size=action_size, seed=0, network="cnn")
# Action
#   receives an state array and returns a action
class Action(Resource):
    def get(self):
        return response['record']

    def post(self):
        args = parser.parse_args()
        state = args['state']
        state = transformer.toarray([data])
        action = agent.act(state, eps)
        print(data)
        transformer.tocsv_single(data,'test_data.csv')
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
        return args['state']

class Setup(Resource):
    def get(self):
        return response['preditc']

    def post(self):
        args = parser.parse_args()
        data = args['data']
        data = transformer.toarray([data])
        print(data)
        transformer.tocsv_single(data,'test_data.csv')
        return args['data']
##
## Actually setup the Api resource routing here
##
api.add_resource(Action, '/action')
api.add_resource(Step, '/step')
api.add_resource(Setup, '/setup')


if __name__ == '__main__':
    app.run(debug=True)