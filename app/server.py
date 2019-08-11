from os import environ
from fastai import *
from fastai.text import *
from flask_api import FlaskAPI
from flask import request as req
import os 

defaults.device = torch.device('cpu')

dir_path = os.path.dirname(os.path.realpath(__file__))
path = Path(dir_path + '/models/autocomplete/trump')
learn = load_learner(path)

app = FlaskAPI(__name__)

@app.route('/autocomplete')
def autocomplete():
    body = req.args.get('body')
    words = int(req.args.get('words')) if 'words' in req.args else 1
    response = learn.predict(body, words, temperature=0.75)
    return { "input": body, "words": words, "output": response }
    
if __name__ == "__main__":
    app.run(host='0.0.0.0')