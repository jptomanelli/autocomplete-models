from os import environ
from fastai import *
from fastai.text import *
from flask_api import FlaskAPI
from flask import request as req

defaults.device = torch.device('cpu')

path = Path('./models/autocomplete/trump')
learn = load_learner(path)

app = FlaskAPI(__name__)

@app.route('/autocomplete')
def autocomplete():
    body = req.args.get('body')
    words = int(req.args.get('words')) if 'words' in req.args else 1
    response = learn.predict(body, words, temperature=0.75)
    return response
    
if __name__ == "__main__":
    app.run(environ.get('PORT'))