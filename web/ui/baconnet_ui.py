from flask import Flask
from flask import render_template
from flask import request
from flask import Response

import requests
import pprint
import json
import os

from baconnet_predictor import Predictor

app = Flask(__name__)
# Common responses
RESPONSE_INVALID_URL = Response("Invalid URL", status=400, mimetype='text/plain')
RESPONSE_INVALID_CONTENT_TYPE = Response("Invalid Content-Type", status=400, mimetype='text/plain')
RESPONSE_INCOMPLETE_JSON = Response("Incomplete JSON", status=400, mimetype='text/plain')
RESPONSE_NOT_FOUND = Response("Not found", status=404, mimetype='text/plain')
RESPONSE_OKAY = Response(status=204)
RESPONSE_SERVER_ERROR = Response("Something bad happened", status=500)

BACONNET_SERVICE_BASE_URL = 'http://localhost:8001'

# Globally available BaconNet predictor
baconnet = Predictor()

@app.route('/hello')
def hello_world():
    """
    a simple message to show it works
    """
    return 'Hello World! I am the BaconNet UI.'


@app.route('/')
def root():
    """
    the main page
    """
    return render_template('layout.html')

@app.route('/about')
def about():
    """
    about BaconNet page
    """
    return render_template('about.html')

@app.route('/submit', methods=["POST"])
def submit():
    """
    image submission endpoint
    """
    global baconnet

    if request.headers['X_foobar'] != 'supersecret':
        return RESPONSE_INVALID_URL
    if request.headers['Content-Type'] != 'image/jpeg':
        return RESPONSE_INVALID_CONTENT_TYPE

    # Send image to service for prediction
    result = baconnet.predict(request.data)
    result = get_message(result)

    return Response(json.dumps(result), status=200, mimetype='application/json')


def get_message(result):
    """
    Add a fun message about the predicted data
    """
    p_list = [result.get("not"), result.get("bacon"), result.get("kevin")]
    p_max = 0.0
    p_index = 0
    for i, p in enumerate(p_list):
        if p > p_max:
            p_max = p
            p_index = i

    message = None
    if p_index == 0:
        message = "Seems like that's not bacon."
    elif p_index == 1:
        message = "Hey, now that looks like bacon!"
    elif p_index == 2:
        message = "Yep. Looks like Kevin Bacon, alright."

    result["message"] = message

    return result


if __name__ == '__main__':
    app.run(port=8000, debug=True)
