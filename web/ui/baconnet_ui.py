from flask import Flask
from flask import render_template
from flask import request
from flask import Response

import requests
import pprint
import json

app = Flask(__name__)
# Common responses
RESPONSE_INVALID_URL = Response("Invalid URL", status=400, mimetype='text/plain')
RESPONSE_INVALID_CONTENT_TYPE = Response("Invalid Content-Type", status=400, mimetype='text/plain')
RESPONSE_INCOMPLETE_JSON = Response("Incomplete JSON", status=400, mimetype='text/plain')
RESPONSE_NOT_FOUND = Response("Not found", status=404, mimetype='text/plain')
RESPONSE_OKAY = Response(status=204)
RESPONSE_SERVER_ERROR = Response("Something bad happened", status=500)

BACONNET_SERVICE_BASE_URL = 'http://localhost:8001'

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

@app.route('/submit', methods=["POST"])
def submit():
    """
    image submission endpoint
    """
    if request.headers['Content-Type'] != 'image/jpeg':
        return RESPONSE_INVALID_CONTENT_TYPE

    result = {"not": 0.0, "bacon": 0.0, "kevin": 0.0}
    return Response(json.dumps(result), status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(port=8000, debug=True)
