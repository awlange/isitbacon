from flask import Flask
from flask import request
app = Flask(__name__)

from flask import Response
from flask import json

# Common responses
RESPONSE_INVALID_URL = Response("Invalid URL", status=400, mimetype='text/plain')
RESPONSE_INVALID_CONTENT_TYPE = Response("Invalid Content-Type", status=400, mimetype='text/plain')
RESPONSE_INCOMPLETE_JSON = Response("Incomplete JSON", status=400, mimetype='text/plain')
RESPONSE_NOT_FOUND = Response("Not found", status=404, mimetype='text/plain')
RESPONSE_OKAY = Response(status=204)
RESPONSE_SERVER_ERROR = Response("Something bad happened", status=500)

@app.route('/')
def hello_world():
    """
    :return: a simple message to show it works
    """
    return 'Hello World! I am the BaconNet service.'


if __name__ == '__main__':
    app.run(port=8001, debug=True)
