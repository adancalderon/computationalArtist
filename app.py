from flask import Flask, request, send_from_directory
import random
from helpers import createArt

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/process_images', methods=['POST'])
def process_images():
    # Get the two uploaded images
    content = request.files['image1']
    style = request.files['image2']

    # Process the images using machine learning
    output = createArt()

    # Return the output as JSON data
    return {'output': 'hola papi'}