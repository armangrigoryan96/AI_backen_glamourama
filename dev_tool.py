import base64
from flask import Flask, request, jsonify, send_file
import os
from PIL import Image

from tryon_dress.utils import image_to_base64

app = Flask(__name__)

@app.route("/dev_tool", methods = ["GET"])
def dev_tool():
    filename = request.form.get('name')
    img = Image.open(filename)
    img_base64 = image_to_base64(img)
    return jsonify({"image": img_base64, "name": filename})

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
