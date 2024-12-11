# Design is inspired by Clear Code Channel on Youtube (which in Photo Editor part in the video)
# Link Video -> https://youtu.be/mop6g-c5HEY?si=u6JdVndQdj38aSkE

from flask import Flask, render_template, request
import process as prc
import numpy as np
import sympy as sp
from PIL import Image
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
import threading

app = Flask(__name__)
app.static_folder = "static"

@app.route("/enc", methods=["GET"])
def home():
    return render_template("encryption.html")

@app.route("/dec", methods=["GET"])
def decryption():
    return render_template("decryption.html")

@app.route("/encrypt", methods=["POST"])
def encrypt():
    image = request.files["image"]
    key = request.form["key"]
    key = [int(x) for x in key.split(',')]
    original_image_path = "static/image/" + image.filename

    encryp = prc.hill_cipher_encrypt(original_image_path, key)
    crypted_image_path = "static/image/encrypted_image.png"

    mse, psnr, npcr, uaci, nc, crypted_entropy, original_entropy, encryption_quality,histogram_path, crypted_histogram_path= prc.evaluate(original_image_path, crypted_image_path)
    return render_template(
        "encryption.html",
        encrypt1=encryp,
        mse_value=mse,
        psnr_value=psnr,
        npcr_value=npcr,
        uaci_value=uaci,
        nc_value=nc,
        crypted_entropy_value=crypted_entropy,
        original_entropy_value= original_entropy,
        encryption_quality_value = encryption_quality,
        histogram=histogram_path,
        c_histogram=crypted_histogram_path
        
    )

@app.route("/decrypt", methods=["POST"])
def decrypt():
    image = request.files["image"]
    key = request.form["key"]
    key = [int(x) for x in key.split(',')]
    encrypted_image_path = "static/image/" + image.filename

    decryp = prc.hill_cipher_decrypt(encrypted_image_path, key)
    crypted_image_path = "static/image/decrypted_image.png"

    mse, psnr, npcr, uaci, nc, crypted_entropy, original_entropy, encryption_quality, histogram_path, crypted_histogram_path   = prc.evaluate(encrypted_image_path, crypted_image_path)

    return render_template(
        "decryption.html",
        decrypt1=decryp,
        mse_value=mse,
        psnr_value=psnr,
        npcr_value=npcr,
        uaci_value=uaci,
        nc_value=nc,
        crypted_entropy_value=crypted_entropy,
        original_entropy_value= original_entropy,
        encryption_quality_value = encryption_quality,
        historgam= histogram_path,
        c_histogram=crypted_histogram_path

        
    )
if __name__ == "__main__":
    app.run()
