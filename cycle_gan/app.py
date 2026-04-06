"""Web dashboard for Selfie2Anime style transfer."""

import io
import argparse
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from style_generator import Generator

app = Flask(__name__, template_folder="templates", static_folder="static")

model = None
device = None


def load_model(device_arg):
    global model, device
    device = device_arg
    model = Generator().to(device)
    model.load_state_dict(torch.load(
        "model/selfie2anime.gen_ab.pth",
        map_location=device,
        weights_only=True
    ))
    model.eval()
    print("Model loaded successfully!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    # Resize keeping aspect ratio, short side = 512
    w, h = img.size
    if h < w:
        new_h, new_w = 512, int(w * 512 / h)
    else:
        new_h, new_w = int(h * 512 / w), 512
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # To tensor [-1, 1]
    arr = np.array(img).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)

    # To image
    out_arr = ((output[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
    out_arr = out_arr.clip(0, 255).astype(np.uint8)
    out_img = Image.fromarray(out_arr)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png", download_name="anime_output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selfie2Anime Web Dashboard")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device("cpu")

    load_model(dev)
    print(f"Starting server on http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
