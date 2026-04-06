# Copyright (c) 2018-2021, RangerUFO
#
# This file is part of cycle_gan.
#
# cycle_gan is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cycle_gan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cycle_gan.  If not, see <https://www.gnu.org/licenses/>.


import io
import re
import sys
import argparse
import http.server
import cgi
import torch
import cv2
import numpy as np
from PIL import Image
from dataset import reconstruct_color
from pix2pix_gan import ResnetGenerator


class CycleGAN(http.server.BaseHTTPRequestHandler):
    _path_pattern = re.compile(r"^(/[^?\s]*)(\?\S*)?$")

    def do_POST(self):
        self._handle_request()
        sys.stdout.flush()
        sys.stderr.flush()

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Keep-Alive,User-Agent,Authorization,Content-Type")
        super().end_headers()

    @torch.no_grad()
    def _handle_request(self):
        m = self._path_pattern.match(self.path)
        if not m or m.group(0) != self.path:
            self.send_error(http.HTTPStatus.BAD_REQUEST)
            return
        if m.group(1) == "/cycle_gan/fake":
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers["Content-Type"]
                }
            )
            if "real" not in form:
                self.send_error(http.HTTPStatus.BAD_REQUEST)
                return
            buf = np.frombuffer(form["real"].value, dtype=np.uint8)
            raw = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            h, w = raw.shape[:2]
            if h < w:
                new_h, new_w = self.resize, int(w * self.resize / h)
            else:
                new_h, new_w = int(h * self.resize / w), self.resize
            raw = cv2.resize(raw, (new_w, new_h))
            tensor = torch.from_numpy(raw).permute(2, 0, 1).float() / 255.0
            tensor = (tensor - 0.5) / 0.5
            fake, _ = self.net(tensor.unsqueeze(0).to(self.device))
            out_tensor = reconstruct_color(fake[0].permute(1, 2, 0).cpu())
            img = Image.fromarray(out_tensor.numpy())
            f = io.BytesIO()
            img.save(f, format="PNG")
            out = f.getvalue()
            self.protocol_version = "HTTP/1.1"
            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Disposition", "fake.png")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        else:
            self.send_error(http.HTTPStatus.NOT_FOUND)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is CycleGAN demo server.")
    parser.add_argument("--reversed", help="reverse transformation", action="store_true")
    parser.add_argument("--model", help="set the model used by the server (default: vangogh2photo)", type=str, default="vangogh2photo")
    parser.add_argument("--resize", help="set the short size of fake image (default: 256)", type=int, default=256)
    parser.add_argument("--addr", help="set address of cycle_gan server (default: 0.0.0.0)", type=str, default="0.0.0.0")
    parser.add_argument("--port", help="set port of cycle_gan server (default: 80)", type=int, default=80)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    CycleGAN.resize = args.resize

    if args.gpu:
        if torch.cuda.is_available():
            CycleGAN.device = torch.device("cuda:%d" % args.device_id)
        elif torch.backends.mps.is_available():
            CycleGAN.device = torch.device("mps")
        else:
            CycleGAN.device = torch.device("cpu")
    else:
        CycleGAN.device = torch.device("cpu")

    print("Loading model...", flush=True)
    CycleGAN.net = ResnetGenerator(use_spectral_norm=False).to(CycleGAN.device)
    if args.reversed:
        CycleGAN.net.load_state_dict(torch.load("model/{}.gen_ba.pth".format(args.model), map_location=CycleGAN.device, weights_only=True))
    else:
        CycleGAN.net.load_state_dict(torch.load("model/{}.gen_ab.pth".format(args.model), map_location=CycleGAN.device, weights_only=True))
    CycleGAN.net.eval()

    httpd = http.server.HTTPServer((args.addr, args.port), CycleGAN)
    httpd.serve_forever()
