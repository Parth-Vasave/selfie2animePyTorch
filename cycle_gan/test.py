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


import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import load_image, reconstruct_color
from pix2pix_gan import ResnetGenerator, PatchDiscriminator


def preprocess(img_np, size):
    h, w = img_np.shape[:2]
    if h < w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    img_np = cv2.resize(img_np, (new_w, new_h))
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)


@torch.no_grad()
def test(images, model, is_reversed, size, device):
    print("Loading models...", flush=True)
    dis_a = PatchDiscriminator().to(device)
    dis_a.load_state_dict(torch.load("model/{}.dis_a.pth".format(model), map_location=device, weights_only=True))
    dis_a.eval()
    dis_b = PatchDiscriminator().to(device)
    dis_b.load_state_dict(torch.load("model/{}.dis_b.pth".format(model), map_location=device, weights_only=True))
    dis_b.eval()
    gen_ab = ResnetGenerator().to(device)
    gen_ab.load_state_dict(torch.load("model/{}.gen_ab.pth".format(model), map_location=device, weights_only=True))
    gen_ab.eval()
    gen_ba = ResnetGenerator().to(device)
    gen_ba.load_state_dict(torch.load("model/{}.gen_ba.pth".format(model), map_location=device, weights_only=True))
    gen_ba.eval()

    for path in images:
        print(path)
        raw = load_image(path)
        real = preprocess(raw, size).to(device)
        real_a_y, _ = dis_a(real)
        real_b_y, _ = dis_b(real)
        if is_reversed:
            fake, _ = gen_ba(real)
            rec, _ = gen_ab(fake)
        else:
            fake, _ = gen_ab(real)
            rec, _ = gen_ba(fake)
        fake_a_y, _ = dis_a(fake)
        fake_b_y, _ = dis_b(fake)
        print("Real score A:", torch.sigmoid(real_a_y).mean().item())
        print("Real score B:", torch.sigmoid(real_b_y).mean().item())
        print("Fake score A:", torch.sigmoid(fake_a_y).mean().item())
        print("Fake score B:", torch.sigmoid(fake_b_y).mean().item())
        plt.subplot(1, 3, 1)
        plt.imshow(raw)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(reconstruct_color(fake[0].permute(1, 2, 0).cpu()).numpy())
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(reconstruct_color(rec[0].permute(1, 2, 0).cpu()).numpy())
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a cycle_gan tester.")
    parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
    parser.add_argument("--reversed", help="reverse transformation", action="store_true")
    parser.add_argument("--model", help="set the model used by the tester (default: vangogh2photo)", type=str, default="vangogh2photo")
    parser.add_argument("--resize", help="set the short size of fake image (default: 256)", type=int, default=256)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:%d" % args.device_id)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    test(
        images=args.images,
        model=args.model,
        is_reversed=args.reversed,
        size=args.resize,
        device=device
    )
