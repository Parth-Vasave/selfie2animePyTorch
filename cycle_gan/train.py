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


import os
import time
import random
import argparse
import torch
import torch.nn as nn
from dataset import load_dataset, get_batches
from pix2pix_gan import ResnetGenerator, PatchDiscriminator, init_weights
from image_pool import ImagePool


def train(dataset, start_epoch, max_epochs, lr_d, lr_g, batch_size, lmda_cyc, lmda_idt, pool_size, device):
    torch.manual_seed(int(time.time()))

    print("Loading dataset...", flush=True)
    training_set_a = load_dataset(dataset, "trainA")
    training_set_b = load_dataset(dataset, "trainB")

    gen_ab = ResnetGenerator().to(device)
    dis_b = PatchDiscriminator().to(device)
    gen_ba = ResnetGenerator().to(device)
    dis_a = PatchDiscriminator().to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    gen_ab_path = "model/{}.gen_ab.pth".format(dataset)
    dis_b_path = "model/{}.dis_b.pth".format(dataset)
    gen_ba_path = "model/{}.gen_ba.pth".format(dataset)
    dis_a_path = "model/{}.dis_a.pth".format(dataset)
    optim_path = "model/{}.optim.pth".format(dataset)

    if os.path.isfile(gen_ab_path):
        gen_ab.load_state_dict(torch.load(gen_ab_path, map_location=device, weights_only=True))
    else:
        gen_ab.apply(init_weights)

    if os.path.isfile(dis_b_path):
        dis_b.load_state_dict(torch.load(dis_b_path, map_location=device, weights_only=True))
    else:
        dis_b.apply(init_weights)

    if os.path.isfile(gen_ba_path):
        gen_ba.load_state_dict(torch.load(gen_ba_path, map_location=device, weights_only=True))
    else:
        gen_ba.apply(init_weights)

    if os.path.isfile(dis_a_path):
        dis_a.load_state_dict(torch.load(dis_a_path, map_location=device, weights_only=True))
    else:
        dis_a.apply(init_weights)

    print("Learning rate of discriminator:", lr_d, flush=True)
    print("Learning rate of generator:", lr_g, flush=True)
    trainer_gen_ab = torch.optim.NAdam(gen_ab.parameters(), lr=lr_g, betas=(0.5, 0.999))
    trainer_dis_b = torch.optim.NAdam(dis_b.parameters(), lr=lr_d, betas=(0.5, 0.999))
    trainer_gen_ba = torch.optim.NAdam(gen_ba.parameters(), lr=lr_g, betas=(0.5, 0.999))
    trainer_dis_a = torch.optim.NAdam(dis_a.parameters(), lr=lr_d, betas=(0.5, 0.999))

    if os.path.isfile(optim_path):
        optim_state = torch.load(optim_path, map_location=device, weights_only=True)
        trainer_gen_ab.load_state_dict(optim_state["gen_ab"])
        trainer_dis_b.load_state_dict(optim_state["dis_b"])
        trainer_gen_ba.load_state_dict(optim_state["gen_ba"])
        trainer_dis_a.load_state_dict(optim_state["dis_a"])

    fake_a_pool = ImagePool(pool_size)
    fake_b_pool = ImagePool(pool_size)

    print("Training...", flush=True)
    for epoch in range(start_epoch, max_epochs):
        ts = time.time()

        random.shuffle(training_set_a)
        random.shuffle(training_set_b)

        training_dis_a_L = 0.0
        training_dis_b_L = 0.0
        training_gen_L = 0.0
        training_batch = 0

        for real_a, real_b in get_batches(training_set_a, training_set_b, batch_size, device=device):
            training_batch += 1

            with torch.no_grad():
                fake_a, _ = gen_ba(real_b)
                fake_b, _ = gen_ab(real_a)

            # Train discriminator A
            trainer_dis_a.zero_grad()
            real_a_y, real_a_cam_y = dis_a(real_a)
            real_a_L = bce_loss(real_a_y, torch.ones_like(real_a_y))
            real_a_cam_L = bce_loss(real_a_cam_y, torch.ones_like(real_a_cam_y))
            fake_a_y, fake_a_cam_y = dis_a(fake_a_pool.query(fake_a))
            fake_a_L = bce_loss(fake_a_y, torch.zeros_like(fake_a_y))
            fake_a_cam_L = bce_loss(fake_a_cam_y, torch.zeros_like(fake_a_cam_y))
            L = real_a_L + real_a_cam_L + fake_a_L + fake_a_cam_L
            L.backward()
            trainer_dis_a.step()
            dis_a_L = L.item()
            if dis_a_L != dis_a_L:
                raise ValueError()

            # Train discriminator B
            trainer_dis_b.zero_grad()
            real_b_y, real_b_cam_y = dis_b(real_b)
            real_b_L = bce_loss(real_b_y, torch.ones_like(real_b_y))
            real_b_cam_L = bce_loss(real_b_cam_y, torch.ones_like(real_b_cam_y))
            fake_b_y, fake_b_cam_y = dis_b(fake_b_pool.query(fake_b))
            fake_b_L = bce_loss(fake_b_y, torch.zeros_like(fake_b_y))
            fake_b_cam_L = bce_loss(fake_b_cam_y, torch.zeros_like(fake_b_cam_y))
            L = real_b_L + real_b_cam_L + fake_b_L + fake_b_cam_L
            L.backward()
            trainer_dis_b.step()
            dis_b_L = L.item()
            if dis_b_L != dis_b_L:
                raise ValueError()

            # Train generators
            trainer_gen_ba.zero_grad()
            trainer_gen_ab.zero_grad()

            fake_a, gen_a_cam_y = gen_ba(real_b)
            fake_a_y, fake_a_cam_y = dis_a(fake_a)
            gan_a_L = bce_loss(fake_a_y, torch.ones_like(fake_a_y))
            gan_a_cam_L = bce_loss(fake_a_cam_y, torch.ones_like(fake_a_cam_y))
            rec_b, _ = gen_ab(fake_a)
            cyc_b_L = l1_loss(rec_b, real_b)
            idt_a, idt_a_cam_y = gen_ba(real_a)
            idt_a_L = l1_loss(idt_a, real_a)
            gen_a_cam_L = bce_loss(gen_a_cam_y, torch.ones_like(gen_a_cam_y)) + bce_loss(idt_a_cam_y, torch.zeros_like(idt_a_cam_y))
            gen_ba_L = gan_a_L + gan_a_cam_L + cyc_b_L * lmda_cyc + idt_a_L * lmda_cyc * lmda_idt + gen_a_cam_L

            fake_b, gen_b_cam_y = gen_ab(real_a)
            fake_b_y, fake_b_cam_y = dis_b(fake_b)
            gan_b_L = bce_loss(fake_b_y, torch.ones_like(fake_b_y))
            gan_b_cam_L = bce_loss(fake_b_cam_y, torch.ones_like(fake_b_cam_y))
            rec_a, _ = gen_ba(fake_b)
            cyc_a_L = l1_loss(rec_a, real_a)
            idt_b, idt_b_cam_y = gen_ab(real_b)
            idt_b_L = l1_loss(idt_b, real_b)
            gen_b_cam_L = bce_loss(gen_b_cam_y, torch.ones_like(gen_b_cam_y)) + bce_loss(idt_b_cam_y, torch.zeros_like(idt_b_cam_y))
            gen_ab_L = gan_b_L + gan_b_cam_L + cyc_a_L * lmda_cyc + idt_b_L * lmda_cyc * lmda_idt + gen_b_cam_L

            L = gen_ba_L + gen_ab_L
            L.backward()
            trainer_gen_ba.step()
            trainer_gen_ab.step()
            gen_L = L.item()
            if gen_L != gen_L:
                raise ValueError()

            training_dis_a_L += dis_a_L
            training_dis_b_L += dis_b_L
            training_gen_L += gen_L
            print("[Epoch %d  Batch %d]  dis_a_loss %.10f  dis_b_loss %.10f  gen_loss %.10f  elapsed %.2fs" % (
                epoch, training_batch, dis_a_L, dis_b_L, gen_L, time.time() - ts
            ), flush=True)

        print("[Epoch %d]  training_dis_a_loss %.10f  training_dis_b_loss %.10f  training_gen_loss %.10f  duration %.2fs" % (
            epoch + 1, training_dis_a_L / training_batch, training_dis_b_L / training_batch, training_gen_L / training_batch, time.time() - ts
        ), flush=True)

        torch.save(gen_ab.state_dict(), gen_ab_path)
        torch.save(gen_ba.state_dict(), gen_ba_path)
        torch.save(dis_a.state_dict(), dis_a_path)
        torch.save(dis_b.state_dict(), dis_b_path)
        torch.save({
            "gen_ab": trainer_gen_ab.state_dict(),
            "gen_ba": trainer_gen_ba.state_dict(),
            "dis_a": trainer_dis_a.state_dict(),
            "dis_b": trainer_dis_b.state_dict(),
        }, optim_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a cycle_gan trainer.")
    parser.add_argument("--dataset", help="set the dataset used by the trainer (default: vangogh2photo)", type=str, default="vangogh2photo")
    parser.add_argument("--start_epoch", help="set the start epoch (default: 0)", type=int, default=0)
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--lr_d", help="set the learning rate of discriminator (default: 0.0003)", type=float, default=0.0003)
    parser.add_argument("--lr_g", help="set the learning rate of generator (default: 0.0001)", type=float, default=0.0001)
    parser.add_argument("--batch_size", help="set the batch size (default: 32)", type=int, default=32)
    parser.add_argument("--lmda_cyc", help="set the lambda of cycle loss (default: 10.0)", type=float, default=10.0)
    parser.add_argument("--lmda_idt", help="set the lambda of identity loss (default: 0.5)", type=float, default=0.5)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:%d" % args.device_id)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("Warning: GPU requested but not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    while True:
        try:
            train(
                dataset=args.dataset,
                start_epoch=args.start_epoch,
                max_epochs=args.max_epochs,
                lr_d=args.lr_d,
                lr_g=args.lr_g,
                batch_size=args.batch_size,
                lmda_cyc=args.lmda_cyc,
                lmda_idt=args.lmda_idt,
                pool_size=50,
                device=device
            )
            break
        except ValueError:
            print("Oops! The value of loss become NaN...")
