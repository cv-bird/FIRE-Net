import os
import time
import math
import cv2
import torch
import logging
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim import Adam
from collections import OrderedDict

from model import build_model, weights_init
from dataset import get_loader




class Solver:
    def __init__(self, train_loader, test_loader, config, save_fold=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_fold = save_fold

        # Normalization mean
        self.mean = torch.tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255.

        # Visualization (optional)
        if config.visdom:
            self.visual = Viz_visdom("trueUnify", 1)

        # Build model
        self.build_model()

        # Load weights
        if self.config.pre_trained:
            self.net.load_state_dict(torch.load(self.config.pre_trained))
        elif config.mode == "test":
            print(f"Loading pre-trained model from {self.config.model}...")
            self.net_bone.load_state_dict(torch.load(self.config.model))
            self.net_bone.eval()

        # Logging
        if config.mode == "train":
            log_dir = os.path.join(config.save_fold, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.log_output = open(os.path.join(log_dir, "log.txt"), "w")

    def print_network(self, model, name):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n{name}")
        print(model)
        print(f"The number of parameters: {num_params}")

    def get_params(self):
        """Separate params for optimizer with different learning rates"""
        param_groups = []
        for name, module in self.net_bone.named_children():
            if name == "loss_weight":
                param_groups.append({"params": module.parameters(), "lr": params["lr_branch"]})
            else:
                param_groups.append({"params": module.parameters()})
        return param_groups

    def build_model(self):
        self.net_bone = build_model(BASE_MODEL_CFG)

        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()

        self.net_bone.eval()
        self.net_bone.apply(weights_init)

        if self.config.mode == "train":
            if not self.config.load_bone:
                if BASE_MODEL_CFG == "vgg":
                    self.net_bone.base.load_pretrained_model(torch.load(self.config.vgg))
                elif BASE_MODEL_CFG == "resnet":
                    self.net_bone.base.load_state_dict(torch.load(self.config.resnet))
            else:
                self.net_bone.load_state_dict(torch.load(self.config.load_bone))

        self.lr_bone = params["lr_bone"]
        self.optimizer_bone = Adam(
            filter(lambda p: p.requires_grad, self.net_bone.parameters()),
            lr=self.lr_bone,
            weight_decay=params["wd"]
        )
        self.print_network(self.net_bone, "trueUnify bone part")

    def update_lr(self, rate):
        for param_group in self.optimizer_bone.param_groups:
            param_group["lr"] *= rate


    def test(self, test_mode=0):
        img_num = len(self.test_loader)
        time_total = 0.0
        save_dir = os.path.join(self.save_fold, "EGNet_ResNet50")
        os.makedirs(save_dir, exist_ok=True)

        for i, data_batch in enumerate(self.test_loader):
            self.config.test_fold = self.save_fold
            images_, name, im_size = data_batch["image"], data_batch["name"][0], np.asarray(data_batch["size"])

            with torch.no_grad():
                images = images_.cuda() if self.config.cuda else images_
                start_time = time.time()
                up_edge, up_sal, up_sal_f = self.net_bone(images)
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                time_total += elapsed

                pred = torch.sigmoid(up_sal_f[-1]).cpu().numpy().squeeze()
                multi_fuse = (pred * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(save_dir, f"{name[:-4]}.png"), multi_fuse)

        print(f"--- {time_total:.4f} seconds ---")
        print("Test Done!")


    def train(self):
        iter_per_epoch = len(self.train_loader.dataset) // self.config.batch_size
        ave_grad = 0

        os.makedirs(TMP_PATH, exist_ok=True)

        for epoch in range(self.config.epoch):
            r_edge_loss, r_sal_loss, r_sum_loss = 0, 0, 0
            self.net_bone.zero_grad()

            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label, sal_edge = data_batch["sal_image"], data_batch["sal_label"], data_batch["sal_edge"]

                if sal_image.size()[2:] != sal_label.size()[2:]:
                    print("Skip this batch due to size mismatch")
                    continue

                if self.config.cuda:
                    sal_image, sal_label, sal_edge = sal_image.cuda(), sal_label.cuda(), sal_edge.cuda()

                up_edge, up_sal, up_sal_f = self.net_bone(sal_image)

                # Edge loss
                edge_loss = sum(bce2d_new(x, sal_edge, reduction="sum") for x in up_edge)
                edge_loss /= (N_AVE_GRAD * self.config.batch_size)
                r_edge_loss += edge_loss.item()

                # Saliency loss
                sal_loss1 = sum(F.binary_cross_entropy_with_logits(x, sal_label, reduction="sum") for x in up_sal)
                sal_loss2 = sum(F.binary_cross_entropy_with_logits(x, sal_label, reduction="sum") for x in up_sal_f)
                sal_loss = (sal_loss1 + sal_loss2) / (N_AVE_GRAD * self.config.batch_size)

                r_sal_loss += sal_loss.item()
                loss = sal_loss + edge_loss
                r_sum_loss += loss.item()

                loss.backward()
                ave_grad += 1

                if ave_grad % N_AVE_GRAD == 0:
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()
                    ave_grad = 0

                # Logging
                if i % SHOW_EVERY == 0:
                    print(f"epoch: [{epoch+1}/{self.config.epoch}], "
                          f"iter: [{i}/{iter_per_epoch}]  ||  "
                          f"Edge : {r_edge_loss*(N_AVE_GRAD*self.config.batch_size)/SHOW_EVERY:.4f}  ||  "
                          f"Sal : {r_sal_loss*(N_AVE_GRAD*self.config.batch_size)/SHOW_EVERY:.4f}  ||  "
                          f"Sum : {r_sum_loss*(N_AVE_GRAD*self.config.batch_size)/SHOW_EVERY:.4f}")
                    print(f"Learning rate: {self.lr_bone}")
                    r_edge_loss = r_sal_loss = r_sum_loss = 0

                # Save visualizations
                if i % 200 == 0:
                    vutils.save_image(torch.sigmoid(up_sal_f[-1].data),
                                      f"{TMP_PATH}/iter{i}-sal-0.jpg", normalize=True, padding=0)
                    vutils.save_image(sal_image.data, f"{TMP_PATH}/iter{i}-sal-data.jpg", padding=0)
                    vutils.save_image(sal_label.data, f"{TMP_PATH}/iter{i}-sal-target.jpg", padding=0)

            # Save checkpoints
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(),
                           f"{self.config.save_fold}/models/epoch_{epoch+1}_bone.pth")

            if epoch in LR_DECAY_EPOCHS:
                self.lr_bone *= 0.1
                self.optimizer_bone = Adam(
                    filter(lambda p: p.requires_grad, self.net_bone.parameters()),
                    lr=self.lr_bone,
                    weight_decay=params["wd"]
                )

        torch.save(self.net_bone.state_dict(), f"{self.config.save_fold}/models/final_bone.pth")



def bce2d_new(input, target, reduction=None):
    assert input.size() == target.size()
    pos = (target == 1).float()
    neg = (target == 0).float()

    num_pos = pos.sum()
    num_neg = neg.sum()
    num_total = num_pos + num_neg + EPSILON

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)
