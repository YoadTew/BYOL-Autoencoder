import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys

from utils import _create_model_training_folder

def log_info(text):
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'{dt_string} | {text}')
    sys.stdout.flush()

class BYOLAutoencoderTrainer:
    def __init__(self, content_encoder, view_encoder, decoder, optimizer, device, **params):
        self.content_encoder = content_encoder
        self.view_encoder = view_encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "autoencoder_main.py", 'autoencoder_trainer.py'])

        self.reconstruction_criterion = nn.MSELoss()
        self.content_loss_weight = params['content_loss_weight']
        self.view_loss_weight = params['view_loss_weight']
        self.reconstruction_loss_weight = params['reconstruction_loss_weight']
        self.use_cross_decode = params['use_cross_decode']

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def reconstruction_loss(self, t1_x1, t2_x1, t1_x2, t2_x2,
                            e1_t1_x1, e1_t2_x1, e1_t1_x2, e1_t2_x2,
                            e2_t1_x1, e2_t2_x1, e2_t1_x2, e2_t2_x2, batch_idx, epoch_counter):

        if self.use_cross_decode:
            # compute generated images
            D_t2_x1 = self.decoder(torch.cat([e1_t1_x1, e2_t2_x2], dim=1))
            D_t1_x2 = self.decoder(torch.cat([e1_t2_x2, e2_t1_x1], dim=1))
            # D_t2_x2 = self.decoder(torch.cat([e1_t1_x2, e2_t2_x1], dim=1))
            # D_t1_x1 = self.decoder(torch.cat([e1_t2_x1, e2_t1_x2], dim=1))

            if batch_idx == 0:
                grid = torchvision.utils.make_grid(D_t2_x1[:32])
                self.writer.add_image('Decoded_t2_x1', grid, global_step=epoch_counter)

                grid = torchvision.utils.make_grid(D_t1_x2[:32])
                self.writer.add_image('Decoded_t1_x2', grid, global_step=epoch_counter)

            loss = self.reconstruction_criterion(D_t2_x1, t2_x1)
            loss += self.reconstruction_criterion(D_t1_x2, t1_x2)
        else:
            D_t1_x1 = self.decoder(torch.cat([e1_t1_x1, e2_t1_x1], dim=1))
            D_t2_x2 = self.decoder(torch.cat([e1_t2_x2, e2_t2_x2], dim=1))

            if batch_idx == 0:
                grid = torchvision.utils.make_grid(D_t1_x1[:32])
                self.writer.add_image('Decoded_t1_x1', grid, global_step=epoch_counter)

                grid = torchvision.utils.make_grid(D_t2_x2[:32])
                self.writer.add_image('Decoded_t2_x2', grid, global_step=epoch_counter)

            loss = self.reconstruction_criterion(D_t1_x1, t1_x1)
            loss += self.reconstruction_criterion(D_t2_x2, t2_x2)

        return loss

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        for epoch_counter in range(self.max_epochs):
            for batch_idx, (batch_img1_view1, batch_img1_view2, batch_img2_view1, batch_img2_view2) in enumerate(train_loader):

                batch_img1_view1 = batch_img1_view1.to(self.device)
                batch_img1_view2 = batch_img1_view2.to(self.device)
                batch_img2_view1 = batch_img2_view1.to(self.device)
                batch_img2_view2 = batch_img2_view2.to(self.device)

                if batch_idx == 0:
                    grid = torchvision.utils.make_grid(batch_img1_view1[:32])
                    self.writer.add_image('img_1_views_1', grid, global_step=epoch_counter)

                    grid = torchvision.utils.make_grid(batch_img1_view2[:32])
                    self.writer.add_image('img_1_views_2', grid, global_step=epoch_counter)

                    grid = torchvision.utils.make_grid(batch_img2_view1[:32])
                    self.writer.add_image('img_2_views_1', grid, global_step=epoch_counter)

                    grid = torchvision.utils.make_grid(batch_img2_view2[:32])
                    self.writer.add_image('img_2_views_2', grid, global_step=epoch_counter)

                content_loss, view_loss, reconstruction_loss = self.update(batch_img1_view1, batch_img1_view2,
                                                                           batch_img2_view1, batch_img2_view2,
                                                                           batch_idx, epoch_counter)
                loss = self.content_loss_weight * content_loss + \
                       self.view_loss_weight * view_loss + \
                       self.reconstruction_loss_weight * reconstruction_loss

                self.writer.add_scalar('Content loss', content_loss, global_step=niter)
                self.writer.add_scalar('View loss', view_loss, global_step=niter)
                self.writer.add_scalar('Reconstruction loss', reconstruction_loss, global_step=niter)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                niter += 1

            log_info("End of epoch {}".format(epoch_counter))

            if (epoch_counter + 1) % 5 == 0:
                self.save_model(os.path.join(model_checkpoints_folder, f'model_{epoch_counter+1}.pth'))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model_last.pth'))

    def update(self, t1_x1, t2_x1, t1_x2, t2_x2, batch_idx, epoch_counter):
        # e1 - Content encoder, e2 - view encoder, t_i - augmentation, x_i - image

        # compute content features
        e1_t1_x1, e1_t2_x1 = self.content_encoder(torch.cat([t1_x1, t2_x1])).view(2, self.batch_size, -1)
        e1_t1_x2, e1_t2_x2 = self.content_encoder(torch.cat([t1_x2, t2_x2])).view(2, self.batch_size, -1)

        # compute augmentation features
        e2_t1_x1, e2_t2_x1 = self.view_encoder(torch.cat([t1_x1, t2_x1])).view(2, self.batch_size, -1)
        e2_t1_x2, e2_t2_x2 = self.view_encoder(torch.cat([t1_x2, t2_x2])).view(2, self.batch_size, -1)

        # compute generated images
        reconstruction_loss = self.reconstruction_loss(
            t1_x1, t2_x1, t1_x2, t2_x2,
            e1_t1_x1, e1_t2_x1, e1_t1_x2, e1_t2_x2,
            e2_t1_x1, e2_t2_x1, e2_t1_x2, e2_t2_x2, batch_idx, epoch_counter)

        # Content loss
        content_loss = self.regression_loss(e1_t1_x1, e1_t2_x1)
        content_loss += self.regression_loss(e1_t1_x2, e1_t2_x2)
        content_loss = content_loss.mean()

        view_loss = self.regression_loss(e2_t1_x1, e2_t1_x2)
        view_loss += self.regression_loss(e2_t2_x1, e2_t2_x2)
        view_loss = view_loss.mean()

        return content_loss, view_loss, reconstruction_loss

    def save_model(self, PATH):
        torch.save({
            'content_encoder_state_dict': self.content_encoder.state_dict(),
            'view_encoder_state_dict': self.view_encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)