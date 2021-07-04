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
    def __init__(self, encoder, decoder, optimizer, device, **params):
        self.encoder = encoder
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

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        for epoch_counter in range(self.max_epochs):
            for (batch_view_1, batch_view_2, batch_other) in train_loader:

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                batch_other = batch_other.to(self.device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_other[:32])
                    self.writer.add_image('views_other', grid, global_step=niter)

                content_loss, reconstruction_loss = self.update(batch_view_1, batch_view_2)
                loss = content_loss + 10. * reconstruction_loss
                self.writer.add_scalar('Content loss', content_loss, global_step=niter)
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

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        projection_view_1 = self.encoder(batch_view_1)
        projection_view_2 = self.encoder(batch_view_2)

        generated_view_1 = self.decoder(projection_view_1)
        generated_view_2 = self.decoder(projection_view_2)

        content_loss = self.regression_loss(projection_view_1, projection_view_2)
        content_loss = content_loss.mean()

        reconstruction_loss = self.reconstruction_criterion(generated_view_1, batch_view_1)
        reconstruction_loss += self.reconstruction_criterion(generated_view_2, batch_view_2)
        reconstruction_loss = reconstruction_loss.mean()

        return content_loss, reconstruction_loss

    def save_model(self, PATH):
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)