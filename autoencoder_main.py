import os

import torch
import yaml
from data.stl10_wrap import STL10DatasetWrap
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from models.decoder import Decoder as DecoderDCGAN
from models.decoder_resnet import Generator as DecoderResnet
from autoencoder_trainer import BYOLAutoencoderTrainer

import sys
import argparse

print(torch.__version__)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description="training script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset_root", type=str, default='/home/work/Datasets/', help="Path to root directory of dataset")
parser.add_argument("--config_path", type=str, default='config/config_autoencoder.yaml', help="Path to config")
args = parser.parse_args()

def main():
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)

    if sys.gettrace() is not None:
        config['trainer']['num_workers'] = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
    print(f"Num workers: {config['trainer']['num_workers']}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])

    train_dataset = STL10DatasetWrap(root_dir=args.dataset_root, data_transform=data_transform,
                                     split='train+unlabeled')

    # encoder
    content_encoder = ResNet18(**config['network']).to(device)

    if config['trainer']['view_input_original_image']:
        view_encoder = ResNet18(in_channels=6, **config['network']).to(device)
    else:
        view_encoder = ResNet18(in_channels=3, **config['network']).to(device)

    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            content_encoder.load_state_dict(load_params['content_encoder_state_dict'])
            view_encoder.load_state_dict(load_params['view_encoder_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # decoder
    if (not 'decoder' in config['network']) or (config['network']['decoder'] == 'dcgan'):
        decoder = DecoderDCGAN(latent_dim=256).to(device)
    else:
        decoder = DecoderResnet(z_dim=256).to(device)

    optimizer = torch.optim.SGD(list(content_encoder.parameters()) + list(view_encoder.parameters()) + list(decoder.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLAutoencoderTrainer(content_encoder=content_encoder,
                                     view_encoder=view_encoder,
                                     decoder=decoder,
                                     optimizer=optimizer,
                                     device=device,
                                     files_to_same=[args.config_path, "autoencoder_main.py", 'autoencoder_trainer.py'],
                                     **config['trainer'])

    trainer.train(train_dataset)

if __name__ == '__main__':
    main()