import os

import torch
import yaml
from data.stl10_wrap import STL10DatasetWrap
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from models.decoder import Decoder
from autoencoder_trainer import BYOLAutoencoderTrainer

import argparse

print(torch.__version__)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description="training script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset_root", type=str, default='/home/work/Datasets/', help="Path to root directory of dataset")
args = parser.parse_args()

def main():
    config = yaml.load(open("./config/config_autoencoder.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])

    train_dataset = STL10DatasetWrap(root_dir=args.dataset_root, data_transform=data_transform,
                                     split='train+unlabeled')

    # encoder
    content_encoder = ResNet18(**config['network']).to(device)
    view_encoder = ResNet18(**config['network']).to(device)
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
    decoder = Decoder(latent_dim=256).to(device)

    optimizer = torch.optim.SGD(list(content_encoder.parameters()) + list(view_encoder.parameters()) + list(decoder.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLAutoencoderTrainer(content_encoder=content_encoder,
                                     view_encoder=view_encoder,
                                     decoder=decoder,
                                     optimizer=optimizer,
                                     device=device,
                                     **config['trainer'])

    trainer.train(train_dataset)

if __name__ == '__main__':
    main()