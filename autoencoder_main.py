import os

import torch
import yaml
from data.stl10_wrap import STL10DatasetWrap
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from models.decoder import Decoder
from autoencoder_trainer import BYOLAutoencoderTrainer

print(torch.__version__)
torch.manual_seed(0)

def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])

    train_dataset = STL10DatasetWrap(root_dir='/home/work/Datasets/', data_transform=data_transform,
                                     split='train+unlabeled')

    # encoder
    encoder = ResNet18(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            encoder.load_state_dict(load_params['encoder_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # decoder
    decoder = Decoder().to(device)

    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLAutoencoderTrainer(encoder=encoder,
                                     decoder=decoder,
                                     optimizer=optimizer,
                                     device=device,
                                     **config['trainer'])

    trainer.train(train_dataset)

if __name__ == '__main__':
    main()