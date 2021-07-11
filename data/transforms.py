from torchvision.transforms import transforms
from data.gaussian_blur import GaussianBlur


def get_simclr_data_transforms(input_shape, s=1,
                               transforms_choice=['gray_scale']):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

    transform_options = {
        'random_crop': transforms.RandomResizedCrop(size=eval(input_shape)[0]),
        'horizontal_flip': transforms.RandomHorizontalFlip(),
        'color_jitter': transforms.RandomApply([color_jitter], p=0.8),
        'gray_scale': transforms.RandomGrayscale(p=0.2), # Original p=0.2
        'gaussian_blur': GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
        'random_rotate': transforms.RandomRotation(45)
    }

    transforms_list = [transform_options[x] for x in transforms_choice]

    # get a set of data augmentation transformations as described in the SimCLR paper.
    data_transforms = transforms.Compose(transforms_list)

    return data_transforms