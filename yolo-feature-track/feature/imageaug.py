from torchvision import transforms


def transform_for_training(image_shape):
    return transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize(image_shape),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )


def transform_for_training1(image_shape):
    return transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize(image_shape),
         transforms.RandomRotation(180, resample=False, expand=False, center=None),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )


def transform_for_infer(image_shape):
    return transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize(image_shape),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
