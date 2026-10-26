import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

img_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

mask_tf = transforms.Compose([
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor()
])
def get_loaders(batch_size = 8, root='data'):
    train_ds = datasets.OxfordIIITPet(
        root=root,
        download=True,
        split='trainval',
        transform=img_tf,
        target_transform=mask_tf,
        target_types='segmentation'
    )

    val_ds = datasets.OxfordIIITPet(
        root=root,
        download=True,
        split='test',
        transform=img_tf,
        target_transform=mask_tf,
        target_types='segmentation'
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
