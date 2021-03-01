import torchvision
import torchvision.transforms as transforms


def get_model(args):
    if args.model_name == 'vits':
        from vit import ViTSmall
        net = ViTSmall(args.in_c, args.num_classes, img_size=args.size, patch=args.patch)
    else:
        raise NotImplementedError(f"{model_name} is not implemented yet...")

    return net

def get_transform(args):
    train_transform = []
    test_transform = []
    if args.dataset == 'c10' or args.dataset=='c100':
        train_transform += [
            transforms.RandomCrop(size=args.size, padding=args.padding),
            transforms.RandomHorizontalFlip()
        ]

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    root = "data"
    if args.dataset == "c10":
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "c100":
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    return experiment_name