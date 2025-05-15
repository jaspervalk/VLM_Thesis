from data.cifar100_custom import CIFAR100Custom
from torchvision import transforms

transform = transforms.ToTensor()

dataset = CIFAR100Custom(root="./data/cifar100", train=True, transform=transform, coarse=True)

print("Coarse label:", dataset[0][1])  # shows 0–19
