from torchvision.datasets import CIFAR100
from torchvision import transforms

transform = transforms.ToTensor()

# Load train set
dataset = CIFAR100(root="./data/cifar100", train=True, transform=transform, download=False)

print("Total samples:", len(dataset))
print("Sample fine label:", dataset.targets[0])
print("Fine label name:", dataset.classes[dataset.targets[0]])
