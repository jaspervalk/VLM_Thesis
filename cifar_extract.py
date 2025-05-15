# test_cifar.py
from torchvision.datasets import CIFAR100
from torchvision import transforms

# Transform: convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Download CIFAR-100 with fine labels
dataset = CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)

print(f"Total samples: {len(dataset)}")
print(f"Example fine label: {dataset.targets[0]}")
print(f"Classes (fine): {dataset.classes[:5]}...")  # 100 fine classes
