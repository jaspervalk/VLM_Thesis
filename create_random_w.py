import numpy as np

shape = (768, 768)
W_random = np.random.randn(*shape).astype(np.float32) * 0.1  


bias = np.zeros((768,), dtype=np.float32)
mean = np.zeros((768,), dtype=np.float32)
std = np.ones((768,), dtype=np.float32)

np.savez('random_transform.npz', weights=W_random, bias=bias, mean=mean, std=std)
