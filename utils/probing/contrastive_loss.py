import torch
import torch.nn as nn
import torch.nn.functional as F



class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.c_entropy = nn.CrossEntropyLoss()

    @staticmethod
    def mask_diagonal(similarities: torch.Tensor) -> torch.Tensor:
        return similarities[~torch.eye(similarities.shape[0], dtype=bool)].reshape(
            similarities.shape[0], -1
        )

    def get_teacher_distribution(self, teacher_similarities: torch.Tensor) -> torch.Tensor:
        return F.softmax(
            self.mask_diagonal(teacher_similarities) / self.temperature, dim=-1
        )

    def cross_entropy_loss(
        self, teacher_similarities: torch.Tensor, student_similarities: torch.Tensor
    ) -> torch.Tensor:
        p = self.get_teacher_distribution(teacher_similarities)
        q_unnormalized = self.mask_diagonal(student_similarities) / self.temperature
        return self.c_entropy(q_unnormalized, p)

    def forward(
        self, teacher_similarities: torch.Tensor, student_similarities: torch.Tensor
    ) -> torch.Tensor:
        return self.cross_entropy_loss(teacher_similarities, student_similarities)
