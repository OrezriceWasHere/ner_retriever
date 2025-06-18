import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):

	def __init__(self, margin=1.0, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.margin = margin
		self.criterion = lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1)


	def forward(self,query, positive_key, negative_keys):
		triplet_loss = F.triplet_margin_with_distance_loss(
			query, positive_key, negative_keys,
			distance_function=self.criterion,
			margin=self.margin
		)
		return triplet_loss


