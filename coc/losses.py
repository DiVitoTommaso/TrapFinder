from environment import *

class FocalRegressionLoss(nn.Module):
    def __init__(self, base_loss="smooth_l1", gamma=2.0, reduction="mean", eps=1e-4):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

        if base_loss not in ["l1", "mse", "smooth_l1"]:
            raise ValueError("base_loss must be 'l1', 'mse', or 'smooth_l1'")
        self.base_loss_name = base_loss

    def forward(self, preds, targets):
        """
        preds, targets: (batch_size, N, 2) tensors of 2D points
        """
        batch_size, N, _ = preds.shape
        all_losses = []

        for b in range(batch_size):
            p = preds[b]  # (N,2)
            t = targets[b]  # (N,2)

            # Compute pairwise distances: (N, N)
            dists = torch.cdist(p, t, p=2)  # Euclidean distance

            # For each predicted point, find nearest target
            min_dists, _ = dists.min(dim=1)  # (N,)

            # Compute base loss
            if self.base_loss_name == "l1":
                base = min_dists
            elif self.base_loss_name == "mse":
                base = min_dists ** 2
            else:  # smooth L1
                base = F.smooth_l1_loss(min_dists, torch.zeros_like(min_dists), reduction="none")

            # Apply gamma weighting
            weights = (base + self.eps).pow(self.gamma)
            loss = weights * base
            all_losses.append(loss)

        # Concatenate all batch losses
        all_losses = torch.cat(all_losses)

        # Reduction
        if self.reduction == "mean":
            return all_losses.mean()
        elif self.reduction == "sum":
            return all_losses.sum()
        return all_losses


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        B, N, _ = preds.shape
        _, M, _ = targets.shape

        # Compute pairwise distance: (B, N, M)
        preds_exp = preds.unsqueeze(2).expand(B, N, M, 2)
        targets_exp = targets.unsqueeze(1).expand(B, N, M, 2)
        dist = torch.norm(preds_exp - targets_exp, dim=-1, p=2)

        # For each predicted point, find nearest target point
        min_dist_pred_to_target, _ = dist.min(dim=2)
        # For each target point, find nearest predicted point
        min_dist_target_to_pred, _ = dist.min(dim=1)

        loss = min_dist_pred_to_target.mean(dim=1) + min_dist_target_to_pred.mean(dim=1)
        return loss.mean()
