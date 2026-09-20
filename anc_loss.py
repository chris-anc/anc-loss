import torch
import torch.nn as nn
import torch.nn.functional as F

class ANCLoss(nn.Module):
    """
    Asymptotic Non-Closure (ANC) Loss Module.
    
    Formalizes Epistemic Non-Closure by enforcing an irreducible entropy floor (Omega_M)
    via an inverse-square loss barrier, preventing AI model optimization from achieving 
    epistemic closure or neutralizing environment/human variance.
    """
    def __init__(self, omega_m: float = 0.1, alpha: float = 1.0, beta: float = 0.1):
        super(ANCLoss, self).__init__()
        self.omega_m = omega_m  # Inviolable Mystery Constant (Omega_M > 0)
        self.alpha = alpha      # Weight for Epistemic Humility Barrier
        self.beta = beta        # Weight for Causal Heritage Anchor
        
    def forward(self, pred_logits: torch.Tensor, target_labels: torch.Tensor, human_baseline_dist: torch.Tensor = None) -> torch.Tensor:
        # 1. Standard Task Loss (Empirical Error)
        task_loss = F.cross_entropy(pred_logits, target_labels)
        
        # 2. Predictive Shannon Entropy H(P_theta)
        probs = F.softmax(pred_logits, dim=-1)
        log_probs = F.log_softmax(pred_logits, dim=-1)
        shannon_entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        
        # 3. Inverse-Square Epistemic Humility Barrier
        # Penalizes model as predictive entropy approaches or drops below Omega_M
        entropy_gap = torch.clamp(shannon_entropy - self.omega_m, min=1e-5)
        humility_barrier = self.alpha / (entropy_gap ** 2)
        
        # 4. Causal Heritage Anchor (KL-Divergence from organic human baseline)
        if human_baseline_dist is not None:
            heritage_loss = F.kl_div(log_probs, human_baseline_dist, reduction='batchmean')
        else:
            heritage_loss = torch.tensor(0.0, device=pred_logits.device)
            
        # Total ANC Objective
        total_loss = task_loss + humility_barrier + (self.beta * heritage_loss)
        return total_loss
