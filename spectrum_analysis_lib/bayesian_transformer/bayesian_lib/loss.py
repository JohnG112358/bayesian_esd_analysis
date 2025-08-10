import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectrum_analysis_lib.bayesian_transformer.bayesian_lib.linear import BayesLinear


class KLLoss(nn.Module):
    def __init__(self, method='analytic', reduction='mean', last_layer_only=False, num_samples=10):
        super(KLLoss, self).__init__()
        
        if method not in ['analytic', 'sampling']:
            raise ValueError(f"Method must be one of ['analytic', 'sampling'], but got {method}")
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"Reduction must be one of ['mean', 'sum'], but got {reduction}")
        
        self.method = method
        self.reduction = reduction
        self.last_layer_only = last_layer_only
        self.num_samples = num_samples
        
        
    def forward(self, model):
        kl_sum = 0.0
        num_params = 0
        
        bayesian_modules = [m for m in model.modules() if isinstance(m, BayesLinear)]
        
        if self.last_layer_only:
            modules_to_process = [bayesian_modules[-1]] if bayesian_modules else []
        else:
            modules_to_process = bayesian_modules
            
        kl_calculator = self._kl_analytic if self.method == 'analytic' else self._kl_sampling
        
        for layer in modules_to_process:
            kl_sum += kl_calculator(layer)
            
            num_params += layer.weight_mu.numel()
            if layer.bias:
                num_params += layer.bias_mu.numel()
        
        if num_params == 0:
            print("KLLoss: No Bayesian parameters found - returning 0")
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        if self.reduction == 'mean':
            return kl_sum / num_params
        else:
            return kl_sum


    def _kl_analytic(self, layer):
        # Add a small epsilon for numerical stability
        epsilon = 1e-8
        
        mu_q_w = layer.weight_mu
        sigma_q_w = self._calculate_sigma(layer, layer.weight_raw_sigma) + epsilon
        total_kl = self._kl_formula(mu_q_w, sigma_q_w, layer.prior_mu, layer.prior_sigma).sum()

        if layer.bias:
            mu_q_b = layer.bias_mu
            sigma_q_b = self._calculate_sigma(layer, layer.bias_raw_sigma) + epsilon
            kl_bias = self._kl_formula(mu_q_b, sigma_q_b, layer.prior_mu, layer.prior_sigma).sum()
            total_kl += kl_bias
        
        return total_kl
    
    
    def _kl_sampling(self, layer):
        epsilon = 1e-8 
        
        mu_q_w = layer.weight_mu
        sigma_q_w = self._calculate_sigma(layer, layer.weight_raw_sigma) + epsilon
        
        # [num_samples, out_features, in_features]
        mu_q_w_expanded = mu_q_w.unsqueeze(0).expand(self.num_samples, -1, -1)
        sigma_q_w_expanded = sigma_q_w.unsqueeze(0).expand(self.num_samples, -1, -1)
        
        # Sample weights (w) from the posterior distribution q using the reparameterization trick
        w = mu_q_w_expanded + sigma_q_w_expanded * torch.randn_like(mu_q_w_expanded)
        
        log_q_w = self._log_prob(w, mu_q_w_expanded, sigma_q_w_expanded)
        log_p_w = self._log_prob(w, layer.prior_mu, layer.prior_sigma)
        
        kl_weights = (log_q_w - log_p_w).sum() / self.num_samples
        
        total_kl = kl_weights
        if layer.bias:
            mu_q_b = layer.bias_mu
            sigma_q_b = self._calculate_sigma(layer, layer.bias_raw_sigma) + epsilon
            
            mu_q_b_expanded = mu_q_b.unsqueeze(0).expand(self.num_samples, -1)
            sigma_q_b_expanded = sigma_q_b.unsqueeze(0).expand(self.num_samples, -1)
            
            b = mu_q_b_expanded + sigma_q_b_expanded * torch.randn_like(mu_q_b_expanded)
            
            log_q_b = self._log_prob(b, mu_q_b_expanded, sigma_q_b_expanded)
            log_p_b = self._log_prob(b, layer.prior_mu, layer.prior_sigma)
            
            total_kl += (log_q_b - log_p_b).sum() / self.num_samples
        
        return total_kl
        
    
    def _log_prob(self, value, mu, sigma):
        """Log probability of a value under a Gaussian distribution."""
        return -torch.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * ((value - mu) / sigma)**2
    
    
    def _kl_formula(self, mu_q, sigma_q, mu_p, sigma_p):
        """The analytical KL divergence formula between two Gaussians."""
        return torch.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5


    def _calculate_sigma(self, layer, raw_sigma):
        """Helper to apply the correct reparameterization function."""
        if layer.reparam == 'softplus':
            return F.softplus(raw_sigma)
        elif layer.reparam == 'exp': 
            return torch.exp(raw_sigma)
        else:
            raise ValueError(f"Undefined layer reparam {layer.reparam}")
