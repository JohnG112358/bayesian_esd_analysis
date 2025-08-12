import math
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F


class BayesLinear(Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, initial_sigma, in_features, out_features, bias=True, reparam='softplus'):
        super().__init__()
        
        if not isinstance(prior_mu, (int, float)) or not math.isfinite(prior_mu):
            raise ValueError(f"prior_mu must be a finite number, but got {prior_mu}")
        if not isinstance(prior_sigma, (int, float)) or not math.isfinite(prior_sigma) or prior_sigma <= 0:
            raise ValueError(f"prior_sigma must be a positive finite number, but got {prior_sigma}")
        if not isinstance(initial_sigma, (int, float)) or not math.isfinite(initial_sigma) or initial_sigma <= 0:
            raise ValueError(f"initial_sigma must be a positive finite number, but got {initial_sigma}")       
        if reparam not in ['softplus', 'exp']:
            raise ValueError(f"Reparameterization function must be one of ['softplus', 'exp'], got {reparam}")
        
        self.reparam = reparam
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer("initial_sigma", torch.tensor(float(initial_sigma), dtype=torch.get_default_dtype()).clamp_min(1e-12))
        self.register_buffer("prior_mu", torch.tensor(float(prior_mu), dtype=torch.get_default_dtype()))
        self.register_buffer("prior_sigma", torch.tensor(float(prior_sigma), dtype=torch.get_default_dtype()).clamp_min(1e-12))
        
        # Weight parameters
        self.weight_mu = Parameter(torch.empty(out_features, in_features))
        self.weight_raw_sigma = Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_eps', None)
        
        # Bias parameters
        self.bias = bias
        if self.bias:
            self.bias_mu = Parameter(torch.empty(out_features))
            self.bias_raw_sigma = Parameter(torch.empty(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_raw_sigma', None)
            self.register_buffer('bias_eps', None)
            
        self.reset_parameters()


    def reset_parameters(self):
        """
        Initializes the parameters of the layer. The raw_sigma parameters are initialized
        with the inverse softplus or log of the initial_sigma 
        """        
        initial_raw_sigma = self._inv_sigma(self.initial_sigma).item()
        
        # Initialize weights
        stdv = 1. / math.sqrt(self.in_features)
        with torch.no_grad():
            self.weight_mu.uniform_(-stdv, stdv)
            self.weight_raw_sigma.fill_(initial_raw_sigma)
            if self.bias:
                self.bias_mu.uniform_(-stdv, stdv)
                self.bias_raw_sigma.fill_(initial_raw_sigma)

    def freeze(self):
        """
        Freezes the random noise for weights and biases. This is useful for
        making deterministic predictions.
        """
        with torch.no_grad():
            self.weight_eps = torch.randn_like(self.weight_raw_sigma)
            if self.bias:
                self.bias_eps = torch.randn_like(self.bias_raw_sigma)
      
        
    def unfreeze(self) :
        """
        Unfreezes the random noise, allowing for stochastic sampling during
        the forward pass.
        """
        self.weight_eps = None
        if self.bias :
            self.bias_eps = None 
      
            
    def forward(self, input):
        """
        Performs the forward pass of the Bayesian linear layer.
        """
        weight_eps = self.weight_eps if self.weight_eps is not None else torch.randn_like(self.weight_raw_sigma)
        weight_sigma = self._sigma(self.weight_raw_sigma)
        weight = self.weight_mu + weight_sigma * weight_eps
        
        if self.bias:
            bias_eps = self.bias_eps if self.bias_eps is not None else torch.randn_like(self.bias_raw_sigma)
            bias_sigma = self._sigma(self.bias_raw_sigma)
            bias = self.bias_mu + bias_sigma * bias_eps
        else:
            bias = None
            
        return F.linear(input, weight, bias)


    def extra_repr(self):
        """
        Returns a string representation of the module.
        """
        return (f'prior_mu={self.prior_mu.item():.4g}, prior_sigma={self.prior_sigma.item():.4g}, '
                f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias}, initial_sigma={self.initial_sigma.item():.4g}')

    
    def _sigma(self, raw):
        if self.reparam == 'softplus':
            return F.softplus(raw)
        elif self.reparam == 'exp':
            return torch.exp(raw)
    
    def _inv_sigma(self, sigma):
        if self.reparam == 'softplus':
            return torch.log(torch.expm1(sigma))
        elif self.reparam == 'exp': 
            return torch.log(sigma)
