import math
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F


class BayesLinear(Module):
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True, reparam='softplus'):
        super(BayesLinear, self).__init__()
        
        if not isinstance(prior_mu, (int, float)) or not math.isfinite(prior_mu):
            raise ValueError(f"prior_mu must be a finite number, but got {prior_mu}")
        if not isinstance(prior_sigma, (int, float)) or prior_sigma <= 0:
            raise ValueError(f"prior_sigma must be a positive number, but got {prior_sigma}")
        
        if reparam not in ['softplus', 'exp']:
            raise ValueError(f"Reparameterization function must be one of ['softplus', 'exp'], got {reparam}")
        self.reparam = reparam
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        
        # Weight parameters
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_raw_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)
        
        # Bias parameters
        self.bias = bias
        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_raw_sigma = Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_raw_sigma', None)
            self.register_buffer('bias_eps', None)
            
        self.reset_parameters()


    def reset_parameters(self):
        """
        Initializes the parameters of the layer. The raw_sigma parameters are initialized
        with the inverse softplus or log of the prior_sigma to ensure that the initial
        sigma matches the prior.
        """        
        if self.reparam == 'softplus':
            initial_raw_sigma = math.log(math.expm1(self.prior_sigma))
        elif self.reparam == 'exp':
            initial_raw_sigma = math.log(self.prior_sigma)
        
        # Initialize weights
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_raw_sigma.data.fill_(initial_raw_sigma)
        
        # Initialize biases
        if self.bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_raw_sigma.data.fill_(initial_raw_sigma)
         

    def freeze(self):
        """
        Freezes the random noise for weights and biases. This is useful for
        making deterministic predictions.
        """
        self.weight_eps = torch.randn_like(self.weight_raw_sigma)
        if self.bias :
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
        
        if self.reparam == 'softplus':
            weight_sigma = F.softplus(self.weight_raw_sigma)
        elif self.reparam == 'exp':
            weight_sigma = torch.exp(self.weight_raw_sigma)
        
        weight = self.weight_mu + weight_sigma * weight_eps
        
        if self.bias:
            bias_eps = self.bias_eps if self.bias_eps is not None else torch.randn_like(self.bias_raw_sigma)
            
            if self.reparam == 'softplus':
                bias_sigma = F.softplus(self.bias_raw_sigma)
            elif self.reparam == 'exp':
                bias_sigma = torch.exp(self.bias_raw_sigma)
                
            bias = self.bias_mu + bias_sigma * bias_eps
        else:
            bias = None
            
        return F.linear(input, weight, bias)


    def extra_repr(self):
        """
        Returns a string representation of the module.
        """
        return (f'prior_mu={self.prior_mu}, prior_sigma={self.prior_sigma}, '
                f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias}')
