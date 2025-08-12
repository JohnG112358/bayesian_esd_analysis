import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spectrum_analysis_lib.bayesian_transformer.bayesian_lib.linear import BayesLinear


class BayesianAttention(nn.Module):
    """
    - W_k, W_v, W_o are learnable Bayesian linear layers.
    - W_q is identity matrix, same with Joan's code.
    """
    def __init__(self, 
                 dim: int, 
                 prior_mu: float = 0.0, 
                 prior_sigma: float = 1.0,
                 initial_sigma = 0.01,
                 reparam = 'softplus'):
        super().__init__()
        self.dim = dim
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.wq = nn.Identity()

        self.wk = BayesLinear(prior_mu, prior_sigma, initial_sigma, dim, dim, bias=False, reparam=reparam)
        self.wv = BayesLinear(prior_mu, prior_sigma, initial_sigma, dim, dim, bias=False, reparam=reparam)
        self.wo = BayesLinear(prior_mu, prior_sigma, initial_sigma, dim, dim, bias=False, reparam=reparam)


    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # get Q, K, V
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # did not use multihead，dim=d_model
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.dim)
        
        # apply causal mask
        scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)

        h = torch.matmul(attn_weights, xv)

        output = self.wo(h)
        return output
    

class BayesianFeedForward(nn.Module):
    """
    2 layer MLP (bayesian version)
    """
    def __init__(self, 
                 dim: int, 
                 hidden_dim: int, 
                 relu: bool = True, 
                 prior_mu: float = 0.0, 
                 prior_sigma: float = 1.0,
                 initial_sigma = 0.01,
                 reparam = 'softplus'):
        super().__init__()
        self.w1 = BayesLinear(prior_mu, prior_sigma, initial_sigma, dim, hidden_dim, bias=False, reparam=reparam)
        self.w2 = BayesLinear(prior_mu, prior_sigma, initial_sigma, hidden_dim, dim, bias=False, reparam=reparam)
        self.relu = relu


    def forward(self, x):
        h = self.w1(x)
        if self.relu:
            h = F.relu(h)
        output = self.w2(h)
        return output
    

class BayesianTransformerBlock(nn.Module):
    """
    Implementation of the 2 layer transformer block in https://arxiv.org/pdf/2406.03068 Section 3
    """
    def __init__(self, 
                 dim: int, 
                 mlp_multiplier: int = 4, 
                 relu: bool = True, 
                 use_ffn: bool = True,
                 prior_mu: float = 0.0, 
                 prior_sigma: float = 1.0, 
                 initial_sigma = 0.01,
                 reparam = 'softplus'):
        super().__init__()
        self.dim = dim
        self.attention = BayesianAttention(dim=dim, prior_mu=prior_mu, prior_sigma=prior_sigma, initial_sigma=initial_sigma, reparam=reparam)
        
        self.use_ffn = use_ffn
        if self.use_ffn:
            hidden_dim = dim * mlp_multiplier
            self.ff = BayesianFeedForward(dim=dim, hidden_dim=hidden_dim, relu=relu,
                                          prior_mu=prior_mu, prior_sigma=prior_sigma, initial_sigma=initial_sigma, reparam=reparam)


    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x:  x̂_t or x¹_t
        
        # h_attn:  h¹_t or h²_t
        h_attn = self.attention(x, mask)
        
        # first res: h_res = x + h_attn
        # h_res:  F_1 or F_2 's input (x_t + h_t)
        h_res = x + h_attn
        
        if not self.use_ffn:
            # if there is no ffn
            return h_res

        # h_ffn:  F₁(h_res) or F₂(h_res)
        h_ffn = self.ff(h_res)

        # second res: output = h_res + h_ffn
        # output:  x¹_t or x²_t
        output = h_res + h_ffn
        
        return output
    

class BayesianTransformer(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 max_seq_len: int,
                 mlp_multiplier: int = 4, 
                 relu: bool = True,
                 prior_mu: float = 0.0, 
                 prior_sigma: float = 1.0, 
                 initial_sigma = 0.01,
                 use_ffn_block1: bool = True,
                 use_ffn_block2: bool = True,
                 tie_weights: bool = False,
                 reparam = 'softplus'):

        super().__init__()
        self.d_model = d_model

        # 1. embedding (W_E) - freezed
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.tok_embeddings.weight.requires_grad = False

        # 2. positional encoding (p_t) - freezed
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # 3. two bayesian Transformer layer, use ffn
        self.layer1 = BayesianTransformerBlock(dim=d_model, mlp_multiplier=mlp_multiplier, relu=relu, 
                                               use_ffn=use_ffn_block1, prior_mu=prior_mu, prior_sigma=prior_sigma, initial_sigma=initial_sigma, reparam=reparam)
        self.layer2 = BayesianTransformerBlock(dim=d_model, mlp_multiplier=mlp_multiplier, relu=relu,
                                               use_ffn=use_ffn_block2, prior_mu=prior_mu, prior_sigma=prior_sigma, initial_sigma=initial_sigma, reparam=reparam)

        # 4. umbedding (decode) (W_U) - freezed
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight.requires_grad = False
        
        if tie_weights:
            self.output_head.weight = self.tok_embeddings.weight
        

    def forward(self, tokens: torch.Tensor):
        # tokens: [batch_size, seq_len]
        batch_size, seq_len = tokens.shape

        #  x̂_t = W_E(z_t) + p_t
        h = self.tok_embeddings(tokens) # [batch_size, seq_len, d_model]
        h = h + self.pe[:seq_len, :].unsqueeze(0) # Add positional encoding

        # Causal mask
        mask = torch.full((seq_len, seq_len), float('-inf'), device=tokens.device, dtype=h.dtype)
        mask = torch.triu(mask, diagonal=1)

        # h -> layer1 -> h -> layer2 -> h
        h = self.layer1(h, mask)
        h = self.layer2(h, mask)
        
        # ξ_T = W_U * x²_T
        last_step_hidden_state = h[:, -1, :] # [batch_size, d_model]
        logits = self.output_head(last_step_hidden_state) # [batch_size, vocab_size]

        return logits
    