import torch
import numpy as np
import warnings
from tqdm import tqdm
import plotly.graph_objects as go
import weightwatcher as ww
import wandb
import os

from spectrum_analysis_lib.bayesian_transformer.bayesian_lib.linear import BayesLinear


class ESDAnalyzer:
    def __init__(self, model, wandb_run):
        self.model = model.eval()
        self.wandb_run = wandb_run
        self.ww = ww.WeightWatcher()
    
    
    def _get_bayes_layer(self, layer_name):
        layer = dict(self.model.named_modules()).get(layer_name)
        if layer is None:
            print(f"❌ Layer '{layer_name}' not found in model.")
            return None
        if not isinstance(layer, BayesLinear):
            print(f"⚠️ Skipping '{layer_name}', not a BayesLinear layer.")
            return None
        return layer
    
    
    def _get_sigma(self, layer):
        with torch.no_grad():
            if layer.reparam == 'softplus':
                return torch.nn.functional.softplus(layer.weight_raw_sigma)
            elif layer.reparam == 'exp':
                return torch.exp(layer.weight_raw_sigma)
            else:
                raise ValueError(f"Unknown reparameterization: {layer.reparam}")
       
        
    def _get_singular_values(self, matrix):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Suppress potential SVD warnings
            W = matrix.detach().cpu().float()
            s_values = torch.linalg.svdvals(W)
        return s_values.numpy()
        
    
    def _sample_weights_and_svals(self, mu, sigma, n):
        weights, svals = [], []
        with torch.no_grad():
            for _ in tqdm(range(n), desc="Sampling Weights", leave=False):
                w = mu + sigma * torch.randn_like(mu)
                weights.append(w.detach().cpu().float())
                svals.append(self._get_singular_values(w))
        return weights, svals
        
        
    def analyze_spectrum_distribution(self, layer_name, n_samples = 100, max_svals=30):
        layer = self._get_bayes_layer(layer_name)
        if layer is None:
            return
        
        print(f"📊 Analyzing Spectrum Distribution for layer: {layer_name}")
        with torch.no_grad():
            mu = layer.weight_mu.clone()
            sigma = self._get_sigma(layer)
        
        # Get singular values of the mean weight matrix
        s_values_mean = self._get_singular_values(mu)
        num_s_values = len(s_values_mean)
        
        # Sample from the posterior to get distributions
        _, sampled_svals = self._sample_weights_and_svals(mu, sigma, n_samples) 
        all_sampled = np.stack(sampled_svals, axis=0)
        
        # downsample to top k singular values
        k = int(min(max_svals, num_s_values))
        idx = np.arange(k)
        
        # Plot a boxplot of each singular value
        fig = go.Figure()
        
        for j, i in enumerate(idx):
            fig.add_trace(go.Box(
                x=[int(j)]*all_sampled.shape[0],
                y=all_sampled[:, i],
                name='Sampled Singular Values' if j == 0 else None,
                showlegend=(j == 0),
                boxpoints=False,
                marker_color='royalblue'
            ))

        fig.add_trace(go.Scatter(
            x=list(range(len(idx))),
            y=s_values_mean[idx],
            mode='markers',
            name='Singular Values of Mean (μ)',
            marker=dict(color='orange', size=7, line=dict(width=1))
        ))

        ticktext = [str(i+1) for i in idx]
        fig.update_layout(
            template='plotly_white',
            title=f'Singular Value Spectrum Distribution: {layer_name}',
            xaxis_title='Singular Value Index',
            yaxis_title='Singular Value',
            yaxis_type="log"
        )
        
        fig.update_xaxes(tickmode='array',
                     tickvals=list(range(len(idx))),
                     ticktext=ticktext,
                     tickangle=0)
        
        self.wandb_run.log({f"spectrum_distribution/{layer_name}": fig})
     
        
    def ww_analysis(self, layer_name, n_samples = 20, num_bins=100):
        layer = self._get_bayes_layer(layer_name) 
        if layer is None:
            return
        
        print(f"🔬 Analyzing ESD with fits for layer: {layer_name}")
        with torch.no_grad():
            mu = layer.weight_mu.clone()
            sigma = self._get_sigma(layer)
        
        # form a block diagonal matrix for single_layer use with weightwatcher
        blocks, sampled_svals = self._sample_weights_and_svals(mu, sigma, n_samples)
        pooled_eigenvalues = np.concatenate([sv ** 2 for sv in sampled_svals]).astype(np.float64)
            
        W_block = torch.block_diag(*blocks)
        del blocks
        
        out_f, in_f = W_block.shape
        lin = torch.nn.Linear(in_f, out_f, bias=False) # wrapper to make weightwatcher happy
        with torch.no_grad():
            lin.weight.copy_(W_block.float())
        
        save_dir = os.path.join("ww_plots", layer_name.replace("/", "_"))
        os.makedirs(save_dir, exist_ok=True)
        
        details = self.ww.analyze(model=lin, plot=True, mp_fit=True, savefig=save_dir)
        del W_block
        
        if details is None or details.empty:
            print(f"  - ⚠️ WeightWatcher analysis failed for {layer_name}")
            return
        
        row = details.iloc[0]
        
        # --------- Core tail and MP metrics  ----------
        alpha = float(row.get('alpha'))                                     # Power-law tail exponent of ESD
        lambda_min = float(row.get('xmin'))                                 # PL fit lower cutoff (start of tail region)
        d_ks = float(row.get('D'))                                          # KS statistic for PL tail fit (lower = better)
        bulk_min = float(row.get('bulk_min'))                               # MP bulk lower edge (noise bulk start)
        bulk_max = float(row.get('bulk_max'))                               # MP bulk upper edge (noise bulk end / “MP edge”)
        sigma_mp = float(row.get('sigma_mp'))                               # MP scale parameter used in bulk fit

        # --------- Spike and rank metrics ----------
        num_spikes = int(row.get('num_spikes'))                             # Eigenvalues above MP edge
        num_pl_spikes = int(row.get('num_pl_spikes'))                       # Outliers above PL tail model (WW heuristic)
        perc_spikes = 100.0 * num_spikes / float(row.get('num_evals'))      # Percentage of eigenvalues that are spikes
        spectral_norm = float(row.get('spectral_norm'))                     # λ_max (largest eigenvalue of correlation matrix)
        norm_frob = float(np.sqrt(row.get('norm')))                         # Frobenius norm ||W||_F
        
        # --------- Misc ----------
        entropy = float(row.get('entropy'))                                 # ESD entropy (higher = flatter)
        
        log_metrics = {
            f"esd_metrics/{layer_name}/alpha": alpha,                 
            f"esd_metrics/{layer_name}/xmin": lambda_min,            
            f"esd_metrics/{layer_name}/D_KS": d_ks,                  
            f"esd_metrics/{layer_name}/bulk_min": bulk_min,           
            f"esd_metrics/{layer_name}/bulk_max": bulk_max,           
            f"esd_metrics/{layer_name}/sigma_mp": sigma_mp,    
            f"esd_metrics/{layer_name}/num_spikes": num_spikes,           
            f"esd_metrics/{layer_name}/num_pl_spikes": num_pl_spikes,  
            f"esd_metrics/{layer_name}/perc_spikes": perc_spikes,   
            f"esd_metrics/{layer_name}/corr_spectral_norm": spectral_norm,    
            f"esd_metrics/{layer_name}/frob_norm": norm_frob,       
            f"esd_metrics/{layer_name}/entropy": entropy,
        }      
        self.wandb_run.log(log_metrics)
        print(f"  - MP edge (bulk_max): {bulk_max:.6f} | spike percentage: {perc_spikes} | alpha: {alpha:.3f} | D_KS: {d_ks:.3f}")
        
        esd_path   = os.path.join(save_dir, "ww.layer0.esd.png")  
        self.wandb_run.log({f"weightwatcher/{layer_name}": wandb.Image(esd_path)})
    
    
    def run_analyses(self):
        print("🚀 Starting comprehensive ESD analysis for all Bayesian layers")
        
        layers_to_analyze = [name for name, module in self.model.named_modules() 
                             if isinstance(module, BayesLinear)]
        if not layers_to_analyze:
            print("⚠️ No BayesianLinear layers found in the model.")
            return
        
        print(f"Found {len(layers_to_analyze)} Bayesian layers to analyze.")
        
        for layer_name in tqdm(layers_to_analyze, desc="ESD Analysis"):
            print(f"\n{'='*20} Analyzing Layer: {layer_name} {'='*20}")
            self.analyze_spectrum_distribution(layer_name)
            self.ww_analysis(layer_name)

        print("✅ All analyses complete.")
