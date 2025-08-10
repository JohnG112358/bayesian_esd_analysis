import torch
import torch.nn as nn
import numpy as np
import warnings
from tqdm import tqdm
import plotly.graph_objects as go
from weightwatcher import WeightWatcher 

from spectrum_analysis_lib.bayesian_transformer.bayesian_lib.linear import BayesLinear


class ESDAnalyzer:
    def __init__(self, model, wandb_run):
        self.model = model.eval()
        self.wandb_run = wandb_run
        self.ww = WeightWatcher()
    
    
    def _get_sigma(self, layer):
        if layer.reparam == 'softplus':
            return torch.nn.functional.softplus(layer.weight_raw_sigma.data)
        elif layer.reparam == 'exp':
            return torch.exp(layer.weight_raw_sigma.data)
        else:
            raise ValueError(f"Unknown reparameterization: {layer.reparam}")
       
        
    def _get_singular_values(self, matrix):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Suppress potential SVD warnings
            W = matrix.detach().cpu().float()
            s_values = torch.linalg.svdvals(W)
        return s_values.numpy()
    
    
    def _calculate_spectral_metrics(self, s_values):
        s_values_sq = s_values**2
        spectral_norm = s_values[0] if len(s_values) > 0 else 0.0
        log_spectral_norm = np.log10(spectral_norm) if spectral_norm > 0 else -np.inf
        
        
        frobenius_norm = np.sqrt(np.sum(s_values_sq))
        log_norm = np.log10(frobenius_norm) if frobenius_norm > 0 else -np.inf
        
        stable_rank = (frobenius_norm**2 / spectral_norm**2) if spectral_norm > 0 else 0.0
        
        return {
            "log_spectral_norm": float(log_spectral_norm),
            "log_norm": float(log_norm),
            "stable_rank": float(stable_rank),
        }
     
        
    def analyze_spectrum_distribution(self, layer_name, n_samples = 100):
        try:
            layer = dict(self.model.named_modules())[layer_name]
            if not isinstance(layer, BayesLinear):
                print(f"⚠️ Skipping '{layer_name}', not a BayesLinear layer.")
                return
        except KeyError:
            print(f"❌ Layer '{layer_name}' not found in model.")
            return

        print(f"📊 Analyzing Spectrum Distribution for layer: {layer_name}")
        mu = layer.weight_mu.data
        sigma = self._get_sigma(layer)
        
        # Get singular values of the mean weight matrix
        s_values_mean = self._get_singular_values(mu)
        num_s_values = len(s_values_mean)
        
        # Log scalar metrics for the mean matrix
        mean_metrics = self._calculate_spectral_metrics(s_values_mean)
        self.wandb_run.log({f"metrics/{layer_name}/{k}": v for k, v in mean_metrics.items()})
        
        # Sample from the posterior to get distributions
        all_sampled_s_values = []
        for _ in tqdm(range(n_samples), desc=f"Sampling {layer_name}", leave=False):
            w_sample = mu + sigma * torch.randn_like(mu)
            all_sampled_s_values.append(self._get_singular_values(w_sample))
        all_sampled_s_values = np.array(all_sampled_s_values)
        
        # Plot a boxplot of each singular value
        fig = go.Figure()
        
        for i in range(num_s_values):
            fig.add_trace(go.Box(
                y=all_sampled_s_values[:, i],
                name=f'SV {i+1}',
                showlegend=False,
                marker_color='royalblue'
            ))

        fig.add_trace(go.Scatter(
            x=list(range(num_s_values)),
            y=s_values_mean,
            mode='markers',
            name='Singular Values of Mean (μ)',
            marker=dict(color='darkorange', size=8, line=dict(width=1, color='black'))
        ))

        fig.update_layout(
            title=f'Singular Value Spectrum Distribution: {layer_name}',
            xaxis_title='Singular Value Index (Sorted Descending)',
            yaxis_title='Singular Value (log scale)',
            yaxis_type="log",
            xaxis=dict(tickvals=list(range(num_s_values)), ticktext=[f"{i+1}" for i in range(num_s_values)])
        )
        
        self.wandb_run.log({f"spectrum_distribution/{layer_name}": fig})
     
        
    def analyze_esd_with_means(self, layer_name, n_samples=100, num_bins=100):
        try:
            layer = dict(self.model.named_modules())[layer_name]
            if not isinstance(layer, BayesLinear):
                print(f"⚠️ Skipping '{layer_name}', not a BayesLinear layer.")
                return
        except KeyError:
            print(f"❌ Layer '{layer_name}' not found in model.")
            return
        
        print(f"📊 Analyzing ESD for layer: {layer_name}")
        mu = layer.weight_mu.data
        sigma = self._get_sigma(layer)
        
        s_values_mean = self._get_singular_values(mu)
        
        # Pool S-values from all samples
        all_sampled_s_values = []
        for _ in tqdm(range(n_samples), desc=f"Sampling {layer_name}", leave=False):
            w_sample = mu + sigma * torch.randn_like(mu)
            all_sampled_s_values.extend(self._get_singular_values(w_sample))
            
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=all_sampled_s_values,
            name=f'ESD of Sampled W ({n_samples} samples)',
            histnorm='probability density',
            marker_color='royalblue',
            nbinsx=num_bins
        ))
        
        fig.add_trace(go.Histogram(
            x=s_values_mean,
            name='ESD of Mean (μ)',
            histnorm='probability density',
            marker_color='darkorange',
            nbinsx=int(num_bins / 2),
            opacity=0.85
        ))
        
        fig.update_layout(
            barmode='overlay',
            title=f'Empirical Spectral Density (ESD): {layer_name}',
            xaxis_title='Singular Value (log scale)',
            yaxis_title='Density (log scale)',
            xaxis_type="log",
            yaxis_type="log"
        )
        fig.update_traces(opacity=0.7)
        
        self.wandb_run.log({f"esd/{layer_name}": fig})
        
        
    def analyze_esd_with_fits(self, layer_name, n_samples = 100, num_bins=100):
        try:
            layer = dict(self.model.named_modules())[layer_name]
            if not isinstance(layer, BayesLinear):
                print(f"⚠️ Skipping '{layer_name}', not a BayesLinear layer.")
                return
        except KeyError:
            print(f"❌ Layer '{layer_name}' not found in model.")
            return
        
        print(f"🔬 Analyzing ESD with fits for layer: {layer_name}")
        mu = layer.weight_mu.data
        sigma = self._get_sigma(layer)
        
        # Pool eigenvalues from all samples
        # WeightWatcher works with eigenvalues (lambda = singular_value^2)
        pooled_eigenvalues = []
        for _ in tqdm(range(n_samples), desc=f"Sampling {layer_name}", leave=False):
            w_sample = mu + sigma * torch.randn_like(mu)
            s_values = self._get_singular_values(w_sample)
            pooled_eigenvalues.extend(s_values**2)
            
        pooled_eigenvalues = np.array(pooled_eigenvalues)
        
        details = self.ww.analyze(eigs=[pooled_eigenvalues], plot=False, mp_fit=True)
        if details is None or details.empty:
            print(f"  - ⚠️ WeightWatcher analysis failed for {layer_name}")
            return
        
        details = details.iloc[0]
        alpha, lambda_min, d_ks, lambda_max_bulk = details.get('alpha'), details.get('xmin'), details.get('D'), details.get('max_rand_eval')
        spikes = pooled_eigenvalues[pooled_eigenvalues > lambda_max_bulk]
        
        spike_metrics = {
            'num_spikes': len(spikes),
            'percentage_spikes': 100 * len(spikes) / len(pooled_eigenvalues) if len(pooled_eigenvalues) > 0 else 0,
            'avg_spike_distance': np.mean(spikes - lambda_max_bulk) if len(spikes) > 0 else 0,
            'relative_spike_mass': np.sum(spikes) / np.sum(pooled_eigenvalues) if np.sum(pooled_eigenvalues) > 0 else 0,
            'mp_edge': lambda_max_bulk
        }
        
        log_metrics = {f"metrics/{layer_name}/{k}": v for k, v in spike_metrics.items()}
        log_metrics[f"metrics/{layer_name}/alpha"] = alpha
        log_metrics[f"metrics/{layer_name}/lambda_min"] = lambda_min
        log_metrics[f"metrics/{layer_name}/D_KS"] = d_ks
        self.wandb_run.log(log_metrics)
        print(f"  - Spikes Found: {spike_metrics['num_spikes']} ({spike_metrics['percentage_spikes']:.2f}%)")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pooled_eigenvalues, 
                                   name='ESD', 
                                   histnorm='probability density', 
                                   marker_color='#1f77b4', 
                                   nbinsx=num_bins, 
                                   showlegend=False))
        
        x_pl, y_pl = self.ww.get_ESD_pl_fit(details.N, alpha, lambda_min, details.xmax)
        if x_pl is not None:
            fig.add_trace(go.Scatter(x=x_pl, 
                                     y=y_pl, 
                                     mode='lines', 
                                     name='Power-Law Fit', 
                                     line=dict(color='red', width=2.5, dash='dash'), 
                                     showlegend=False))
        
        
        fig.add_vline(x=lambda_min, line_width=2, 
                      line_dash="solid", 
                      line_color="red")
        
        fig.add_trace(go.Scatter(x=[None], 
                                 y=[None], 
                                 mode='lines', 
                                 name='λ_min (Fit Start)', 
                                 line=dict(color='red', width=2, dash='solid')))
        
        fig.add_vline(x=lambda_max_bulk, 
                      line_width=2, 
                      line_dash="dash", 
                      line_color="blue")
        
        fig.add_trace(go.Scatter(x=[None], 
                                 y=[None], 
                                 mode='lines', 
                                 name='MP Edge (Bulk End)', 
                                 line=dict(color='blue', width=2, dash='dash')))
        
        sigma_val = np.sqrt(details.get('sigma_sq'))
        title_text = f"Log-Log ESD for Layer {layer_name.split('.')[-1]}<br>" + \
                     f"<sup>α = {alpha:.3f}; D_KS = {d_ks:.3f}; λ_min = {lambda_min:.3f}; σ_MP = {sigma_val:.3f}</sup>"
                     
        fig.update_layout(title=dict(text=title_text, x=0.5), 
                          xaxis_title='Eigenvalue (λ)', 
                          yaxis_title='Density', 
                          xaxis_type="log", 
                          yaxis_type="log", 
                          legend=dict(x=0.8, y=0.98), 
                          plot_bgcolor='white')
        
        fig.update_xaxes(showline=True, 
                         linewidth=1, 
                         linecolor='black', 
                         mirror=True, 
                         gridcolor='lightgrey')
        
        fig.update_yaxes(showline=True, 
                         linewidth=1, 
                         linecolor='black', 
                         mirror=True, 
                         gridcolor='lightgrey')
        
        self.wandb_run.log({f"esd_with_fits/{layer_name}": fig})
    
    
    def run_analyses(self):
        print("🚀 Starting comprehensive ESD analysis for all Bayesian layers...")
        
        layers_to_analyze = [name for name, module in self.model.named_modules() 
                             if isinstance(module, BayesLinear)]
        
        if not layers_to_analyze:
            print("⚠️ No BayesianLinear layers found in the model.")
            return
        
        print(f"Found {len(layers_to_analyze)} Bayesian layers to analyze.")
        
        for layer_name in layers_to_analyze:
            print(f"\n{'='*20} Analyzing Layer: {layer_name} {'='*20}")
            self.analyze_spectrum_distribution(layer_name)
            self.analyze_esd_with_means(layer_name)
            self.analyze_esd_with_fits(layer_name)

        print("\n✅ All analyses complete.")
