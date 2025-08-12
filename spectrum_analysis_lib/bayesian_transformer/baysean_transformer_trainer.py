import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import torch.nn.functional as F

from spectrum_analysis_lib.bayesian_transformer.bayesian_lib.linear import BayesLinear
from spectrum_analysis_lib.bayesian_transformer.bayesian_lib.loss import KLLoss

class BayesianTransformerTrainer:
    def __init__(self,
                 model,
                 train_data_generator,
                 test_data_generator,
                 num_train_steps,
                 batch_size, 
                 learning_rate, 
                 kl_weight, 
                 wandb_run, 
                 device,
                 grad_clip_value = 1.0,
                 kl_method='sampling',
                 eval_interval=200,
                 pruning_rhos = [0, 0.02, 0.1, 0.9],
                 num_eval_batches = 10,
                 num_eval_weight_samples = 10
                 ):
        
        # housekeeping
        self.device = device
        self.wandb_run = wandb_run
        
        # base setup
        self.model = model.to(self.device)
        self.train_gen = train_data_generator
        self.test_gen = test_data_generator
        
        # training arguments
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.grad_clip_value = grad_clip_value
        
        # evaluation arguments
        self.eval_interval = eval_interval
        self.pruning_rhos = pruning_rhos
        self.num_eval_batches = num_eval_batches
        self.num_eval_weight_samples = num_eval_weight_samples
       
        # intermediate saving dir before wandb push
        self.save_dir = self._setup_directories()
        
        # loss functions
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate
        )
        
        self.kl_loss = KLLoss(method=kl_method)
        self.ce_loss = nn.CrossEntropyLoss()

        # get information about model layers
        self.layers_to_prune = [name for name, mod in model.named_modules() if isinstance(mod, BayesLinear)]
    
    
    def _setup_directories(self):
        """Creates a directory for saving final artifacts."""
        save_dir = os.path.join('results', self.wandb_run.id)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    
    
    def train(self):
        progress_bar = tqdm(range(1, self.num_train_steps+1), desc="Training Step: ")
        
        for step in progress_bar:
            input_tokens, target_tokens = self.train_gen.generate_batch(self.batch_size)
            
            ce, kl, elbo = self._train_step(input_tokens, target_tokens)
            
            self.wandb_run.log({
                'train/ce_loss': ce, 
                'train/kl_loss': kl, 
                'train/elbo_loss': elbo
            }, step=step)
            
            if not (np.isfinite(ce) and np.isfinite(kl)):
                raise ValueError(f"Loss is invalid, aborting training and throwing an error. CE Loss: {ce} KL Loss: {kl}")
            
            progress_bar.set_postfix(CE=f"{ce:.3f}", KL=f"{kl:.3f}")
            
            if (step + 1) % self.eval_interval == 0 or (step + 1) == self.num_train_steps:
                self._evaluate_and_log(step + 1)
        
        return self.model
        
        
    def _train_step(self, tokens, targets):
        self.model.train()
        tokens, targets = tokens.to(self.device), targets.to(self.device)
        
        self.optimizer.zero_grad()
        
        logits = self.model(tokens)
        
        prediction_loss = self.ce_loss(logits, targets)
        kl = self.kl_loss(self.model) * self.kl_weight
        elbo = prediction_loss + kl
        
        elbo.backward()
        
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip_value)
        self.optimizer.step()
        
        return prediction_loss.item(), kl.item(), elbo.item()
    
    
    def _evaluate_and_log(self, step): 
        # Get pruning accuracy results
        pruning_results = self._evaluate_with_pruning()
        
        # Get relative variance results
        variance_results = self._get_relative_variances()
        
        # Create a single dictionary to log all metrics
        log_payload = {}
        
        log_payload['eval/baseline_accuracy'] = pruning_results.pop('baseline')
        
        for layer_name in self.layers_to_prune:
            log_payload[f'eval/variance/{layer_name}'] = variance_results[layer_name]
            for rho, acc in pruning_results[layer_name].items():
                log_payload[f'eval_pruned_accuracy/{layer_name}/rho_{rho}'] = acc
                
        self.wandb_run.log(log_payload, step=step)
    
    
    @torch.no_grad()
    def _evaluate_with_pruning(self):
        results = {'baseline': self._calculate_accuracy()}
        
        # Store the original model state to restore it after pruning
        original_state_dict = OrderedDict((k, v.clone()) for k, v in self.model.state_dict().items())
        
        for layer_name in tqdm(self.layers_to_prune, desc="Pruning layers", leave=False):
            results[layer_name] = {}
            layer_module = dict(self.model.named_modules())[layer_name]
            
            for rho in self.pruning_rhos:
                # Work on a temporary copy of the state dict for each pruning level
                temp_state_dict = OrderedDict((k, v.clone()) for k, v in original_state_dict.items())

                mu_key = f"{layer_name}.weight_mu"
                pruned_mu = self._prune_matrix(temp_state_dict[mu_key], rho)
                temp_state_dict[mu_key] = pruned_mu
                
                # If a layer is fully pruned also collapse its variance
                if rho == 0.0:
                    sigma_key = f"{layer_name}.weight_raw_sigma"
                    temp_state_dict[sigma_key].fill_(-20.0)
                    
                    if layer_module.bias:
                        bias_mu_key = f"{layer_name}.bias_mu"
                        temp_state_dict[bias_mu_key].fill_(0.0)
                        
                        bias_sigma_key = f"{layer_name}.bias_raw_sigma"
                        temp_state_dict[bias_sigma_key].fill_(-20.0)
                        
                self.model.load_state_dict(temp_state_dict, strict=True)
                acc = self._calculate_accuracy()
                results[layer_name][rho] = acc    
                
        self.model.load_state_dict(original_state_dict)
        return results
              
                
    @torch.no_grad()
    def _calculate_accuracy(self):
        self.model.eval()
        vocab_size = self.model.output_head.out_features
        
        total_correct = 0
        total_count = 0

        for _ in range(self.num_eval_batches):
            tokens, targets = self.test_gen.generate_batch(self.batch_size)
            tokens, targets = tokens.to(self.device), targets.to(self.device)
            
            output_probs = torch.zeros(tokens.size(0), vocab_size, device=self.device)
            for _ in range(self.num_eval_weight_samples):
                logits = self.model(tokens)
                output_probs += torch.softmax(logits, dim=-1)
            
            avg_probs = output_probs / self.num_eval_weight_samples
            preds = torch.argmax(avg_probs, dim=-1)
            
            total_correct += (preds == targets).sum().item()
            total_count += tokens.size(0)
        
        if total_count == 0:
            print("No eval tokens generated, check your batch size")
            return 0.0
            
        return total_correct / total_count
    
    
    @staticmethod
    def _prune_matrix(matrix, rho):
        """Low rank approximation using SVD"""
        if rho == 1.0:
            return matrix.clone()  # rho=1 means no change

        U, S, Vh = torch.linalg.svd(matrix)

        k = int(rho * S.size(0))
        if k == 0:
            return torch.zeros_like(matrix)  # rho=0 means fully remove

        reconstructed_matrix = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        return reconstructed_matrix
    
    
    def _get_relative_variances(self):
        rel_variances = {}
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, BayesLinear):
                    mu = module.weight_mu.detach()
                    raw_sigma = module.weight_raw_sigma.detach()
                    
                    if module.reparam == 'softplus':
                        sigma = F.softplus(raw_sigma)
                    elif module.reparam == 'exp':
                        sigma = torch.exp(raw_sigma)
                    
                    variance = sigma.pow(2).mean().item()
                
                    mean_abs_mu = mu.abs().mean().item()
                
                    relative_variance = variance / (mean_abs_mu + 1e-8)
                    rel_variances[name] = relative_variance
                
        return rel_variances
    
    
    def save_model(self):
        model_path = os.path.join(self.save_dir, 'final_model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        artifact = wandb.Artifact(
            name=f"run_{self.wandb_run.id}_model",
            type="model"
        )
        
        artifact.add_file(model_path)
        self.wandb_run.log_artifact(artifact)
        print("Model successfully saved to wandb")
    
    
    @staticmethod
    def load_model(wandb_run, artifact_address, btransformer_instance, device):
        try:
            artifact = wandb_run.use_artifact(artifact_address, type='model')
            artifact_dir = artifact.download()

            state_dict_path = os.path.join(artifact_dir, 'final_model.pt')
            btransformer_instance.load_state_dict(torch.load(state_dict_path, weights_only=True))
            btransformer_instance.to(device).eval()
            print("Pretrained Bayesian Transformer loaded successfully.")
            return btransformer_instance
        except wandb.errors.CommError as e:
            print(f"Pretrained Bayesian Transformer '{artifact_address}' not found.")
            return None