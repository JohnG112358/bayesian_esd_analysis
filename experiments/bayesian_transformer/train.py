import torch
import random
import numpy as np
import wandb
import string

from spectrum_analysis_lib.data.data import ShakespeareDataProcessor
from spectrum_analysis_lib.data.data_generator import SyntheticDataGenerator
from spectrum_analysis_lib.bayesian_transformer.bayesian_transformer import BayesianTransformer
from spectrum_analysis_lib.bayesian_transformer.baysean_transformer_trainer import BayesianTransformerTrainer

# ---------------------
# Setup
# ---------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"🖥️  Using device: {device}")
if device.type == "cuda":
    idx = torch.cuda.current_device()
    print(f"Using GPU {idx}")
    print(torch.cuda.get_device_name(0))
    
# wandb run id
rand_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)) 
print(f"🔬 Experiment ID: {rand_id}")

# ---------------------
# Config
# ---------------------
config = dict(
    seed                    = 42,
    seq_len                 = 256,
    prob_noise_replacement  = 0.0,
    num_trigger_tokens      = 1,
    num_det_noise_tokens    = 1,
    random_noise_tokens     = False,
    d_model                 = 256,
    prior_mu                = 0,
    prior_sigma             = 0.01,
    reparam                 = 'softplus',
    use_ffn_block1          = True,
    use_ffn_block2          = True,
    num_train_steps         = 6000,
    batch_size              = 128,
    lr                      = 1e-4,
    kl_weight               = 1.0 / 6000000,
    grad_clip               = 1.0,
    
)

random.seed(config['seed'])
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])    
if device == 'cuda':
    torch.cuda.manual_seed_all(config['seed']) 
    
wandb_run = wandb.init(
    project="esd-analysis",
    config=config,
    name=f"btransform-{rand_id}",
    id=rand_id
)

# ---------------------
# Data
# ---------------------
dataset = ShakespeareDataProcessor(wandb_run)
meta = dataset.load_data()

if meta is None:
    meta = dataset.process()
    dataset.save_data()
   
train_data_generator = SyntheticDataGenerator(meta=meta, T=config['seq_len'], k=config['num_trigger_tokens'], 
                                              num_noise_tokens=config['num_det_noise_tokens'], alpha=config['prob_noise_replacement'],
                                              random_noise_tokens=config['random_noise_tokens'])

test_data_generator = SyntheticDataGenerator(meta=meta, T=config['seq_len'], k=config['num_trigger_tokens'], 
                                              num_noise_tokens=config['num_det_noise_tokens'], alpha=0,
                                              random_noise_tokens=config['random_noise_tokens'])

# ---------------------
# Model
# ---------------------
model = BayesianTransformer(vocab_size=meta['vocab_size'], d_model=config['d_model'], max_seq_len=config['seq_len'],
                            prior_mu=config['prior_mu'], prior_sigma=config['prior_sigma'], use_ffn_block1=config['use_ffn_block1'],
                            use_ffn_block2=config['use_ffn_block2'], reparam=config['reparam'])

from spectrum_analysis_lib.bayesian_transformer.bayesian_lib.linear import BayesLinear
import torch.nn.functional as F
for name, m in model.named_modules():
    if isinstance(m, BayesLinear):
        s = (F.softplus(m.weight_raw_sigma) if m.reparam=='softplus' else m.weight_raw_sigma.exp()).mean().item()
        print(name, s)

# ---------------------
# Training
# ---------------------
trainer = BayesianTransformerTrainer(model=model, train_data_generator=train_data_generator, test_data_generator=test_data_generator,
                                     num_train_steps=config['num_train_steps'], batch_size=config['batch_size'], learning_rate=config['lr'],
                                     kl_weight=config['kl_weight'], grad_clip_value=config['grad_clip'], wandb_run=wandb_run, device=device)
trainer.train()
trainer.save_model()

