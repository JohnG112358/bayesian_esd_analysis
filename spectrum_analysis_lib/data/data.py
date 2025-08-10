import os
import pickle
import requests
import numpy as np
import wandb
from collections import Counter, defaultdict
from itertools import chain
import pickle

class ShakespeareDataProcessor:
    def __init__(self, wandb_run, train_split=0.9):
        self.data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        self.wandb_run = wandb_run
        self.train_split = train_split
        
        self.output_dir = self._setup_directories()
        self.input_file_path = os.path.join(self.output_dir, 'input.txt')
        self.train_bin_path = os.path.join(self.output_dir, 'train.bin')
        self.val_bin_path = os.path.join(self.output_dir, 'val.bin')
        self.meta_pkl_path = os.path.join(self.output_dir, 'meta.pkl')
        
        self.data = None
        self.chars = None
        self.vocab_size = None
        self.stoi = None
        self.itos = None
    
    
    def _setup_directories(self):
        """Creates a directory for saving final artifacts."""
        data_dir = os.path.join('data')
        os.makedirs(data_dir, exist_ok=True)
        return data_dir 
      
        
    def _load_data(self):
        if not os.path.exists(self.input_file_path):
            print(f"Downloading dataset to {self.input_file_path}...")
            with open(self.input_file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(self.data_url).text)
        
        with open(self.input_file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()
            
        print(f"Length of dataset in characters: {len(self.data):,}")
        
    
    def _build_vocab(self):
        """Builds the vocabulary and character mappings."""
        self.chars = sorted(list(set(self.data)))
        self.vocab_size = len(self.chars)
        print("All unique characters:", ''.join(self.chars))
        print(f"Vocab size: {self.vocab_size:,}")

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    
    def encode(self, s):
        """Encoder: takes a string, outputs a list of integers."""
        return [self.stoi[c] for c in s]
    
    
    def decode(self, l):
        """Decoder: takes a list of integers, outputs a string."""
        return ''.join([self.itos[i] for i in l])
    
    
    def process_and_save(self):
        """Main method to run the full processing pipeline."""
        self._load_data()
        self._build_vocab()
        
        # Create train and validation splits
        n = len(self.data)
        train_data = self.data[:int(n * self.train_split)]
        val_data = self.data[int(n * self.train_split):]

        # Encode splits to integers
        train_ids = self.encode(train_data)
        val_ids = self.encode(val_data)
        print(f"Train has {len(train_ids):,} tokens")
        print(f"Val has {len(val_ids):,} tokens")
        
        # Export to bin files
        train_ids_np = np.array(train_ids, dtype=np.uint16)
        val_ids_np = np.array(val_ids, dtype=np.uint16)
        train_ids_np.tofile(self.train_bin_path)
        val_ids_np.tofile(self.val_bin_path)
        print(f"Saved {self.train_bin_path} and {self.val_bin_path}")
        
        # Calculate n-grams for metadata
        unigrams = dict(Counter(self.data))
        bigrams = dict(Counter(chain(zip(self.data[::2], self.data[1::2]), zip(self.data[1::2], self.data[2::2]))))
        bigrams_cond = defaultdict(dict)
        for (w1, w2), cnt in bigrams.items():
            bigrams_cond[w1][w2] = cnt
            
        meta = {
            'vocab_size': self.vocab_size,
            'itos': self.itos,
            'stoi': self.stoi,
            'unigrams': unigrams,
            'bigrams': bigrams,
            'bigrams_cond': dict(bigrams_cond), # convert defaultdict to dict for pickling
        }
        with open(self.meta_pkl_path, 'wb') as f:
            pickle.dump(meta, f)
        print(f"Saved metadata to {self.meta_pkl_path}")
        
        
    def save_data(self):
        """Logs the processed files as an artifact to Weights & Biases."""
        
        artifact = wandb.Artifact(
            name='shakespeare-char-dataset',
            type='tiny-shakespeare'
        )
        
        artifact.add_file(self.train_bin_path)
        artifact.add_file(self.val_bin_path)
        artifact.add_file(self.meta_pkl_path)
        
        self.wandb_run.log_artifact(artifact)


    def load_data(self):
        try:
            artifact = self.wandb_run.use_artifact("shakespeare-char-dataset:latest", type='tiny-shakespeare')
            artifact_dir = artifact.download()
            
            train_bin_path = os.path.join(artifact_dir, 'train.bin')
            val_bin_path = os.path.join(artifact_dir, 'val.bin')
            meta_pkl_path = os.path.join(artifact_dir, 'meta.pkl')
            
            with open(meta_pkl_path, 'rb') as f:
                meta = pickle.load(f)
            
            self.stoi = meta['stoi']
            self.itos = meta['itos']
            self.vocab_size = meta['vocab_size']
            
            self.train_ids = np.fromfile(train_bin_path, dtype=np.uint16)
            self.val_ids = np.fromfile(val_bin_path, dtype=np.uint16)
            
            print("Successfully loaded data and metadata from artifact.")
            print(f"  Train tokens: {len(self.train_ids):,}")
            print(f"  Validation tokens: {len(self.val_ids):,}")

            return meta      
        except wandb.errors.CommError:
            print("Artifact 'shakespeare-char-dataset:latest' not found. Will process data from scratch.")
            return None
        