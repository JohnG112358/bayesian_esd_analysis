import torch
import numpy as np
import random
from typing import List

class SyntheticDataGenerator:
    def __init__(
        self,
        meta,
        T,
        k=1,
        num_noise_tokens=1,
        alpha=0.5, 
        random_noise_tokens=False):
        """
        This data generator does not guarantee trigger token clue (q, y_bar) appears at least once before T-th token

        Args:
            meta: Dictionary of metadata
            T (int): sequence length
            k (int): how many kinds of trigger tokens
            num_noise_tokens (int): how many kinds of noise tokens
            alpha (float): probability of generate noise token
        """
        self.T = T
        self.k = k
        self.num_noise_tokens = num_noise_tokens
        self.alpha = alpha
        self.random_noise_tokens = random_noise_tokens
        
        self.vocab_size = meta['vocab_size']
        self.itos = meta['itos']
        self.stoi = meta['stoi']
        
        self.marginal_probs = np.zeros(self.vocab_size)
        for char, count in meta['unigrams'].items():
            self.marginal_probs[self.stoi[char]] = count
        self.marginal_probs /= self.marginal_probs.sum()

        self.cond_probs = np.zeros((self.vocab_size, self.vocab_size))
        for (w1, w2), count in meta['bigrams'].items():
            self.cond_probs[self.stoi[w1], self.stoi[w2]] = count
        row_sums = self.cond_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.cond_probs = self.cond_probs / row_sums
        
        sorted_tokens = list(self.marginal_probs.argsort()[::-1])
        
        # most common k-th as trigger
        self.trigger_tokens: List[int] = sorted_tokens[:k]
        
        # least used num_noise_tokens tokens are noises
        self.noise_token_ids: List[int] = sorted_tokens[-num_noise_tokens:]

        print("Data Generator (Parametric, Original Logic) Initialization Done.")
        print(f"  Vocab size: {self.vocab_size}")
        trigger_str = [f'{t} ("{self.itos[t]}")' for t in self.trigger_tokens]
        noise_str = [f'{t} ("{self.itos[t]}")' for t in self.noise_token_ids]
        print(f"  {self.k} Trigger token(s): {trigger_str}")
        if not self.random_noise_tokens:
            print(f"  {self.num_noise_tokens} Fixed Noise token(s): {noise_str}")
        else:
            print(f"  Noise tokens will be chosen randomly per sample.")


    def generate_batch(self, batch_size):
        sequences = torch.zeros((batch_size, self.T + 1), dtype=torch.long)
        
        for i in range(batch_size):
            # i. sample a y_bar (step i)
            possible_y_bar_pool = list(set(range(self.vocab_size)) - set(self.trigger_tokens) - set(self.noise_token_ids))
            y_bar = random.choice(possible_y_bar_pool)
            
            # for a truly random noise token
            random_noise_pool = list(set(range(self.vocab_size)) - set(self.trigger_tokens) - {y_bar})
            
            # --- ii. Markov: z_1 to z_{T-1} ---
            # z_1 ~ π_u(·)
            z_t = np.random.choice(self.vocab_size, p=self.marginal_probs)
            sequences[i, 0] = z_t
            
            for t in range(self.T - 2): # generate z_2 to z_{T-1}
                if z_t in self.trigger_tokens:
                    # z_{t+1} ~ p_{α, y_bar}(·)
                    if random.random() < self.alpha:
                        z_t_plus_1 = random.choice(self.noise_token_ids)
                    else:
                        z_t_plus_1 = y_bar
                else:
                    # z_{t+1} ~ π_b(·|z_t)
                    probs = self.cond_probs[z_t]
                    z_t_plus_1 = np.random.choice(self.vocab_size, p=probs)
                
                sequences[i, t + 1] = z_t_plus_1
                z_t = z_t_plus_1

            # --- iii. set z_T = q and generate target y ---
            # randomly choose a trigger token z_T
            q_final = random.choice(self.trigger_tokens)
            sequences[i, self.T - 1] = q_final
            
            # y = z_{T+1} ~ p_{α, y_bar}(·)
            if random.random() < self.alpha:
                if self.random_noise_tokens:
                    sequences[i, self.T] = random.choice(random_noise_pool)
                else:
                    sequences[i, self.T] = random.choice(self.noise_token_ids)
            else:
                sequences[i, self.T] = y_bar
                
        input_tokens = sequences[:, :-1]
        target_tokens = sequences[:, -1]
        
        return input_tokens, target_tokens
    