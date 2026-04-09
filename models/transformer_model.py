"""
Transformer Autoregressive Model Module
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(nn.Module):
    """Positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerAutoregressive(nn.Module):
    """Transformer Autoregressive Model"""
    
    def __init__(self, seq_len, feature_dim, d_model=512, nhead=8, 
                 num_layers=4, dropout=0.1, device='cpu'):
        super(TransformerAutoregressive, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, feature_dim)
        )
        
        # Residual connection
        self.residual = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        self.scaler = None
        self._is_trained = False
        
        # Initialize weights better
        self._initialize_weights()
        
        # Move model to specified device
        self.to(device)
    
    def _initialize_weights(self):
        """Better weight initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_causal_mask(self, size):
        """Generate causal mask to prevent seeing future information"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(self, x, tgt=None):
        """
        Forward propagation
        x: [batch_size, seq_len, feature_dim] Input sequence (memory)
        tgt: [batch_size, tgt_len, feature_dim] Target sequence (used in training)
        """
        batch_size, seq_len, _ = x.shape
        
        # Save residual
        if tgt is not None:
            residual = tgt
        else:
            residual = x
        
        # Project to d_model dimension
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        if tgt is not None:
            # Training mode
            tgt_len = tgt.size(1)
            tgt = self.input_projection(tgt)
            tgt = self.positional_encoding(tgt)
            # Generate causal mask to prevent seeing future information
            mask = self._generate_causal_mask(tgt_len).to(self.device)
            output = self.transformer_decoder(tgt, x, tgt_mask=mask)
            # output = self.transformer_decoder(tgt, x, tgt_mask=None)
        else:
            mask = self._generate_causal_mask(seq_len).to(self.device)
            output = self.transformer_decoder(x, x, tgt_mask=mask)
            # output = self.transformer_decoder(x, x, tgt_mask=None)
        
        output = self.output_projection(output)
        
        # Add residual connection
        output = output + self.residual(residual)
        # output = output

        # Layer normalization
        output = self.layer_norm(output)
        
        return output
    
    def generate(self, n_samples, n_steps=None, denormalize=True, temperature=0.7):
        """
        Generate new samples
        """
        self.eval()
        
        if n_steps is None:
            n_steps = self.seq_len
        
        with torch.no_grad():
            all_samples = []
            batch_size = min(100, n_samples)
            
            for i in range(0, n_samples, batch_size):
                current_batch = min(batch_size, n_samples - i)
                
                generated = torch.randn(current_batch, 1, self.feature_dim).to(self.device)
                
                for step in range(1, n_steps):
                    current_seq = generated

                    if current_seq.size(1) < self.seq_len:
                        padding = torch.zeros(current_batch, self.seq_len - current_seq.size(1), self.feature_dim).to(self.device)
                        padded_seq = torch.cat([current_seq, padding], dim=1)
                    else:
                        padded_seq = current_seq[:, -self.seq_len:, :]
                    
                    next_step_pred = self.forward(padded_seq)

                    next_step = next_step_pred[:, -1:, :]

                    next_step = next_step + torch.randn_like(next_step) * temperature

                    if np.random.random() < 0.3:
                        next_step = next_step + torch.randn_like(next_step) * 0.1

                    generated = torch.cat([generated, next_step], dim=1)

                generated = generated[:, :n_steps, :]
                all_samples.append(generated.cpu().numpy())
            
            generated_data = np.vstack(all_samples)

        if denormalize and self.scaler is not None:
            for i in range(generated_data.shape[-1]):
                min_val = self.scaler['min'][i]
                max_val = self.scaler['max'][i]
                if max_val - min_val > 0:
                    generated_data[..., i] = (generated_data[..., i] + 1) / 2 * (max_val - min_val) + min_val

        noise_scale = 0.05 * np.std(generated_data)
        generated_data += np.random.normal(0, noise_scale, generated_data.shape)

        if self.scaler is not None:
            for i in range(generated_data.shape[-1]):
                generated_data[..., i] = np.clip(generated_data[..., i], self.scaler['min'][i], self.scaler['max'][i])
        
        return generated_data
    
    def _normalize(self, data):
        """normalization to [-1, 1]"""
        min_val = data.min()
        max_val = data.max()
        if max_val - min_val > 0:
            data = 2 * (data - min_val) / (max_val - min_val) - 1
        return data, min_val, max_val
    
    def _denormalize(self, data, min_val, max_val):
        """denormalization"""
        if max_val - min_val > 0:
            data = (data + 1) / 2 * (max_val - min_val) + min_val
        return data
    
    def fit(self, real_data, epochs=1000, batch_size=32, learning_rate=0.0001,
            weight_decay=1e-5, verbose=True, save_scaler=True, save_dir='.', save_freq=50):
        """
        Train model
        """

        if save_scaler:
            self.scaler = {
                'min': real_data.min(axis=(0, 1)),
                'max': real_data.max(axis=(0, 1))
            }

        real_data_norm, _, _ = self._normalize(real_data)
        real_data_tensor = torch.FloatTensor(real_data_norm).to(self.device)

        dataset = TensorDataset(real_data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, 
                               weight_decay=weight_decay, betas=(0.9, 0.999))

        warmup_epochs = min(100, epochs // 10)
        
        def lambda_lr(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        self.loss_history = {'train': []}

        os.makedirs(save_dir, exist_ok=True)
        
        print("\nTraining Transformer Autoregressive Model...")
        
        best_loss = float('inf')
        best_epoch = 0
        best_model_path = os.path.join(save_dir, 'transformer_autoregressive_best.pth')
        
        for epoch in range(epochs):
            total_loss = 0
            self.train()
            
            for batch in dataloader:
                x = batch[0]  # [batch_size, seq_len, feature_dim]
                batch_size_actual = x.size(0)

                input_seq = x[:, :-1, :]  # [batch_size, seq_len-1, feature_dim]
                target_seq = x[:, 1:, :]  # [batch_size, seq_len-1, feature_dim]
                
                # forward pass
                output = self.forward(input_seq, target_seq)
                
                # loss calculation
                loss_mse = mse_loss(output, target_seq)
                loss_l1 = l1_loss(output, target_seq)
                loss = loss_mse + 0.1 * loss_l1
                # loss = loss_mse

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * batch_size_actual
            

            avg_loss = total_loss / len(dataset)
            self.loss_history['train'].append(avg_loss)
            
            # update learning rate
            scheduler.step()
            
            # save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                self.save_model(best_model_path)
                if verbose:
                    print(f"  ★ New best model saved at epoch {best_epoch} (Loss: {best_loss:.6f})")
            
            # save checkpoint periodically
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(save_dir, f'autoregressive_checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'scaler': self.scaler,
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'loss_history': self.loss_history
                }, checkpoint_path)
                if verbose:
                    print(f"  📁 Checkpoint saved: {checkpoint_path}")
            
            if verbose and (epoch + 1) % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                # current_lr = 0.0001
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f} | Best: {best_loss:.6f}")
        
        self._is_trained = True
        print(f"\nTraining completed!")
        print(f"Best model: epoch {best_epoch}, Loss: {best_loss:.6f}")
        print(f"Best model saved to: {best_model_path}")
    
    def save_model(self, filepath):
        """Save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'scaler': self.scaler,
            'seq_len': self.seq_len,
            'feature_dim': self.feature_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        print(f"Model loaded from {filepath}")
        return self
    
    def plot_losses(self, save_path=None):
        """Plot training curve"""
        plt.figure(figsize=(10, 6))
        
        if self.loss_history and 'train' in self.loss_history:
            plt.plot(self.loss_history['train'], label='Training Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Loss plot saved to {save_path}")
            plt.show()
        else:
            print("No loss history to plot")
        
        plt.close()
