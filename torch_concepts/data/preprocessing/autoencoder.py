import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class SimpleAutoencoder(nn.Module):
    """A simple feedforward autoencoder.
    Args:
        input_shape (int): The number of input features.
        latent_dim (int): The dimension of the latent space.
    """
    def __init__(self, input_shape, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(0.1),
            nn.Linear(latent_dim, input_shape),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoencoderTrainer:
    def __init__(
            self, 
            input_shape: int, 
            noise: float = 0.5,
            latent_dim: int = 32,
            lr: float = 0.0005,
            epochs: int = 2000,
            batch_size: int = 512,
            patience: int = 50,
            device='cpu'
    ):  
        self.noise_level = noise
        self.latend_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.model = SimpleAutoencoder(input_shape, self.latend_dim)
        self.model.to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = device

    def train(self, dataset):
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size)
        
        best_loss = float('inf')
        patience_counter = 0

        print('Autoencoder training started...')
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_loss = 0.0
            for data in self.data_loader:
                if 'cuda' in self.device:
                    data = data.to(self.device)
                self.optimizer.zero_grad()
                _, outputs = self.model(data)
                loss = self.criterion(outputs, data)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            if epoch % 300 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}')

            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                best_model_wts = self.model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print('Early stopping')
                break

        print(f'Epoch {epoch+1}/{self.epochs}, Final Train Loss: {train_loss:.4f}')
        self.best_model_wts = best_model_wts

    def extract_latent(self):
        # Generate the latent representations
        self.model.load_state_dict(self.best_model_wts)
        self.model.eval()
        latent = []
        with torch.no_grad():
            for data in self.data_loader:
                data = data.to(self.device)
                encoded, _ = self.model(data)
                if self.noise_level > 0:
                    encoded = (1 - self.noise_level)*encoded + self.noise_level*torch.randn_like(encoded)
                latent.append(encoded)

        latent = torch.cat(latent, dim=0)
        return latent


def extract_embs_from_autoencoder(df, autoencoder_kwargs):
    """Extract embeddings from a pandas DataFrame using an autoencoder.
    
    Args:
        df (pd.DataFrame): Input data
        autoencoder_kwargs (dict): Configuration for the autoencoder
        
    Returns:
        torch.Tensor: Latent representations of the input data
    """
    # Convert DataFrame to tensor
    data = torch.tensor(df.values, dtype=torch.float32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train autoencoder
    trainer = AutoencoderTrainer(
        input_shape=data.shape[1],
        device=device,
        **autoencoder_kwargs
    )
    
    # Train and get transformed dataset
    trainer.train(data)
    latent = trainer.extract_latent()
    return latent