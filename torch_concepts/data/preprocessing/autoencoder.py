"""
Autoencoder preprocessing for dimensionality reduction.

This module provides autoencoder-based preprocessing to learn low-dimensional
representations of high-dimensional concept data.
"""
import torch.nn as nn
import torch
import torch.optim as optim
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SimpleAutoencoder(nn.Module):
    """
    Simple feedforward autoencoder for dimensionality reduction.

    A standard autoencoder with encoder and decoder networks using ReLU activations.
    Useful for preprocessing high-dimensional concept spaces.

    Attributes:
        encoder (nn.Sequential): Encoder network.
        decoder (nn.Sequential): Decoder network.

    Args:
        input_shape: Number of input features.
        latent_dim: Dimension of the latent space.

    Example:
        >>> import torch
        >>> from torch_concepts.data.preprocessing.autoencoder import SimpleAutoencoder
        >>>
        >>> # Create autoencoder
        >>> autoencoder = SimpleAutoencoder(input_shape=784, latent_dim=32)
        >>>
        >>> # Forward pass
        >>> x = torch.randn(4, 784)
        >>> encoded, decoded = autoencoder(x)
        >>> print(f"Encoded shape: {encoded.shape}")
        Encoded shape: torch.Size([4, 32])
        >>> print(f"Decoded shape: {decoded.shape}")
        Decoded shape: torch.Size([4, 784])
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
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_shape).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (encoded, decoded) where
                - encoded has shape (batch_size, latent_dim)
                - decoded has shape (batch_size, input_shape)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoencoderTrainer:
    """
    Trainer class for autoencoder models with early stopping.

    Provides training loop, early stopping, and latent representation extraction
    for autoencoder models.

    Attributes:
        model (SimpleAutoencoder): The autoencoder model.
        criterion (nn.MSELoss): Reconstruction loss function.
        optimizer (optim.Adam): Optimizer for training.
        device (str): Device to train on ('cpu' or 'cuda').

    Args:
        input_shape: Number of input features.
        noise: Noise level to add to latent representations (default: 0.5).
        latent_dim: Dimension of latent space (default: 32).
        lr: Learning rate (default: 0.0005).
        epochs: Maximum training epochs (default: 2000).
        batch_size: Batch size for training (default: 512).
        patience: Early stopping patience in epochs (default: 50).
        device: Device to use for training (default: 'cpu').

    Example:
        >>> import torch
        >>> from torch_concepts.data.preprocessing.autoencoder import AutoencoderTrainer
        >>>
        >>> # Create synthetic data
        >>> data = torch.randn(1000, 100)
        >>>
        >>> # Create and train autoencoder
        >>> trainer = AutoencoderTrainer(
        ...     input_shape=100,
        ...     latent_dim=16,
        ...     epochs=100,
        ...     batch_size=64,
        ...     device='cpu'
        ... )
        >>>
        >>> # Train
        >>> trainer.train(data)
        Autoencoder training started...
        >>>
        >>> # Extract latent representations
        >>> latent = trainer.extract_latent()
        >>> print(latent.shape)
        torch.Size([1000, 16])
    """
    def __init__(
            self, 
            input_shape: int, 
            noise: float = 0.,
            latent_dim: int = 32,
            lr: float = 0.0005,
            epochs: int = 2000,
            batch_size: int = 512,
            patience: int = 50,
            device=None
    ):  
        self.noise_level = noise
        self.latend_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = SimpleAutoencoder(input_shape, self.latend_dim)
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.best_model_wts = None

    def train(self, dataset):
        """
        Train the autoencoder on the given dataset.

        Implements training loop with MSE reconstruction loss and early stopping
        based on validation loss.

        Args:
            dataset: PyTorch dataset or tensor to train on.
        """
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size)
        
        best_loss = float('inf')
        patience_counter = 0

        logger.info('Autoencoder training started...')
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_loss = 0.0
            for data in self.data_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                _, outputs = self.model(data)
                loss = self.criterion(outputs, data)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(self.data_loader)

            if epoch % 300 == 0:
                logger.info(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}')

            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                self.best_model_wts = self.model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                logger.info('Early stopping')
                break

        logger.info(f'Epoch {epoch+1}/{self.epochs}, Final Train Loss: {train_loss:.4f}')
        self.is_fitted = True

    def extract_latent(self):
        """
        Extract latent representations from the trained autoencoder.

        Uses the best model weights (lowest reconstruction loss) to encode
        the entire dataset. Optionally adds noise to latent representations.

        Returns:
            torch.Tensor: Latent representations of shape (n_samples, latent_dim).

        Example:
            >>> # After training
            >>> latent = trainer.extract_latent()
            >>> print(latent.shape)
            torch.Size([1000, 16])
        """
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


def extract_embs_from_autoencoder(
        df, 
        autoencoder_kwargs={}
    ):
    """
    Extract embeddings from a pandas DataFrame using an autoencoder.

    Convenience function that trains an autoencoder on tabular data and
    returns the learned latent representations.

    Args:
        df: Input pandas DataFrame.
        autoencoder_kwargs: Dictionary of keyword arguments for AutoencoderTrainer.
            Can include 'device' to specify training device (default: 'cpu').

    Returns:
        torch.Tensor: Latent representations of shape (n_samples, latent_dim).

    Example:
        >>> import pandas as pd
        >>> import torch
        >>> from torch_concepts.data.preprocessing.autoencoder import extract_embs_from_autoencoder
        >>>
        >>> # Create sample DataFrame
        >>> df = pd.DataFrame(torch.randn(100, 50).numpy())
        >>>
        >>> # Extract embeddings
        >>> embeddings = extract_embs_from_autoencoder(
        ...     df,
        ...     autoencoder_kwargs={
        ...         'latent_dim': 10,
        ...         'epochs': 50,
        ...         'batch_size': 32,
        ...         'noise': 0.1,
        ...         'device': 'cpu'  # or 'cuda' if desired
        ...     }
        ... )
        >>> print(embeddings.shape)
        torch.Size([100, 10])
    """
    # Convert DataFrame to tensor
    data = torch.tensor(df.values, dtype=torch.float32)
    
    # Train autoencoder
    trainer = AutoencoderTrainer(
        input_shape=data.shape[1],
        **autoencoder_kwargs
    )
    
    # Train and get transformed dataset
    trainer.train(data)
    latent = trainer.extract_latent()
    return latent
