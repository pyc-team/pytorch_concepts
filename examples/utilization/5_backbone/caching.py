"""
Example: Explicit Backbone Embedding Precomputation (data side)

A pretrained ``Backbone`` can preprocess a dataset once: embeddings are
computed over the whole dataset, cached to disk, and swapped in as the
dataset's ``input_data``. This is an explicit step — call
``precompute_embeddings`` *before* ``setup()``; ``setup()`` then only handles
splitting.

Flow:
1. Load CelebA, limited to a small subset via ``max_samples``.
2. ``dm.precompute_embeddings(Backbone('resnet18'))`` — computes or loads the
   cache at ``{dataset.root_dir}/bkb_embs_resnet18.pt``.
3. ``dm.setup('fit')`` — splitting only.
4. Train a ConceptBottleneckModel on the embeddings.

The same call exists one level down on the dataset itself:
``dataset.precompute_embeddings(backbone, batch_size=...)``.
"""
import time

import torch
from pytorch_lightning import Trainer

from torch_concepts import Backbone, seed_everything
from torch_concepts.data import CelebADataModule
from torch_concepts.nn import ConceptBottleneckModel, MLP


def main():
    seed_everything(42)

    # 1. Data: CelebA images
    dm = CelebADataModule(
        root='./data/celeba',
        max_samples=1000,
        batch_size=128,
        # splitter=None replaces the native split (defined on the full dataset) 
        # with a random split.
        splitter=None,
        seed=42
    )
    print(f"Raw input shape: {tuple(dm.n_features)}")

    # 2. Explicit preprocessing: precompute backbone embeddings before setup()
    backbone = Backbone('facebook/dinov3-vits16-pretrain-lvd1689m', freeze=True)

    t0 = time.perf_counter()
    dm.precompute_embeddings(backbone, cache=True, force=True)  # first call: computes + saves to disk
    print(f"First precompute_embeddings call:  {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    dm.precompute_embeddings(backbone, cache=True)  # second call: loads the cache
    print(f"Second precompute_embeddings call: {time.perf_counter() - t0:.2f}s (cache hit)")

    print(f"Embedded input shape: {tuple(dm.n_features)} "
          f"(= backbone.out_features = {backbone.out_features})")

    # 3. setup() only splits
    dm.setup('fit')
    print(f"Splits: train={dm.train_len}, val={dm.val_len}, test={dm.test_len}")

    # 4. Train a CBM on the cached embeddings
    model = ConceptBottleneckModel(
        input_size=dm.n_features[-1],
        annotations=dm.annotations,
        task_names=['Attractive'],
        backbone=MLP(backbone.out_features, 128), # embeddings are precomputed, backbone reduces to a simple latent encoder
        latent_size=128,  
        lightning=True,
        loss=torch.nn.BCEWithLogitsLoss(),
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 0.01},
    )
    trainer = Trainer(max_epochs=200, logger=False)
    trainer.fit(model, datamodule=dm)
    trainer.test(ckpt_path='best', datamodule=dm)


if __name__ == '__main__':
    main()
