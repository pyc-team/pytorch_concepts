import logging
import os
import pandas as pd
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns


from pytorch_lightning import Trainer
from torch_concepts.nn.models import AVAILABLE_MODELS
from torchvision import transforms
from utils import set_seed, GaussianNoiseTransform



def plot_metric(
    results,
    run_names,
    metric_name,
    save_path=None,
    title="",
    show=False,
):
    """
        Plot the accuracy of all models on the test set.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    sns.barplot(x="model", y=metric_name, data=results, ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)
    if title:
        ax.set_title(title, fontsize=24)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()



def plot_intervenability(results, save_path=None, show=False):
    """
    Plot the intervenability of the models on the test set.
    For each noise level, plot the test accuracy as a function of the
    intervention probability. The plot will have as many subplots as the
    noise levels.
    """
    # subplots as the noise levels
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    sns.lineplot(x="int_prob", y="test_y_acc", hue="model", data=results, ax=ax)
    ax.set_xlabel("Intervention probability")
    ax.set_ylabel("Test accuracy")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()