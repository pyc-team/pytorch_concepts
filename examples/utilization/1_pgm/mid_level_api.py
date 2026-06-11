import warnings
import torch
import torch.nn.functional as F
import torch.nn as nn
import pyro
import pyro.distributions as dist

from torch_concepts.nn import (
    ConceptVariable, EmbeddingVariable, 
    ParametricCPD,
    DeterministicInference, AncestralInference, 
    VariationalInference, RejectionSampling, BayesianNetwork
)

warnings.simplefilter("always")


class ConstParam(nn.Module):
    """Wraps a learnable tensor as a root-CPD parametrization module.

    Root CPD parametrization modules must take no arguments and return the
    distribution parameter tensor.  Wrap ``nn.Parameter`` values in this
    class instead of passing them directly to ``ParametricCPD``.
    """
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.value = nn.Parameter(value.clone())
    def forward(self):
        return self.value


torch.manual_seed(0); pyro.set_rng_seed(0)


### Variables

# Instantiating a single variable
A = ConceptVariable('A', distribution=dist.Normal, size=3)
print(A)

# Instantiating multiple varibales with shared distribution and size
B1, B2 = ConceptVariable(['B1', 'B2'], distribution=dist.Bernoulli, size=1)
print(B1, B2)

# Instantiating multiple variables with shared distribution but different sizes
C1, C2 = ConceptVariable(['C1', 'C2'], distribution=dist.Bernoulli, size=[1, 2]) 
print(C1, C2)

# Instantiating multiple variables with different distributions and sizes
D1, D2 = ConceptVariable(
    names=['D1', 'D2'], 
    distribution=[dist.Bernoulli, dist.Normal], 
    size=[1, 3]
)
print(D1, D2)

### CPDs

# Defining a root variable
root = EmbeddingVariable('root', distribution=dist.Normal, size=2)

# Defining the prior for the root variable
root_cpd = ParametricCPD(
    variable=root, 
    parametrization={
        'loc': ConstParam(torch.zeros(root.size)), 
        'scale': ConstParam(torch.ones(root.size))
    }
)
root_cpd

# ConstParam wraps an nn.Parameter in an nn.Module. Root CPD parametrization
# modules must take no arguments and return the distribution parameter tensor.
print(next(iter(root_cpd.parametrization.parameters())))

# We can make quries without providing constant evidence to the root
inf = DeterministicInference(BayesianNetwork(
    variables=[root], 
    factors=[root_cpd])
)
print(inf.query(query=['root'], evidence={}))

# A root CPD whose modules require input arguments will not fail at construction
# time but will raise a TypeError at runtime when the engine calls them with no arguments.
root_cpd = ParametricCPD(
    variable=root,
    parametrization={
        "loc": nn.Linear(4, 4),
        "scale": nn.Linear(4, 4)
        },
    parents=[]
)
inf = DeterministicInference(BayesianNetwork(variables=[root], factors=[root_cpd]))

# It will raise error since it tries to call the nn.Linear module 
# without providing the required input.
# inf.query(query=['root'], evidence={})    // it will raise error.

# Create non-root cpd by specifying just a torch module
X = ConceptVariable('X', distribution=dist.Normal, shape=(2,))
x_cpd = ParametricCPD(
    variable=X,
    parametrization={
        "loc": nn.Linear(2,4),
        "scale": nn.Linear(2,4)
    },
    parents=[root]
)
print(x_cpd)

# Create cpd specifying the parameters
x_cpd = ParametricCPD(
    variable=X,
    parametrization={'loc': nn.Linear(2,2), 'scale': nn.Linear(2,2)},
    parents=[root]
)
print(x_cpd)



####### Model instantiation

# Helpers

# Build model
def make_pgm():
    # Define variables
    # x = InputVariable('x', size=16)
    x = EmbeddingVariable('x', distribution=dist.Normal, size=16) 
    c1, c2 = ConceptVariable(['c1', 'c2'], distribution=dist.Bernoulli, size=1)
    y = ConceptVariable('y', distribution=dist.Bernoulli, size=1)

    # Define CPDs
    cpd_x  = ParametricCPD(
        x,
        parametrization={
            "loc": ConstParam(torch.zeros(16)),
            "scale": ConstParam(torch.ones(16))
        }
    )
    cpd_c1 = ParametricCPD(c1, parametrization={"probs": nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())}, parents=[x])
    cpd_c2 = ParametricCPD(c2, parametrization={"probs": nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())}, parents=[x])

    cpd_y  = ParametricCPD(y, parametrization={"probs": nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())}, parents=[c1, c2])

    # Assemble PGM
    pgm = BayesianNetwork(
        variables=[x, c1, c2, y],
        factors=[cpd_x, cpd_c1, cpd_c2, cpd_y],
    )
    return pgm


# synthetic data batch: XOR
def make_batch(B=256):
    X = torch.randn(B, 16)
    # Two ground-truth concepts read off two coordinates of x.
    # Shape (B, 1) to match variable shape=(1,).
    C1 = (X[:, 0] > 0).float().unsqueeze(-1)
    C2 = (X[:, 1] > 0).float().unsqueeze(-1)
    Y = (C1.squeeze(-1).long() ^ C2.squeeze(-1).long()).float().unsqueeze(-1)
    return {'x': X, 'c1': C1, 'c2': C2, 'y': Y}

# Config
EPOCHS = 5000


##### Deterministic inference


# Independent training
# Here we train the model using deterministic inference and p_int=1, 
# which states that ground-truth values are always propagated instead of model predictions.
warnings.simplefilter("ignore")

print("=" * 100)
print("Deterministic Inference with all concepts observed")
print("This corresponds to the sequential training of the CBM's paper.")
print()
pgm = make_pgm()
det = DeterministicInference(pgm, p_int=1.0)
optim = torch.optim.Adam(pgm.parameters(), lr=1e-3)
loss_fn = F.binary_cross_entropy

for step in range(EPOCHS):
    batch = make_batch()
    out = det(
        query={'c1': batch['c1'], 'c2': batch['c2'], 'y': batch['y']}, 
        evidence={'x': batch['x']}
    )
    
    loss_c1 = loss_fn(out.model_params['c1']['probs'], batch['c1'])
    loss_c2 = loss_fn(out.model_params['c2']['probs'], batch['c2'])
    loss_y = loss_fn(out.model_params['y']['probs'], batch['y'])
    loss = loss_c1 + loss_c2 + loss_y
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 1000 == 0:
        print(f"Step {step}: Loss {loss.item():.4f} | c1 acc {(out.model_params['c1']['probs'] > 0.5).float().eq(batch['c1']).float().mean().item():.4f} | c2 acc {(out.model_params['c2']['probs'] > 0.5).float().eq(batch['c2']).float().mean().item():.4f} | y acc {(out.model_params['y']['probs'] > 0.5).float().eq(batch['y']).float().mean().item():.4f}")

print()
print("Deterministic inference at test-time (only input x provided as evidence)")
test_batch = make_batch(B=512)
det_test = DeterministicInference(pgm, p_int=0.0)
with torch.no_grad():
    out = det_test(query={'c1': test_batch['c1'], 'c2': test_batch['c2'], 'y': test_batch['y']}, evidence={'x': test_batch['x']})
y_pred = torch.where(out.model_params['y']['probs'] > 0.5, 1.0, 0.0)
y_true = test_batch['y']
acc = (y_pred == y_true).float().mean()
print(f'test accuracy on y: {acc.item():.3f}')
print()


# Joint training
# Here we train the model using deterministic inference and p_int=0, 
# which states that ground-truth values are never propagated and model predictions are always used instead.
print("=" * 100)
print("Deterministic Inference with only the x observed (concepts predicted)")
print("This corresponds to the joint training of the CBM's paper.")
print()
pgm = make_pgm()
det = DeterministicInference(pgm, p_int=0.0)
optim = torch.optim.Adam(pgm.parameters(), lr=1e-3)
loss_fn = F.binary_cross_entropy

for step in range(EPOCHS):
    batch = make_batch()
    out = det(query={'c1': batch['c1'], 'c2': batch['c2'], 'y': batch['y']}, evidence={'x': batch['x']})
    
    loss_c1 = loss_fn(out.model_params['c1']['probs'], batch['c1'])
    loss_c2 = loss_fn(out.model_params['c2']['probs'], batch['c2'])
    loss_y = loss_fn(out.model_params['y']['probs'], batch['y'])
    loss = loss_c1 + loss_c2 + loss_y
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 1000 == 0:
        print(f"Step {step}: Loss {loss.item():.4f} | c1 acc {(out.model_params['c1']['probs'] > 0.5).float().eq(batch['c1']).float().mean().item():.4f} | c2 acc {(out.model_params['c2']['probs'] > 0.5).float().eq(batch['c2']).float().mean().item():.4f} | y acc {(out.model_params['y']['probs'] > 0.5).float().eq(batch['y']).float().mean().item():.4f}")

print()
print("Deterministic inference at test-time (only input x provided as evidence)")
test_batch = make_batch(B=512)
det_test = DeterministicInference(pgm, p_int=0.0)
with torch.no_grad():
    out = det_test(query={'c1': test_batch['c1'], 'c2': test_batch['c2'], 'y': test_batch['y']}, evidence={'x': test_batch['x']})
y_pred = torch.where(out.model_params['y']['probs'] > 0.5, 1.0, 0.0)
y_true = test_batch['y']
acc = (y_pred == y_true).float().mean()
print(f'test accuracy on y: {acc.item():.3f}')
print()


# Random interventions during training
# Here we train the model using deterministic inference and p_int=0.5, 
# which states that ground-truth values are propagated with 50% probability, and model predictions
# are used the rest of the time.
print("=" * 100)
print("Deterministic Inference with only the x + randomly sampled concepts observed (random interventions)")
print("This corresponds to the training performed in the CEM's paper.")
print()
pgm = make_pgm()
det = DeterministicInference(pgm, p_int=0.5)
optim = torch.optim.Adam(pgm.parameters(), lr=1e-3)
loss_fn = F.binary_cross_entropy

for step in range(EPOCHS):
    batch = make_batch()
    out = det(query={'c1': batch['c1'], 'c2': batch['c2'], 'y': batch['y']}, evidence={'x': batch['x']})
    
    loss_c1 = loss_fn(out.model_params['c1']['probs'], batch['c1'])
    loss_c2 = loss_fn(out.model_params['c2']['probs'], batch['c2'])
    loss_y = loss_fn(out.model_params['y']['probs'], batch['y'])
    loss = loss_c1 + loss_c2 + loss_y
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 1000 == 0:
        print(f"Step {step}: Loss {loss.item():.4f} | c1 acc {(out.model_params['c1']['probs'] > 0.5).float().eq(batch['c1']).float().mean().item():.4f} | c2 acc {(out.model_params['c2']['probs'] > 0.5).float().eq(batch['c2']).float().mean().item():.4f} | y acc {(out.model_params['y']['probs'] > 0.5).float().eq(batch['y']).float().mean().item():.4f}")

print()
print("Deterministic inference at test-time (only input x provided as evidence)")
test_batch = make_batch(B=512)
det_test = DeterministicInference(pgm, p_int=0.0)
with torch.no_grad():
    out = det_test(query={'c1': test_batch['c1'], 'c2': test_batch['c2'], 'y': test_batch['y']}, evidence={'x': test_batch['x']})
y_pred = torch.where(out.model_params['y']['probs'] > 0.5, 1.0, 0.0)
y_true = test_batch['y']
acc = (y_pred == y_true).float().mean()
print(f'test accuracy on y: {acc.item():.3f}')
print()



####### Ancestral inference

# Here we will show only the random interventions training scheme for brevity, 
# but the same can be done for the other two schemes as well by just changing the 
# p_int parameter of the AncestralInference engine.

print("=" * 100)
print("Ancestral Inference with only the x + randomly sampled concepts observed (random interventions)")
print("This corresponds to the training performed in the CEM's paper.")
print()
pgm = make_pgm()
det = AncestralInference(pgm, p_int=0.5)
optim = torch.optim.Adam(pgm.parameters(), lr=1e-3)
loss_fn = F.binary_cross_entropy

for step in range(EPOCHS):
    batch = make_batch()
    out = det(query={'c1': batch['c1'], 'c2': batch['c2'], 'y': batch['y']}, evidence={'x': batch['x']})
    
    loss_c1 = loss_fn(out.model_params['c1']['probs'], batch['c1'])
    loss_c2 = loss_fn(out.model_params['c2']['probs'], batch['c2'])
    loss_y = loss_fn(out.model_params['y']['probs'], batch['y'])
    loss = loss_c1 + loss_c2 + loss_y
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 1000 == 0:
        print(f"Step {step}: Loss {loss.item():.4f} | c1 acc {(out.model_params['c1']['probs'] > 0.5).float().eq(batch['c1']).float().mean().item():.4f} | c2 acc {(out.model_params['c2']['probs'] > 0.5).float().eq(batch['c2']).float().mean().item():.4f} | y acc {(out.model_params['y']['probs'] > 0.5).float().eq(batch['y']).float().mean().item():.4f}")

print()
print("Ancestral inference at test-time (only input x provided as evidence)")
test_batch = make_batch(B=512)
det_test = AncestralInference(pgm, p_int=0.0)
with torch.no_grad():
    out = det_test(query={'c1': test_batch['c1'], 'c2': test_batch['c2'], 'y': test_batch['y']}, evidence={'x': test_batch['x']})
c1_pred = out.samples['c1']
c1_true = test_batch['c1']
acc_c1 = (c1_pred == c1_true).float().mean()

c2_pred = out.samples['c2']
c2_true = test_batch['c2']
acc_c2 = (c2_pred == c2_true).float().mean()

y_pred = out.samples['y']
y_true = test_batch['y']
acc = (y_pred == y_true).float().mean()
print(f'test accuracy on c1: {acc_c1.item():.3f}')
print(f'test accuracy on c2: {acc_c2.item():.3f}')
print(f'test accuracy on y: {acc.item():.3f}')
print()



#### Variational inference
# Here cannot train the model to approximate the conditional query $P(C,Y\mid X)$. 
# Indeed, what variational inference does is to approximate the marginal over 
# the observable variables $P(A,B,D)$, with $C$ as the latent, using the ELBO.
# We allow the user to have full control over the guides that have to be set 
# for the unobservable variables.     

# The user can change: the conditioning of the guide, 
# the parametrization and the distribution of the unobservable variable.

print("=" * 100)
print("Variational Inference - Case 2: c1 and c2 latent")
print()
pgm = make_pgm()

c1_guide = ParametricCPD(
    pgm.name_to_variable('c1'),
    {"probs": nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())},
    parents=[pgm.name_to_variable('x')])
c2_guide = ParametricCPD(
    pgm.name_to_variable('c2'),
    {"probs": nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())},
    parents=[pgm.name_to_variable('x')]
)

vi = VariationalInference(
    pgm, 
    latents={
        'c1': c1_guide,
        'c2': c2_guide
    }
)

def bernoulli_kl(q_probs, p_probs):
    q_dist = dist.Bernoulli(probs=q_probs)
    p_dist = dist.Bernoulli(probs=p_probs)
    kl = torch.distributions.kl_divergence(q_dist, p_dist)
    return kl.mean()

def binary_accuracy(pred_probs, true_labels):
    pred_labels = (pred_probs > 0.5).float()
    acc = (pred_labels == true_labels).float().mean()
    return acc

optim = torch.optim.AdamW(pgm.parameters(), lr=1e-3)

for step in range(EPOCHS):
    batch = make_batch()
    # c1 and c2 are latent/unobservable; x and y are passed via query.
    out = vi.query(
        query={'x': batch['x'], 'y': batch['y']},
    )

    recon = F.binary_cross_entropy(out.params['y']['probs'], batch['y'])
    kl_c1 = bernoulli_kl(out.guide_params['c1']['probs'], out.params['c1']['probs'])
    kl_c2 = bernoulli_kl(out.guide_params['c2']['probs'], out.params['c2']['probs'])
    kl = kl_c1 + kl_c2
    loss = recon + kl
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 1000 == 0:
        c1_acc = binary_accuracy(out.guide_params['c1']['probs'].detach(), batch['c1']).item()
        c2_acc = binary_accuracy(out.guide_params['c2']['probs'].detach(), batch['c2']).item()
        y_acc = binary_accuracy(out.params['y']['probs'].detach(), batch['y']).item()
        print(f"Step {step}: recon {recon.item():.4f} | kl {kl.item():.4f} | c1 acc {c1_acc:.4f} | c2 acc {c2_acc:.4f} | y acc {y_acc:.4f}")

print()
print("Ancestral inference at test-time (only input x provided as evidence)")
test_batch = make_batch(B=512)
det_test = AncestralInference(pgm, p_int=0.0)
with torch.no_grad():
    out = det_test(query={'c1': test_batch['c1'], 'c2': test_batch['c2'], 'y': test_batch['y']}, evidence={'x': test_batch['x']})
c1_pred = out.samples['c1']
c1_true = test_batch['c1']
acc_c1 = (c1_pred == c1_true).float().mean()

c2_pred = out.samples['c2']
c2_true = test_batch['c2']
acc_c2 = (c2_pred == c2_true).float().mean()

y_pred = out.samples['y']
y_true = test_batch['y']
acc = (y_pred == y_true).float().mean()
print(f'test accuracy on c1: {acc_c1.item():.3f}')
print(f'test accuracy on c2: {acc_c2.item():.3f}')
print(f'test accuracy on y: {acc.item():.3f}')
print()


###### Rejection sampling
# Here instead, we do not use this inference engine for training but for 
# approximating arbitrary conditional queries.

print("=" * 100)
print("Ancestral Inference with all concepts observed")
print("This corresponds to the sequential training of the CBM's paper.")
print()
pgm = make_pgm()
det = AncestralInference(pgm, p_int=1.0)
optim = torch.optim.Adam(pgm.parameters(), lr=1e-3)
loss_fn = F.binary_cross_entropy

for step in range(EPOCHS):
    batch = make_batch()
    out = det(query={'c1': batch['c1'], 'c2': batch['c2'], 'y': batch['y']}, evidence={'x': batch['x']})
    
    loss_c1 = loss_fn(out.model_params['c1']['probs'], batch['c1'])
    loss_c2 = loss_fn(out.model_params['c2']['probs'], batch['c2'])
    loss_y = loss_fn(out.model_params['y']['probs'], batch['y'])
    loss = loss_c1 + loss_c2 + loss_y
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 1000 == 0:
        print(f"Step {step}: Loss {loss.item():.4f} | c1 acc {(out.model_params['c1']['probs'] > 0.5).float().eq(batch['c1']).float().mean().item():.4f} | c2 acc {(out.model_params['c2']['probs'] > 0.5).float().eq(batch['c2']).float().mean().item():.4f} | y acc {(out.model_params['y']['probs'] > 0.5).float().eq(batch['y']).float().mean().item():.4f}")

print()
print("Ancestral inference at test-time (only input x provided as evidence)")
test_batch = make_batch(B=512)
det_test = AncestralInference(pgm, p_int=0.0)
with torch.no_grad():
    out = det_test(query={'c1': test_batch['c1'], 'c2': test_batch['c2'], 'y': test_batch['y']}, evidence={'x': test_batch['x']})
c1_pred = out.samples['c1']
c1_true = test_batch['c1']
acc_c1 = (c1_pred == c1_true).float().mean()

c2_pred = out.samples['c2']
c2_true = test_batch['c2']
acc_c2 = (c2_pred == c2_true).float().mean()

y_pred = out.samples['y']
y_true = test_batch['y']
acc = (y_pred == y_true).float().mean()
print(f'test accuracy on c1: {acc_c1.item():.3f}')
print(f'test accuracy on c2: {acc_c2.item():.3f}')
print(f'test accuracy on y: {acc.item():.3f}')
print()

# Use rejection sampling to estimate the posterior of c1 given x and y.
print("=" * 100)
print("Rejection Sampling Inference with only the x observed (concepts predicted)")
print()

rejection = RejectionSampling(
    pgm, # the probabilistic model
    n_samples=1000 # the number of samples to produce for the estimation of the posterior distribution
)

out = rejection.query(
    query={'c1': torch.Tensor([[1], [1]])}, 
    evidence={'c2': torch.Tensor([[1],[1]]), 'y': torch.Tensor([[0],[1]])} # we condition on x and y
)

# Here we have evidence on c2 and y.
# We are asking to the engine the probability that P(c1=1 | c2=1, y=0) and P(c1=1 | c2=1, y=1).
# Since y is the xor of c1 and c2, then we expect c1 to be 1 in the first case and zero in the second case.
print(out.probabilities[0], out.probabilities[1])