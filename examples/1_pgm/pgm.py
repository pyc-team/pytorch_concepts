import torch
from torch.distributions import Bernoulli, Categorical
from torch.nn import Linear

from torch_concepts import Variable
from torch_concepts.distributions import Delta
from torch_concepts.nn import Factor, HyperLinearPredictor, ProbEncoderFromEmb, ExogEncoder, \
    ProbabilisticGraphicalModel, ForwardInference, ProbEncoderFromExog

latent_dims = 10
torch.manual_seed(42)
batch_size = 87

# Variable setup
emb_variable = Variable(["emb"], parents=[], distribution=Delta, size=7, metadata={"type": "embedding"})
exog_variable = Variable(["c1ex", "c2ex"], parents=["emb"], distribution=Delta, size=5)
ca_variable = Variable(["c1"], parents=["c1ex"], distribution=Bernoulli, size=1)
ca_variable2 = Variable(["c2"], parents=["c2ex"], distribution=Bernoulli, size=1)
cc_variable = Variable(["xor_class"], parents=["c1", "c2"]+[f"xor_class_ex"], distribution=Categorical, size=4, metadata={"target": True})
exog_variable_cat = Variable([f"xor_class_ex"], parents=["emb"], distribution=Delta, size=5)
cc2_variable = Variable(["xor_class2"], parents=["c1", "c2"], distribution=Bernoulli, size=1, metadata={"target": True})

# Factor setup
emb_factor = Factor(["emb"], module_class=Linear(in_features=latent_dims, out_features=emb_variable.size))
exog_factor = Factor(["c1ex", "c2ex"], module_class=ExogEncoder(in_features_embedding=emb_variable.out_features, embedding_size=11, out_features=ca_variable.size))
ca_factor = Factor(["c1"], module_class=ProbEncoderFromExog(in_features_exogenous=11, out_features=ca_variable.size))
ca_factor2 = Factor(["c2"], module_class=ProbEncoderFromExog(in_features_exogenous=11, out_features=ca_variable2.size))
exog_factor_cat = Factor([f"xor_class_ex"], module_class=ExogEncoder(in_features_embedding=emb_variable.out_features, embedding_size=11, out_features=cc_variable.size))
cc_factor = Factor(["xor_class"], module_class=HyperLinearPredictor(in_features_logits=ca_variable.out_features + ca_variable2.out_features, in_features_exogenous=11, embedding_size=19, out_features=cc_variable.size))
cc_factor2 = Factor(["xor_class2"], module_class=Linear(in_features=ca_variable.out_features + ca_variable2.out_features, out_features=cc2_variable.size))

# PGM Initialization
model = ProbabilisticGraphicalModel(
    variables=[emb_variable, exog_variable, ca_variable, ca_variable2, cc_variable, cc2_variable, exog_variable_cat],
    factors=[emb_factor, exog_factor, ca_factor, ca_factor2, cc_factor, cc_factor2, exog_factor_cat]
)

# get cpt
f = model.get_factor_of_variable("xor_class2")
cpt = f.build_cpt()
print("CPT for 'xor_class2':")
print(cpt)

# get potential
potential = f.build_potential()
print("Potential for 'xor_class2':")
print(potential)

# --- Inference Usage ---
print("## PGM Inference Query Results")
print("---")

# 1. Initialize Inference
inference_engine = ForwardInference(model)

# 2. Define initial input for the root node ('emb') matching the user's desired format
initial_latent_input = torch.randn(batch_size, latent_dims)
initial_input = {'emb': initial_latent_input}

# 3. Predict all concepts using the new .query() method
query_concepts = ["c2", "c1", "xor_class"]
concept_preds_tensor = inference_engine.query(query_concepts, evidence=initial_input)

# 4. Print results
print(f"Query Concepts: {query_concepts}")
print(f"Batch Size: {batch_size}, Latent Dims: {latent_dims}\n")
print(f"Topological Order: {[v.concepts[0] for v in inference_engine.sorted_variables]}\n")
print(f"Resulting Tensor Shape: {concept_preds_tensor.shape}")
print(f"Resulting Tensor (first row, all 6 features): {concept_preds_tensor[0].tolist()}")

print("---")