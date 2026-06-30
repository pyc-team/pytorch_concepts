import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from torch.func import jacrev, vmap, functional_call
import torch_concepts as pyc

pyc.seed_everything(42)


def load_model(model_path, model_name):
    print(f"Loading model from local path: {model_name}")
    print("Applying laptop-optimized fast inference settings...")

    full_model_path = os.path.join(model_path, model_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(full_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Simplified model loading for better stability
    model = AutoModelForCausalLM.from_pretrained(
        full_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,  # Disable cache initially for debugging
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    return tokenizer, model


class QwenHead(torch.nn.Module):
    def __init__(self, norm, head):
        super().__init__()
        self.norm = norm
        self.head = head

    def forward(self, x, mode="all"):
        normalized_final = self.norm(x)
        if mode == "max":
            logits = self.head(normalized_final.unsqueeze(0)[0, -1, :])
            max_logit, _ = torch.max(logits, dim=-1)
            logits = max_logit.unsqueeze(0)
        else:
            logits = self.head(normalized_final[0, -1, :])
        return logits


class QwenModelWrapper(torch.nn.Module):
    def __init__(self, language_model, tokenizer, device):
        super().__init__()
        self.language_model = language_model
        # Get the token IDs for "Yes" and "No"
        # Note: Tokenizers are case-sensitive and space-sensitive.
        # Usually, you want the version with a leading space.
        yes_id = tokenizer.convert_tokens_to_ids("Yes")
        no_id = tokenizer.convert_tokens_to_ids("No")
        # Use a tensor of indices to keep it differentiable
        self.target_indices = torch.tensor([yes_id, no_id], device=device)
        self.device = device

    def forward(self, input_embeds, attention_mask):
        outputs = self.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask)

        # This slice maintains the gradient path back to the model weights
        relevant_logits = outputs.logits[0, -1, self.target_indices]

        # Apply softmax to just these two
        relative_probs = F.softmax(relevant_logits, dim=0)

        prob_yes_rel = relative_probs[0]
        return prob_yes_rel


def gradient_alignment_loss(g_c, g_y):
    # 2. Get Orthonormal Bases for the subspaces using QR decomposition
    # We transpose because we want the basis for the row-space (the gradients)
    # g_c.transpose(1, 2) is (Batch, dim_z, dim_out)
    q_c, _ = torch.linalg.qr(g_c.transpose(1, 2))
    q_y, _ = torch.linalg.qr(g_y.transpose(1, 2))

    # 3. Compute the Projection Matrices: P = Q @ Q.T
    # This matrix represents the subspace spanned by the gradients
    p_c = torch.bmm(q_c, q_c.transpose(1, 2))
    p_y = torch.bmm(q_y, q_y.transpose(1, 2))

    # 4. The Loss: Frobenius norm of the difference
    # This is 0 if subspaces are identical, and maxed if they are orthogonal
    loss = torch.linalg.matrix_norm(p_c - p_y, ord='fro')

    return loss.mean()


def normalized_gradient_alignment_metric(g_c, g_y):
    """
    g_c, g_y shape: (Batch=1, InputDim, EmbDim)
    We treat InputDim as the number of vectors spanning the subspace.
    """
    # 1. Get Orthonormal Bases
    # We want the basis for the space spanned by the InputDim vectors.
    # QR expects (M, N) and returns Q (M, K).
    # We transpose to (EmbDim, InputDim) to find the basis in EmbDim space.
    q_c, _ = torch.linalg.qr(g_c.reshape([1, 1, -1]).transpose(1, 2), mode='reduced')
    q_y, _ = torch.linalg.qr(g_y.reshape([1, 1, -1]).transpose(1, 2), mode='reduced')

    # k is the number of basis vectors (rank)
    k_c = q_c.shape[2]
    k_y = q_y.shape[2]

    # 3. Compute the Projection Matrices: P = Q @ Q.T
    # This matrix represents the subspace spanned by the gradients
    p_c = torch.bmm(q_c, q_c.transpose(1, 2))
    p_y = torch.bmm(q_y, q_y.transpose(1, 2))

    # 3. Frobenius Norm of the difference
    raw_loss = torch.linalg.matrix_norm(p_c - p_y, ord='fro')

    # 4. Normalization
    # The max distance between two subspaces of rank k_c and k_y
    # is sqrt(k_c + k_y) if they are completely orthogonal.
    max_val = torch.sqrt(torch.tensor(k_c + k_y, dtype=raw_loss.dtype, device=raw_loss.device))

    return raw_loss / max_val


def compute_gradients(language_model, tokenizer, inputs2, device):
    # predict the next token for inputs2 only using the language model
    embed_layer = language_model.get_input_embeddings()
    input_embeds = embed_layer(inputs2['input_ids'])
    input_embeds.retain_grad()

    model = QwenModelWrapper(language_model, tokenizer, device)

    prob_yes_rel = model(input_embeds, inputs2['attention_mask'])

    print(f"Relative Probability 'Yes': {prob_yes_rel.item():.2%}")
    print(f"Relative Probability 'No':  {(1-prob_yes_rel).item():.2%}")

    language_model.zero_grad()
    prob_yes_rel.backward()
    input_gradients = input_embeds.grad

    return input_gradients


def generate_text(language_model, tokenizer, prompt1, prompt2, max_length=50, device=torch.device("cpu")):
    inputs1 = tokenizer(
        prompt1,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=256
    )
    inputs1 = {k: v.to(device) for k, v in inputs1.items()}

    inputs2 = tokenizer(
        prompt2,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=256
    )
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}

    # For simplicity, we concatenate the two prompts and treat them as a single input sequence
    inputs = {k: torch.cat([inputs1[k], inputs2[k][:, 1:]], dim=1) for k in inputs1.keys()}

    g_c = compute_gradients(language_model, tokenizer, inputs2, device)
    g_y = compute_gradients(language_model, tokenizer, inputs, device)[:, inputs1["input_ids"].shape[1]-1:, :]

    return g_c, g_y


def main():
    os.makedirs("llm_experiment", exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model_name = "qwen2.5-3b"
    model_path = './models/'

    tokenizer, model = load_model(model_path=model_path, model_name=model_name)

    prompt1 = ("Tell me whether the following statement is a complete explanation of the question that will follow. "
               "Statement: for the transitive property of inequalities, if A is older than B and B is older than C, then A is older than C. "
               "Do not answer the question yet, just say whether the statement is a complete explanation of the question.")
    prompt2 = ("Know that Socrates is older than Plato and Plato is older than Aristotle. "
               "Question: Is Socrates older than Plato?")
    g_c, g_y = generate_text(model, tokenizer, prompt1, prompt2, max_length=2, device=device)
    print(f"Prompt:\n{prompt1}")
    print(f"{prompt2}")

    # Compute the Symmetry II Metric (Gradient Alignment Loss)
    symmetryII_metric = normalized_gradient_alignment_metric(g_c.cpu().float(), g_y.cpu().float())

    print()
    print(f"Symmetry II Metric (Gradient Alignment Loss): {symmetryII_metric.item():.4f}")
    with open(os.path.join("llm_experiment", "results.csv"), "a") as f:
        f.write(f"{model_name},{symmetryII_metric.item()}\n")

if __name__ == "__main__":
    main()
