import torch
import itertools



def clamp_messages_to_evidence(messages, evidence, md, eps=1e-20):
    """
    Clamp messages so that observed variables become delta distributions.

    messages: [B, total_edge_states]   (this can be v2f or f2v)
    evidence: dict {var_name: observed_state}  (same evidence for all B)
    md:       metadata from build_graph_metadata

    Returns:
        messages_clamped: [B, total_edge_states]
    """
    B, S = messages.shape
    assert S == md["total_edge_states"]

    var_names = md["var_names"]
    var_arity = md["var_arity"]
    var_state_offset = md["var_state_offset"]      # [V]
    vs_id_for_edge_state = md["vs_id_for_edge_state"]  # [S]
    edge_id = md["edge_id_per_state"]             # [S]
    E = md["E"]

    # 1) Build a boolean mask over variable-states: which (var, state) are allowed?
    num_vs = md["total_var_states"]
    allowed_vs = torch.ones(num_vs, dtype=torch.bool, device=messages.device)

    for vname, s_obs in evidence.items():
        v = var_names.index(vname)
        a = int(var_arity[v])
        start = int(var_state_offset[v])

        # default: disallow all states of v
        allowed_vs[start:start + a] = False
        # allow only the observed state
        allowed_vs[start + int(s_obs)] = True

    # 2) Map this to edge-states
    allowed_es = allowed_vs[vs_id_for_edge_state]  # [S]

    # 3) Zero out forbidden edge-states
    messages_clamped = messages.clone()
    messages_clamped[:, ~allowed_es] = 0.0

    # 4) Renormalize per edge (still fully tensorized)
    edge_id_b = edge_id.unsqueeze(0).expand(B, -1)     # [B, S]
    sum_per_edge = torch.zeros(B, E,
                               device=messages.device,
                               dtype=messages.dtype)
    sum_per_edge.scatter_add_(1, edge_id_b, messages_clamped)
    norm = sum_per_edge.gather(1, edge_id_b) + eps
    messages_clamped = messages_clamped / norm

    return messages_clamped


# ------------------------------------------------------------------
# 1. Build global metadata / indexing
# ------------------------------------------------------------------

def build_graph_metadata(variables, factors):
    """
    variables: dict {var_name: arity}
    factors:   dict {factor_name: [var_name1, var_name2, ...]}  (ordered scope)
    """
    # ----- variables -----
    var_names = list(variables.keys())
    V = len(var_names)
    var_index = {name: i for i, name in enumerate(var_names)}
    var_arity = torch.tensor([variables[name] for name in var_names], dtype=torch.long)

    # ----- factors & edges -----
    factor_names = list(factors.keys())
    F = len(factor_names)

    edge2var = []
    edge2factor = []
    edge_pos_in_factor = []
    factor_deg = []
    factor_edge_offset = []
    E = 0
    for fi, fname in enumerate(factor_names):
        scope = factors[fname]  # list of var names, ordered
        factor_edge_offset.append(E)
        factor_deg.append(len(scope))
        for j, vname in enumerate(scope):
            edge2var.append(var_index[vname])
            edge2factor.append(fi)
            edge_pos_in_factor.append(j)
            E += 1

    factor_edge_offset = torch.tensor(factor_edge_offset, dtype=torch.long)
    factor_deg = torch.tensor(factor_deg, dtype=torch.long)
    edge2var = torch.tensor(edge2var, dtype=torch.long)
    edge2factor = torch.tensor(edge2factor, dtype=torch.long)
    edge_pos_in_factor = torch.tensor(edge_pos_in_factor, dtype=torch.long)
    edge_arity = var_arity[edge2var]  # arity per edge

    # ----- edge-state indexing: each (edge, state) gets a global index -----
    edge_state_offset = torch.zeros(E, dtype=torch.long)
    offset = 0
    for e in range(E):
        edge_state_offset[e] = offset
        offset += int(edge_arity[e])
    total_edge_states = int(offset)

    # edge_id_per_state[g] = which edge does global state g belong to?
    edge_id_per_state = torch.empty(total_edge_states, dtype=torch.long)
    for e in range(E):
        a = int(edge_arity[e])
        edge_id_per_state[edge_state_offset[e]:edge_state_offset[e]+a] = e

    # ----- variable-state indexing: each (var, state) gets a group id -----
    var_state_offset = torch.zeros(V, dtype=torch.long)
    off = 0
    for v in range(V):
        var_state_offset[v] = off
        off += int(var_arity[v])
    total_var_states = int(off)

    # vs_id_for_edge_state[g] = id of (var, state) for global edge state g
    vs_id_for_edge_state = torch.empty(total_edge_states, dtype=torch.long)
    for e in range(E):
        v = int(edge2var[e])
        a = int(edge_arity[e])
        start = int(edge_state_offset[e])
        for s in range(a):
            vs_id_for_edge_state[start + s] = var_state_offset[v] + s

    # ----- factor assignments + triples (assignment, edge, state) -----
    factor_num_assign = []
    factor_assign_offset = torch.zeros(F, dtype=torch.long)
    all_triple_fa = []
    all_triple_edge = []
    all_triple_state_in_edge = []
    off_assign = 0

    for fi, fname in enumerate(factor_names):
        scope = factors[fname]
        arities = [variables[vname] for vname in scope]
        num_assign = 1
        for a in arities:
            num_assign *= a
        factor_num_assign.append(num_assign)
        factor_assign_offset[fi] = off_assign

        # edges for this factor are contiguous
        start_edge = int(factor_edge_offset[fi])

        # enumerate assignments in lexicographic order over the scope
        for local_idx, local_assign in enumerate(itertools.product(*[range(a) for a in arities])):
            fa = off_assign + local_idx  # global assignment id
            # for each var in factor, we store a triple row
            for j, vname in enumerate(scope):
                edge = start_edge + j
                state = local_assign[j]
                all_triple_fa.append(fa)
                all_triple_edge.append(edge)
                all_triple_state_in_edge.append(state)

        off_assign += num_assign

    total_assignments = off_assign
    triple2fa = torch.tensor(all_triple_fa, dtype=torch.long)               # [T]
    triple2edge = torch.tensor(all_triple_edge, dtype=torch.long)           # [T]
    triple_state_in_edge = torch.tensor(all_triple_state_in_edge, dtype=torch.long)  # [T]
    T = triple2fa.shape[0]

    # factor index per assignment
    fa2factor = torch.empty(total_assignments, dtype=torch.long)
    for fi in range(F):
        n = factor_num_assign[fi]
        start = int(factor_assign_offset[fi])
        fa2factor[start:start+n] = fi

    metadata = dict(
        var_names=var_names,
        factor_names=factor_names,
        var_arity=var_arity,
        edge2var=edge2var,
        edge2factor=edge2factor,
        edge_pos_in_factor=edge_pos_in_factor,
        edge_arity=edge_arity,
        edge_state_offset=edge_state_offset,
        edge_id_per_state=edge_id_per_state,
        var_state_offset=var_state_offset,
        vs_id_for_edge_state=vs_id_for_edge_state,
        factor_edge_offset=factor_edge_offset,
        factor_deg=factor_deg,
        factor_assign_offset=factor_assign_offset,
        factor_num_assign=torch.tensor(factor_num_assign, dtype=torch.long),
        fa2factor=fa2factor,
        triple2fa=triple2fa,
        triple2edge=triple2edge,
        triple_state_in_edge=triple_state_in_edge,
        total_edge_states=total_edge_states,
        total_var_states=total_var_states,
        total_assignments=total_assignments,
        T=T,
        E=E,
        V=V,
        F=F,
    )
    return metadata


# ------------------------------------------------------------------
# 2. Variable -> Factor messages (tensorized, no loops)
# ------------------------------------------------------------------

def update_var_to_factor(messages_f2v, md, eps=1e-20):
    """
    messages_f2v: [B, total_edge_states]
        factor->variable messages, stored per (edge,state).
    Returns:
        messages_v2f: [B, total_edge_states]
    """
    B, S = messages_f2v.shape
    assert S == md["total_edge_states"]

    vs_id = md["vs_id_for_edge_state"]      # [S], group id for each (edge,state) -> (var,state)
    num_vs = md["total_var_states"]

    # log-domain so product over neighbors becomes sum
    log_m_f2v = torch.log(messages_f2v + eps)       # [B, S]
    vs_id_b = vs_id.unsqueeze(0).expand(B, -1)      # [B, S]

    # sum logs per (var,state)
    log_sum_vs = torch.zeros(B, num_vs,
                             device=messages_f2v.device,
                             dtype=messages_f2v.dtype)
    log_sum_vs.scatter_add_(1, vs_id_b, log_m_f2v)

    # for each edge-state, retrieve total for its (var,state)
    total_for_edge_state = log_sum_vs.gather(1, vs_id_b)  # [B, S]

    # exclude self: sum_{g != current factor} log m_{g->v}
    log_m_v2f = total_for_edge_state - log_m_f2v

    # back to probability domain
    m_v2f = torch.exp(log_m_v2f)

    # normalize per edge
    edge_id = md["edge_id_per_state"]   # [S]
    E = md["E"]
    edge_id_b = edge_id.unsqueeze(0).expand(B, -1)
    sum_per_edge = torch.zeros(B, E,
                               device=m_v2f.device,
                               dtype=m_v2f.dtype)
    sum_per_edge.scatter_add_(1, edge_id_b, m_v2f)
    norm = sum_per_edge.gather(1, edge_id_b) + eps
    m_v2f = m_v2f / norm

    return m_v2f


# ------------------------------------------------------------------
# 3. Factor -> Variable messages (tensorized, no loops)
# ------------------------------------------------------------------

def update_factor_to_var(messages_v2f, factor_eval_list, md, eps=1e-20):
    """
    messages_v2f: [B, total_edge_states]
        variable->factor messages, per (edge,state).
    factor_eval_list: list length F
        factor_eval_list[fi] has shape [B, num_assign_fi] in the SAME assignment
        ordering used in build_graph_metadata (lexicographic over scope).
    Returns:
        messages_f2v: [B, total_edge_states]
    """
    B, S = messages_v2f.shape
    assert S == md["total_edge_states"]

    # concat all factor potentials along assignment dimension
    phi_flat = torch.cat(factor_eval_list, dim=1)  # [B, total_assignments]
    assert phi_flat.shape[1] == md["total_assignments"]

    triple2fa = md["triple2fa"]                     # [T]
    triple2edge = md["triple2edge"]                 # [T]
    triple_state_in_edge = md["triple_state_in_edge"]   # [T]
    edge_state_offset = md["edge_state_offset"]
    total_assignments = md["total_assignments"]
    T = md["T"]

    # global edge-state index for each triple
    # esi[t] = edge_state_offset[edge] + local_state
    esi = edge_state_offset[triple2edge] + triple_state_in_edge  # [T]

    # gather incoming messages for each (assignment, var)
    m_for_triple = messages_v2f[:, esi]  # [B, T]

    # compute product over vars for each assignment via log-sum trick
    log_m_for_triple = torch.log(m_for_triple + eps)
    fa_id_b = triple2fa.unsqueeze(0).expand(B, -1)  # [B, T]

    sum_log_m_per_fa = torch.zeros(B, total_assignments,
                                   device=messages_v2f.device,
                                   dtype=messages_v2f.dtype)
    sum_log_m_per_fa.scatter_add_(1, fa_id_b, log_m_for_triple)
    prod_m_per_fa = torch.exp(sum_log_m_per_fa)  # [B, total_assignments]

    # multiply by factor potentials: weight per assignment
    weight_per_fa = phi_flat * prod_m_per_fa  # [B, total_assignments]

    # for each triple, remove its own variable's contribution from the product
    weight_without_self = weight_per_fa[:, triple2fa] / (m_for_triple + eps)  # [B, T]

    # sum over assignments grouped by (edge,state)
    esi_b = esi.unsqueeze(0).expand(B, -1)  # [B, T]
    messages_f2v_num = torch.zeros(B, S,
                                   device=messages_v2f.device,
                                   dtype=messages_v2f.dtype)
    messages_f2v_num.scatter_add_(1, esi_b, weight_without_self)

    # normalize per edge
    edge_id = md["edge_id_per_state"]  # [S]
    E = md["E"]
    edge_id_b = edge_id.unsqueeze(0).expand(B, -1)
    sum_per_edge = torch.zeros(B, E,
                               device=messages_v2f.device,
                               dtype=messages_f2v_num.dtype)
    sum_per_edge.scatter_add_(1, edge_id_b, messages_f2v_num)
    norm = sum_per_edge.gather(1, edge_id_b) + eps
    messages_f2v = messages_f2v_num / norm

    return messages_f2v


# ------------------------------------------------------------------
# 4. (Optional) helper: variable marginals from factor->var messages
# ------------------------------------------------------------------

def compute_var_marginals(messages_f2v, md, eps=1e-20):
    """
    Approximate variable marginals from final factor->variable messages.
    This does use a small Python loop over variables, but it's not in the
    hot path of message propagation.
    """
    B, S = messages_f2v.shape
    vs_id = md["vs_id_for_edge_state"]
    num_vs = md["total_var_states"]
    var_arity = md["var_arity"]
    V = md["V"]
    var_state_offset = md["var_state_offset"]

    log_m_f2v = torch.log(messages_f2v + eps)
    vs_id_b = vs_id.unsqueeze(0).expand(B, -1)

    log_sum_vs = torch.zeros(B, num_vs,
                             device=messages_f2v.device,
                             dtype=messages_f2v.dtype)
    log_sum_vs.scatter_add_(1, vs_id_b, log_m_f2v)

    marginals = []
    for v in range(V):
        a = int(var_arity[v])
        start = int(var_state_offset[v])
        m_v = torch.exp(log_sum_vs[:, start:start + a])   # [B, a]
        m_v = m_v / (m_v.sum(dim=-1, keepdim=True) + eps)
        marginals.append(m_v)
    return marginals



def compute_exact_marginals_bruteforce(variables, factors, factor_eval_list, md, eps=1e-20):
    """
    Exact marginals by enumerating all assignments of all variables.

    variables: dict {var_name: arity}
    factors:   dict {factor_name: [var_name1, ...]}    (same order as factor_eval_list)
    factor_eval_list: list length F
        factor_eval_list[fi]: [B, num_assign_fi], in SAME assignment ordering
        as build_graph_metadata (lexicographic over factor scope).
    md: metadata from build_graph_metadata

    Returns:
        exact_marginals: list of length V
            exact_marginals[v] has shape [B, arity_v]
    """
    var_names = md["var_names"]
    var_arity = md["var_arity"]
    V = md["V"]
    factor_names = md["factor_names"]
    F = md["F"]

    B = factor_eval_list[0].shape[0]

    # --- 1. Build global assignments over all variables ---
    # order: var_names[0], var_names[1], ...
    ranges = [range(int(a)) for a in var_arity]
    global_assignments = list(itertools.product(*ranges))  # list of tuples length V
    G = len(global_assignments)  # total number of global assignments

    # --- 2. Precompute local index mapping for each factor ---
    # For each factor fi, map local assignment (tuple of var states in its scope)
    # to the local index used in factor_eval_list[fi].
    factor_local_index = []
    for fi, fname in enumerate(factor_names):
        scope = factors[fname]  # e.g. ["v1", "v2"]
        arities = [variables[vname] for vname in scope]
        mapping = {}
        for local_idx, local_assign in enumerate(itertools.product(*[range(a) for a in arities])):
            mapping[tuple(local_assign)] = local_idx
        factor_local_index.append(mapping)

    # Map var_name -> index in var_names order
    var_index = {name: i for i, name in enumerate(var_names)}

    # --- 3. Compute unnormalized joint over all global assignments ---
    joint = torch.zeros(B, G, device=factor_eval_list[0].device,
                        dtype=factor_eval_list[0].dtype)

    for g_idx, g_assign in enumerate(global_assignments):
        # g_assign is a tuple of length V, e.g. (x_v1, x_v2, ..., x_vV)
        # Start with ones per batch element, then multiply factor contributions
        phi = torch.ones(B, device=factor_eval_list[0].device,
                         dtype=factor_eval_list[0].dtype)
        for fi, fname in enumerate(factor_names):
            scope = factors[fname]
            # Extract local assignment of scope variables from global assignment
            local_states = tuple(g_assign[var_index[vname]] for vname in scope)
            local_idx = factor_local_index[fi][local_states]
            phi = phi * factor_eval_list[fi][:, local_idx]
        joint[:, g_idx] = phi

    # --- 4. Normalize joint per batch ---
    Z = joint.sum(dim=1, keepdim=True) + eps
    joint = joint / Z  # [B, G]

    # --- 5. Compute exact marginals per variable ---
    exact_marginals = []
    for v in range(V):
        a = int(var_arity[v])
        marg_v = torch.zeros(B, a, device=joint.device, dtype=joint.dtype)
        for g_idx, g_assign in enumerate(global_assignments):
            state_v = g_assign[v]
            marg_v[:, state_v] += joint[:, g_idx]
        # Normalize for numerical safety
        marg_v = marg_v / (marg_v.sum(dim=-1, keepdim=True) + eps)
        exact_marginals.append(marg_v)

    return exact_marginals



# ------------------------------------------------------------------
# 5. Example usage
# ------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # CHAIN GRAPH EXAMPLE
    # variables = {"v1": 3, "v2": 4, "v3": 1}
    # factors = {
    #     "f1": ["v1", "v2"],  # 3 x 4 -> 12 assignments
    #     "f2": ["v2", "v3"],  # 4 x 1 -> 4 assignments
    # }

    # STAR GRAPH EXAMPLE
    # variables = {"v1": 3, "v2": 2, "v3": 2, "v4": 4, "v5": 2}
    # factors = {
    #     "f12": ["v1", "v2"],
    #     "f13": ["v1", "v3"],
    #     "f14": ["v1", "v4"],
    #     "f15": ["v1", "v5"],
    # }

    # LOOP GRAPH EXAMPLE
    # variables = {"v1": 3, "v2": 2, "v3": 4}
    # factors = {
    #     "f12": ["v1", "v2"],
    #     "f23": ["v2", "v3"],
    #     "f31": ["v3", "v1"],
    # }


    # FACTOR GRAPH WITH HIGHER-ORDER FACTORS (LOOPY)
    variables = {"v1": 2, "v2": 2, "v3": 3, "v4": 2}
    factors = {
        "f124": ["v1", "v2", "v4"],  # size 2×2×2 = 8
        "f243": ["v2", "v4", "v3"],  # size 2×2×3 = 12
    }

    md = build_graph_metadata(variables, factors)
    print("Variables:", md["var_names"])
    print("Factors:", md["factor_names"])
    print("Total edge-states:", md["total_edge_states"])
    print("Total assignments:", md["total_assignments"])

    B = 2

    # Create random factor evals **consistent with metadata**
    factor_eval_list = []
    for fi, fname in enumerate(md["factor_names"]):
        num_assign = int(md["factor_num_assign"][fi])
        print(f"Factor {fname}: num_assign = {num_assign}")
        f_eval = torch.rand(B, num_assign)
        factor_eval_list.append(f_eval)

    # Initialize factor->variable messages randomly and normalize per edge
    S = md["total_edge_states"]
    E = md["E"]
    messages_f2v = torch.rand(B, S)

    edge_id = md["edge_id_per_state"]  # [S]
    edge_id_b = edge_id.unsqueeze(0).expand(B, -1)  # [B, S]
    sum_per_edge = torch.zeros(B, E)
    sum_per_edge.scatter_add_(1, edge_id_b, messages_f2v)
    messages_f2v = messages_f2v / (sum_per_edge.gather(1, edge_id_b) + 1e-20)

    # Run BP
    evidence = {
        "v2": 1,  # for example: v2 is observed to be state index 1
        "v4": 0,
    }

    num_iters = 10
    for it in range(num_iters):
        messages_v2f = update_var_to_factor(messages_f2v, md)
        messages_v2f = clamp_messages_to_evidence(messages_v2f, evidence, md)  
        messages_f2v = update_factor_to_var(messages_v2f, factor_eval_list, md)
    # BP marginals
    bp_marginals = compute_var_marginals(messages_f2v, md)

    # Exact marginals
    exact_marginals = compute_exact_marginals_bruteforce(
        variables, factors, factor_eval_list, md
    )

    print("\nApproximate (BP) vs exact marginals after", num_iters, "iterations:")
    for i, (m_bp, m_ex) in enumerate(zip(bp_marginals, exact_marginals)):
        name = md["var_names"][i]
        print(f"\nVariable {name}:")
        print("  BP   :", m_bp)
        print("  Exact:", m_ex)
        print("  L1 diff per batch:", (m_bp - m_ex).abs().sum(dim=-1))