import torch
import itertools

from statsmodels.tsa.vector_ar.util import varsim
from torch.distributions import RelaxedBernoulli, RelaxedOneHotCategorical

from torch_concepts.distributions import Delta
from torch_concepts.nn import BaseInference, ProbabilisticModel


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
        edge_id_per_state[edge_state_offset[e]:edge_state_offset[e] + a] = e

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
        fa2factor[start:start + n] = fi

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
# 1b. Evidence handling: build (var,state) log-mask in batch
# ------------------------------------------------------------------

def build_evidence_logmask(evidence, md):
    """
    evidence: [B, V] with -1 for unobserved,
              k in [0, arity_v-1] for observed.
    Returns:
        logmask_vs: [B, total_var_states] with 0 or -inf.
        0    -> allowed state
        -inf -> forbidden state
    """
    B, V = evidence.shape
    var_arity = md["var_arity"]           # [V]
    var_state_offset = md["var_state_offset"]  # [V]
    total_vs = md["total_var_states"]

    device = evidence.device
    dtype = torch.float32  # can be changed to match messages dtype

    # By default, everything is allowed: log(1) = 0
    logmask_vs = torch.zeros(B, total_vs, device=device, dtype=dtype)

    for v in range(V):
        a = int(var_arity[v])
        start = int(var_state_offset[v])
        ev_v = evidence[:, v]      # [B]

        # Indices where this variable is observed
        observed = ev_v >= 0
        if not observed.any():
            continue

        # For observed batch entries, forbid all states first
        logmask_vs[observed, start:start + a] = float("-inf")

        # Then re-enable the observed state
        obs_states = ev_v[observed].long()               # [B_obs]
        rows = torch.arange(B, device=device)[observed]  # [B_obs]
        logmask_vs[rows, start + obs_states] = 0.0

    return logmask_vs


# ------------------------------------------------------------------
# 2. Variable -> Factor messages (tensorized, no loops)
# ------------------------------------------------------------------

def update_var_to_factor(messages_f2v, md, evidence_logmask_vs=None, eps=1e-20):
    """
    messages_f2v: [B, total_edge_states]
        factor->variable messages, stored per (edge,state).
    evidence_logmask_vs: [B, total_var_states] or None
        0 for allowed (var,state), -inf for forbidden.

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

    # sum logs per (var,state) over neighboring factors
    log_sum_vs = torch.zeros(B, num_vs,
                             device=messages_f2v.device,
                             dtype=messages_f2v.dtype)
    log_sum_vs.scatter_add_(1, vs_id_b, log_m_f2v)

    # Apply evidence AFTER aggregation (avoid -inf - -inf)
    if evidence_logmask_vs is not None:
        # unary log-potentials on (var,state)
        log_sum_vs = log_sum_vs + evidence_logmask_vs

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
                               device=messages_f2v_num.device,
                               dtype=messages_f2v_num.dtype)
    sum_per_edge.scatter_add_(1, edge_id_b, messages_f2v_num)
    norm = sum_per_edge.gather(1, edge_id_b) + eps
    messages_f2v = messages_f2v_num / norm

    return messages_f2v


# ------------------------------------------------------------------
# 4. Variable marginals from factor->var messages (with evidence)
# ------------------------------------------------------------------

def compute_var_marginals(messages_f2v, md, evidence_logmask_vs=None, eps=1e-20):
    """
    Approximate variable marginals from final factor->variable messages.
    If evidence_logmask_vs is given, it is applied as unary log-potentials
    on (var,state) before normalization.
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

    # apply evidence as log-potentials on (var,state)
    if evidence_logmask_vs is not None:
        log_sum_vs = log_sum_vs + evidence_logmask_vs

    marginals = []
    for v in range(V):
        a = int(var_arity[v])
        start = int(var_state_offset[v])
        m_v = torch.exp(log_sum_vs[:, start:start + a])   # [B, a]
        m_v = m_v / (m_v.sum(dim=-1, keepdim=True) + eps)
        marginals.append(m_v)
    return marginals


# ------------------------------------------------------------------
# 5. Exact marginals (uncond OR conditional, via brute force)
# ------------------------------------------------------------------

def compute_exact_marginals_bruteforce(
    variables,
    factors,
    factor_eval_list,
    md,
    evidence=None,
    eps=1e-20,
):
    """
    Exact marginals by enumerating all assignments of all variables.

    variables: dict {var_name: arity}
    factors:   dict {factor_name: [var_name1, ...]}    (same order as factor_eval_list)
    factor_eval_list: list length F
        factor_eval_list[fi]: [B, num_assign_fi], in SAME assignment ordering
        as build_graph_metadata (lexicographic over factor scope).
    md: metadata from build_graph_metadata
    evidence: None or [B, V] Long tensor
        -1 -> unobserved; k in [0, arity_v-1] -> observed.
        If given, returns p(X | evidence); otherwise p(X).

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

    device = factor_eval_list[0].device
    dtype = factor_eval_list[0].dtype

    # --- 1. Build global assignments over all variables ---
    ranges = [range(int(a)) for a in var_arity]
    global_assignments = list(itertools.product(*ranges))  # list of tuples length V
    G = len(global_assignments)  # total number of global assignments

    # Tensor form: [G, V]
    global_assign_tensor = torch.tensor(global_assignments, device=device, dtype=torch.long)

    # --- 2. Precompute local index mapping for each factor ---
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
    joint = torch.zeros(B, G, device=device, dtype=dtype)

    for g_idx, g_assign in enumerate(global_assignments):
        # g_assign is a tuple of length V, e.g. (x_v1, x_v2, ..., x_vV)
        # Start with ones per batch element, then multiply factor contributions
        phi = torch.ones(B, device=device, dtype=dtype)
        for fi, fname in enumerate(factor_names):
            scope = factors[fname]
            # Extract local assignment of scope variables from global assignment
            local_states = tuple(g_assign[var_index[vname]] for vname in scope)
            local_idx = factor_local_index[fi][local_states]
            phi = phi * factor_eval_list[fi][:, local_idx]
        joint[:, g_idx] = phi

    # --- 3b. Apply evidence if given: zero out inconsistent assignments ---
    if evidence is not None:
        evidence = evidence.to(device=device)
        # Shape to [B, G, V]
        ev_exp = evidence.unsqueeze(1).expand(B, G, V)              # [B, G, V]
        ga_exp = global_assign_tensor.unsqueeze(0).expand(B, G, V)  # [B, G, V]

        # Valid if: for all v, evidence[b,v] == -1 or equals assignment
        cond_ok = ((ev_exp < 0) | (ev_exp == ga_exp)).all(dim=-1)  # [B, G] bool
        mask = cond_ok.to(dtype)
        joint = joint * mask

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




class BPInference(BaseInference):

    def __init__(self, model, iters = 5):
        super().__init__()
        self.model : ProbabilisticModel = model
        self.iters = iters


        variables = {}
        factors = {}
        for var in self.model.variables:
            if var.distribution is RelaxedBernoulli:
                variables[var.concepts[0]] = 2
            elif var.distribution is RelaxedOneHotCategorical:
                variables[var.concepts[0]] = var.size
            elif var.distribution is Delta:
                variables[var.concepts[0]] = 1
            else:
                raise NotImplementedError("Distribution for variable unknown.")
            factors[var.concepts[0]] = [var.concepts[0]] + [c.concepts[0] for c in var.parents] #TODO: check this ordering is correct

        self.metadata = build_graph_metadata(variables, factors)
        self.assignments_factors = self.build_assignments(self.metadata, variables, factors)


    def build_assignments(self, md, variables, factors):
        """
        Build factor evaluations by calling your factor functions.

        variables: dict {var_name: arity}
        factors:   dict {factor_name: [var_name1, var_name2, ...]}  (ordered scope)
        md: metadata from build_graph_metadata
        Returns:
            factor_eval_list: list length F
                factor_eval_list[fi]: [B, num_assign_fi], in SAME assignment ordering
                as build_graph_metadata (lexicographic over factor scope).
        """
        assignments_factors = {}

        for fname in md["factor_names"]:

            vars_in_factor = factors[fname]  # e.g. ["v1", "v2", "v4"]
            arities = [variables[v] for v in vars_in_factor]  # e.g. [2, 2, 2]


            #We filter the variable representing the factor output
            arities = arities[1:]  # Exclude the first variable which is the target variable

            # --- 1. Enumerate all local assignments in the SAME order as build_graph_metadata ---
            # This is crucial so that factor_eval_list aligns with metadata.
            # Order is lexicographic over scope: product(range(a1), range(a2), ...)
            all_local_assign = list(itertools.product(*[range(a) for a in arities]))
            # shape: [num_assign, degree_of_factor]
            assign_tensor = torch.tensor(all_local_assign)
            assignments_factors[fname] = assign_tensor  # [num_assign, num_vars]
        return assignments_factors






    def query(self, query, evidence):

        # TODO assumption is that cpts are unary (they are parameterizing a single variable per time.
        # TODO we do not consider the optimization where multiple cpts with the same parents are batched together into a single factor)

        embeddings_dict = evidence

        batch_size = list(evidence.values())[0].shape[0]
        factor_eval_list = []

        assert all([v.concepts[0] in embeddings_dict.keys() for v in self.model.variables if v.distribution is Delta]), "All delta variables must have embeddings provided in evidence."

        for name_cpd, cpd in self.model.parametric_cpds.items(): # Iterate over factors. TODO: check that this is the right way to get factors
            input = []
            num_assignments = self.assignments_factors[name_cpd].shape[0]

            if cpd.variable.distribution is Delta:
                # Delta distribution: no need to evaluate the parameterization, just create a factor eval of ones
                factor_eval = torch.ones([batch_size,1], device=list(embeddings_dict.values())[0].device)
                factor_eval_list.append(factor_eval)
                continue
            else:
                for i, p in enumerate(cpd.variable.parents):

                    if p.distribution is Delta:
                        emb = embeddings_dict[p.concepts[0]] # [B, emb_dim]
                        #repeat for each assignment in the factor
                        emb_exp = emb.unsqueeze(1).expand(-1, num_assignments, -1)  # [B, num_assignments, emb_dim]
                        input.append(emb_exp)
                    elif p.distribution is RelaxedBernoulli:
                        assign = self.assignments_factors[name_cpd][:, i]
                        #repeat for batch size
                        assign = assign.unsqueeze(0).expand(batch_size, -1)  # [B, num_assignments]
                        assign = assign.unsqueeze(2)  # [B, num_assignments, 1]
                        input.append(assign)
                    elif p.distribution is RelaxedOneHotCategorical:
                        arity = p.size
                        one_hot = torch.nn.functional.one_hot(self.assignments_factors[name_cpd][:, i].long(), num_classes=arity).float()
                        one_hot = one_hot.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_assignments, arity]
                        input.append(one_hot)
                    else:
                        raise NotImplementedError("Unknown parent distribution in CPD2FactorWrapper.")



                input = torch.cat(input, dim=-1)

                #save shape
                input_shape = input.shape  # [B, num_assignments, input_dim]

                # turn into bidimentional tensor: [B * num_assignments, input_dim]
                input = input.view(batch_size * num_assignments, -1)
                evaluation = cpd.parametrization(input)

                # reshape back to [B, num_assignments, output_dim]
                evaluation = evaluation.view(batch_size, num_assignments, -1)

                # TODO: assumption is that embeddings are only input so now the output can be either a categorical (output size = arity) or a Bernoulli (output size = 1).
                # TODO: We need to turn them into factor evaluations. In each factor, the target variable of the CPD is the first variable in the scope so we can do a simple reshape
                # TODO: check that this is the case

                if cpd.variable.distribution is RelaxedOneHotCategorical:
                    #TODO: Check that it is concatenating the third dimension into the num_assignments dimension

                    # this is the tensorial equivalent to torch.cat([evaluation[:, :, i] for i in range(evaluation.shape[2])], dim=1)
                    factor_eval = evaluation.permute(0, 2, 1).reshape(batch_size, -1)

                elif cpd.variable.distribution is RelaxedBernoulli:
                    # Bernoulli output: need to create a factor eval of size 2
                    prob_1 = evaluation.view(batch_size, -1)
                    prob_0 = 1.0 - prob_1
                    factor_eval = torch.cat([prob_0, prob_1], dim=1)
                elif cpd.variable.distribution is Delta:
                    factor_eval = torch.ones([batch_size,1], device=evaluation.device)
                else:
                    raise NotImplementedError("Unknown CPD distribution in CPD2FactorWrapper.")

                factor_eval_list.append(factor_eval)

        B = batch_size
        S = self.metadata["total_edge_states"]
        E = self.metadata["E"]
        messages_f2v_init = torch.rand(B, S)

        edge_id = self.metadata["edge_id_per_state"]  # [S]
        edge_id_b = edge_id.unsqueeze(0).expand(B, -1)  # [B, S]
        sum_per_edge = torch.zeros(B, E)
        sum_per_edge.scatter_add_(1, edge_id_b, messages_f2v_init)
        messages_f2v_init = messages_f2v_init / (sum_per_edge.gather(1, edge_id_b) + 1e-20)

        messages_f2v_uncond = messages_f2v_init.clone()
        for it in range(self.iters):
            messages_v2f_uncond = update_var_to_factor(
                messages_f2v_uncond, self.metadata, evidence_logmask_vs=None
            )
            messages_f2v_uncond = update_factor_to_var(
                messages_v2f_uncond, factor_eval_list, self.metadata
            )
        bp_marginals_uncond = compute_var_marginals(
            messages_f2v_uncond, self.metadata, evidence_logmask_vs=None
        )

        return bp_marginals_uncond








if __name__ == "__main__":
    torch.manual_seed(0)

    # # FACTOR GRAPH WITH HIGHER-ORDER FACTORS (LOOPY)
    # variables = {"v1": 2, "v2": 2, "v3": 3, "v4": 2}
    # factors = {
    #     "f124": ["v1", "v2", "v4"],  # size 2×2×2 = 8
    #     "f243": ["v2", "v4", "v3"],  # size 2×2×3 = 12
    # }


    # STAR GRAPH EXAMPLE
    variables = {"v1": 3, "v2": 2, "v3": 3, "v4": 4, "v5": 2}
    factors = {
        "f12": ["v1", "v2"],
        "f13": ["v1", "v3"],
        "f14": ["v1", "v4"],
        "f15": ["v1", "v5"],
    }

    md = build_graph_metadata(variables, factors)
    print("Variables:", md["var_names"])
    print("Factors:", md["factor_names"])
    print("Total edge-states:", md["total_edge_states"])
    print("Total assignments:", md["total_assignments"])

    B = 2  # batch size

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
    messages_f2v_init = torch.rand(B, S)

    edge_id = md["edge_id_per_state"]  # [S]
    edge_id_b = edge_id.unsqueeze(0).expand(B, -1)  # [B, S]
    sum_per_edge = torch.zeros(B, E)
    sum_per_edge.scatter_add_(1, edge_id_b, messages_f2v_init)
    messages_f2v_init = messages_f2v_init / (sum_per_edge.gather(1, edge_id_b) + 1e-20)

    # ------------------------------------------------------------------
    # Evidence:
    #   -1 = unobserved
    #   otherwise the observed state index
    #
    # Example:
    #   batch 0: observe v1 = 1
    #   batch 1: observe v3 = 2
    # ------------------------------------------------------------------
    V = md["V"]
    evidence = torch.full((B, V), -1, dtype=torch.long)  # [B, V]
    # var_names order: ["v1", "v2", "v3", "v4"]
    evidence[0, 0] = 1  # batch 0: v1 = 1
    evidence[1, 2] = 2  # batch 1: v3 = 2

    evidence_logmask_vs = build_evidence_logmask(evidence, md)

    num_iters = 10

    # ------------------------
    # Unconditional BP
    # ------------------------
    messages_f2v_uncond = messages_f2v_init.clone()
    for it in range(num_iters):
        messages_v2f_uncond = update_var_to_factor(
            messages_f2v_uncond, md, evidence_logmask_vs=None
        )
        messages_f2v_uncond = update_factor_to_var(
            messages_v2f_uncond, factor_eval_list, md
        )
    bp_marginals_uncond = compute_var_marginals(
        messages_f2v_uncond, md, evidence_logmask_vs=None
    )

    # ------------------------
    # Conditional BP
    # ------------------------
    messages_f2v_cond = messages_f2v_init.clone()
    for it in range(num_iters):
        messages_v2f_cond = update_var_to_factor(
            messages_f2v_cond, md, evidence_logmask_vs=evidence_logmask_vs
        )
        messages_f2v_cond = update_factor_to_var(
            messages_v2f_cond, factor_eval_list, md
        )
    bp_marginals_cond = compute_var_marginals(
        messages_f2v_cond, md, evidence_logmask_vs=evidence_logmask_vs
    )

    # ------------------------
    # Exact marginals
    # ------------------------
    exact_marginals_uncond = compute_exact_marginals_bruteforce(
        variables, factors, factor_eval_list, md, evidence=None
    )
    exact_marginals_cond = compute_exact_marginals_bruteforce(
        variables, factors, factor_eval_list, md, evidence=evidence
    )

    # ------------------------
    # Print comparisons
    # ------------------------
    print("\n=== Unconditional: BP vs Exact ===")
    for i, (m_bp, m_ex) in enumerate(zip(bp_marginals_uncond, exact_marginals_uncond)):
        name = md["var_names"][i]
        print(f"\nVariable {name}:")
        print("  BP   (uncond):", m_bp)
        print("  Exact(uncond):", m_ex)
        print("  L1 diff per batch:", (m_bp - m_ex).abs().sum(dim=-1))

    print("\n=== Conditional on evidence: BP vs Exact ===")
    print("Evidence tensor (per batch, per var):", evidence)
    for i, (m_bp, m_ex) in enumerate(zip(bp_marginals_cond, exact_marginals_cond)):
        name = md["var_names"][i]
        print(f"\nVariable {name}:")
        print("  BP   (cond):", m_bp)
        print("  Exact(cond):", m_ex)
        print("  L1 diff per batch:", (m_bp - m_ex).abs().sum(dim=-1))
