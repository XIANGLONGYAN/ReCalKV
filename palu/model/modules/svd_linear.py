import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
from loguru import logger
from .quant import Quantizer
from .hadamard_utils import apply_hadamard
from sklearn.metrics.pairwise import cosine_similarity


def _per_head_whiten_decomposition_from_weight(weight, scaling_diag_matrix, rank):
    original_dtype = weight.dtype
    try:
        scaling_diag_matrix = scaling_diag_matrix.to(weight.device)
    except AttributeError:
        raise FileExistsError("Cache may not be loaded correctly")
    
    # Get the inverse of scaling_diag_matrix
    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

    # Multiply scaling_diag_matrix to weight matrix
    W_scale = torch.matmul(weight.to(torch.float32), scaling_diag_matrix.to(torch.float32))
    
    U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
    
    V = torch.matmul(Vt, scaling_matrix_inv)
    
    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    
    sqrtSigma = torch.sqrt(torch.diag(S))

    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, V).to(original_dtype)
    
    return L, R

def _per_head_whiten_decomposition_from_weight_nowhiten(weight, scaling_diag_matrix, rank):
    original_dtype = weight.dtype
    try:
        scaling_diag_matrix = scaling_diag_matrix.to(weight.device)
    except AttributeError:
        raise FileExistsError("Cache may not be loaded correctly")
    
    
    # Get the inverse of scaling_diag_matrix
    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

    # Multiply scaling_diag_matrix to weight matrix
    # W_scale = torch.matmul(weight.to(torch.float32), scaling_diag_matrix.to(torch.float32))
    W_scale = weight
    W_scale = W_scale.to(torch.float32)
    U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
    
    V = torch.matmul(Vt, scaling_matrix_inv)
    
    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    
    sqrtSigma = torch.sqrt(torch.diag(S))

    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, V).to(original_dtype)
    
    return L, R

def _per_head_whiten_update_decomposition_from_weight(weight, scaling_diag_matrix, rank, inps, outs, num_iter):
    original_dtype = weight.dtype
    try:
        scaling_diag_matrix = scaling_diag_matrix.to(weight.device)
    except AttributeError:
        raise FileExistsError("Cache may not be loaded correctly")
    
    # Get the inverse of scaling_diag_matrix
    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

    # Multiply scaling_diag_matrix to weight matrix
    W_scale = torch.matmul(weight.to(torch.float32), scaling_diag_matrix.to(torch.float32))
    
    U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
    
    V = torch.matmul(Vt, scaling_matrix_inv)
    
    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    
    sqrtSigma = torch.sqrt(torch.diag(S))

    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, V).to(original_dtype)
    
    
    # inps = torch.stack(inps)
    # outs = torch.stack(outs)
    
    inps = inps.view(inps.shape[0] * inps.shape[1], inps.shape[2])
    outs = outs.view(outs.shape[0] * outs.shape[1], outs.shape[2])
    
    for _ in range(num_iter):
        
        ori_output = torch.matmul(torch.matmul(inps.float(), (R.T).float()), (L.T).float())
        error = torch.sqrt(torch.sum((outs - ori_output)**2)).item() / torch.norm(outs, p='fro').item()
        logger.debug(f"[OVC] original error: {error}")

        L = torch.linalg.lstsq(torch.matmul(inps.float(), (R.T).float()),outs.float()).solution
        updated_output = torch.matmul(torch.matmul(inps.float(), (R.T).float()), L)
        error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        logger.debug(f"[OVC] updated error (L): {error}")

        R = ((weight.T).float() @ torch.linalg.pinv(L)).T
        updated_output = torch.matmul(torch.matmul(inps.float(), (R.T).float()), L)
        error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        logger.debug(f"[OVC] updated error (R): {error}")
        
        
    if num_iter == 0:
        return L, R
    else:
        return L.t(), R

def _per_head_decomposition_from_weight(weight, rank):
    original_dtype = weight.dtype
    # Get weight matrix decomposed
    U, S, Vt = torch.linalg.svd(weight.to(torch.float32), full_matrices=False)

    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]

    sqrtSigma = torch.sqrt(torch.diag(S))
    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, Vt).to(original_dtype)
    # assert torch.allclose(torch.matmul(L, R), weight, atol=1e-3), "SVD decomposition failed"
    return L, R

# permutation function
def apply_permutation_to_weight_matrix(w, permutation_info, head_num):

    total_rows = w.shape[0]

    m = total_rows // head_num
    
    perm = [permutation_info[i] for i in range(head_num)]
    
    w_permuted = torch.zeros_like(w)
    for i in range(head_num):
        start_row = i * m
        end_row = (i + 1) * m
        
        new_head_idx = perm[i]
        new_start_row = new_head_idx * m
        new_end_row = (new_head_idx + 1) * m
        
        w_permuted[new_start_row:new_end_row, :] = w[start_row:end_row, :]
    
    return w_permuted

def apply_permutation_to_x_matrix(w, permutation_info, head_num):
    
    batch_size, total_rows, features = w.shape
    m = features // head_num
    perm = [permutation_info[i] for i in range(head_num)]
    
    w_permuted = torch.zeros_like(w)
    
    for i in range(head_num):
        start_col = i * m
        end_col = (i + 1) * m
        
        new_head_idx = perm[i]
        new_start_col = new_head_idx * m
        new_end_col = (new_head_idx + 1) * m
        
        w_permuted[:, :, new_start_col:new_end_col] = w[:, :, start_col:end_col]
    
    return w_permuted

def apply_inverse_permutation_to_output(outputs_permutation, permutation_info, head_num):
    """
    Apply the inverse permutation to the output matrix `outputs_permutation` based on `permutation_info`.
    The permutation is applied to columns (each head corresponds to a set of columns).
    
    Args:
    - outputs_permutation (Tensor): Output matrix of shape (total_rows, features)
    - permutation_info (dict): Dictionary with the permutation, where key is the original head index
                               and value is the new head index.
    - head_num (int): The number of heads.
    
    Returns:
    - Tensor: Permuted output matrix.
    """
    batch_size, total_rows, features = outputs_permutation.shape
    m = features // head_num
    inverse_perm = {v: k for k, v in permutation_info.items()}
    
    outputs_restored = torch.zeros_like(outputs_permutation)
    
    for i in range(head_num):
        start_col = i * m
        end_col = (i + 1) * m
        
        new_head_idx = inverse_perm[i]
        new_start_col = new_head_idx * m
        new_end_col = (new_head_idx + 1) * m
        
        outputs_restored[:, :, new_start_col:new_end_col] = outputs_permutation[:, :, start_col:end_col]
    
    return outputs_restored

def generate_permutation_info():
    random.seed() 
    permutation = list(range(32))
    random.shuffle(permutation)
    
    permutation_info = {val: idx for idx, val in enumerate(permutation)}
    
    return permutation_info

def compute_cosine_similarity_matrix(tensor, num):
    matrices = [tensor[i * (tensor.shape[0] // num): (i + 1) * (tensor.shape[0] // num), :] for i in range(num)]

    similarity_matrix = np.zeros((num, num))

    for i in range(num):
        for j in range(i, num):
            tensor_i = matrices[i].T.cpu().numpy()
            tensor_j = matrices[j].T.cpu().numpy()
            
            similarity = cosine_similarity(tensor_i.T, tensor_j.T)[0][0]
            
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    return similarity_matrix

def compute_svd_error(matrix, rank):
    """
    SVD-decompose the matrix to the given rank and return the Frobenius reconstruction error.
    
    :param matrix: input matrix of shape (m, n)
    :param rank: rank to keep
    :return: error (Frobenius norm)
    """
    U, S, V = torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
    
    S_truncated = torch.diag(S[:rank])
    
    w_approx = torch.mm(U[:, :rank], torch.mm(S_truncated, V[:, :rank].T))
    
    error = torch.norm(matrix - w_approx, 'fro')
    
    return error

def compute_group_errors(w, ranks):
    """
    Compute the SVD error for each group.
    
    :param w: tensor of shape (num_groups, m, n)
    :param ranks: per-group ranks (length num_groups)
    :return: list of per-group errors
    """
    errors = []
    
    for i in range(len(ranks)):
        group_matrix = w[i]
        rank = ranks[i]
        error = compute_svd_error(group_matrix, rank)
        errors.append(error)
    
    return errors

def save_errors_to_file(errors, filename="errors.txt"):
    """
    Append the given error tensors to a file.
    Each tensor is written as one scalar per line.
    """
    with open(filename, "a") as file:
        for i, error_tensor in enumerate(errors):
            error_value = error_tensor.item()
            file.write(f"Layer {i+1} error: {error_value}\n")
    print("errors saved to file.")

def center_matrix(X):
    """
    Center the matrix by subtracting the mean of each column.
    """
    # If X is a PyTorch tensor, use torch.mean()
    return X - torch.mean(X, dim=0)

def compute_gram_matrix(X):
    """
    Compute the Gram matrix (kernel matrix) of a centered matrix.
    This uses the linear kernel (dot product).
    """
    return torch.mm(X, X.T)

def compute_hsic(KX, KY):
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC)
    between two Gram matrices.
    """
    n = KX.shape[0]
    H = torch.eye(n) - torch.ones((n, n)) / n  # Centering matrix
    H = H.to(KX.device)
    return torch.trace(torch.mm(torch.mm(KX, H), torch.mm(KY, H))) / (n - 1) ** 2

def compute_cka(X, Y):
    """
    Compute the CKA (Centered Kernel Alignment) between two matrices X and Y.
    """
    # Step 1: Center the matrices
    X_centered = center_matrix(X)
    Y_centered = center_matrix(Y)

    # Step 2: Compute the Gram matrices (kernel matrices)
    KX = compute_gram_matrix(X_centered)
    KY = compute_gram_matrix(Y_centered)

    
    # Step 3: Compute the HSIC between the two Gram matrices
    hsic = compute_hsic(KX, KY)
    
    # Step 4: Normalize the HSIC to get CKA
    hsic_x = compute_hsic(KX, KX)
    hsic_y = compute_hsic(KY, KY)
    
    cka = hsic.to(torch.float32) / torch.sqrt(hsic_x.to(torch.float32) * hsic_y.to(torch.float32))
    return cka

def compute_cka_matrix_from_flat(w, num_chunks=32, verbose=False):
    """
    Split w row-wise into num_chunks chunks and compute pairwise CKA similarity.
    
    Args:
        w (Tensor): 2D tensor whose row count is divisible by num_chunks
        num_chunks (int): number of chunks (default 32)
        verbose (bool): whether to log progress
    Returns:
        cka_matrix (Tensor): symmetric [num_chunks, num_chunks] CKA matrix
    """
    assert w.dim() == 2, "w must be a 2D tensor"
    total_rows = w.shape[0]
    assert total_rows % num_chunks == 0, f"row count {total_rows} must be divisible by num_chunks={num_chunks}"

    chunk_size = total_rows // num_chunks
    device = w.device
    cka_matrix = torch.zeros(num_chunks, num_chunks, device=device)

    chunks = w.split(chunk_size, dim=0)

    total_pairs = num_chunks * num_chunks
    pair_count = 0

    for i in range(num_chunks):
        for j in range(i, num_chunks):
            X = chunks[i].T.to(torch.float32)  # [chunk_size, dim] -> [dim, chunk_size]
            Y = chunks[j].T.to(torch.float32)

            cka_value = compute_cka(X, Y)
            cka_matrix[i, j] = cka_value
            cka_matrix[j, i] = cka_value

            pair_count += 1
            if verbose and pair_count % 10 == 0:
                logger.debug(f"[HSR] CKA progress: {pair_count}/{total_pairs} pairs completed.")

    return cka_matrix

def append_cka_to_pt(cka_matrix, file_path="all_cka.pt"):
    if os.path.exists(file_path):
        cka_list = torch.load(file_path)
    else:
        cka_list = []

    cka_list.append(cka_matrix.cpu())

    torch.save(cka_list, file_path)

def greedy_grouping(cka_matrix, num_groups=8, group_size=4):
    """Greedily assign heads into `num_groups` groups of `group_size` heads each,
    prioritizing head pairs with high mutual CKA similarity (HSR).

    :param cka_matrix: an (n x n) CKA similarity matrix over the n heads
    :param num_groups: number of groups (n == num_groups * group_size)
    :param group_size: number of heads per group
    :return: list of groups, each a list of original head indices
    """
    n = cka_matrix.shape[0]

    groups = [[] for _ in range(num_groups)]

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((cka_matrix[i, j], i, j))

    edges.sort(reverse=True, key=lambda x: x[0])

    head_to_group = [-1] * n

    for cka_value, head1, head2 in edges:
        if head_to_group[head1] == -1 and head_to_group[head2] == -1:
            for group_id in range(num_groups):
                if len(groups[group_id]) < group_size:
                    groups[group_id].append(head1)
                    groups[group_id].append(head2)
                    head_to_group[head1] = group_id
                    head_to_group[head2] = group_id
                    break
        elif head_to_group[head1] != -1 and head_to_group[head2] == -1:
            group_id = head_to_group[head1]
            if len(groups[group_id]) < group_size:
                groups[group_id].append(head2)
                head_to_group[head2] = group_id
        elif head_to_group[head2] != -1 and head_to_group[head1] == -1:
            group_id = head_to_group[head2]
            if len(groups[group_id]) < group_size:
                groups[group_id].append(head1)
                head_to_group[head1] = group_id

    remaining_heads = [i for i in range(n) if head_to_group[i] == -1]

    for group_id in range(num_groups):
        while len(groups[group_id]) < group_size and remaining_heads:
            head = remaining_heads.pop()
            groups[group_id].append(head)
            head_to_group[head] = group_id

    for group_id in range(num_groups):
        while len(groups[group_id]) > group_size:
            extra_head = groups[group_id].pop()
            for other_group_id in range(num_groups):
                if len(groups[other_group_id]) < group_size:
                    groups[other_group_id].append(extra_head)
                    head_to_group[extra_head] = other_group_id
                    break

    return groups

def generate_permutation(groups):
    """
    Generate a permutation mapping each head to its new position given the grouping.
    
    :param groups: list of groups; each holds original head indices
    :return: dict mapping original index -> new position
    """
    permutation = {}
    current_position = 0
    
    for group in groups:
        for head in group:
            permutation[head] = current_position
            current_position += 1
    
    return permutation

class HeadwiseLowRankModule(nn.Module):
    """ Headwise low rank module """

    def __init__(self, ranks, in_features, out_features, bias, permutation=False, permutation_info=None, head_num=-1):
        super().__init__()

        self.ranks = ranks
        self.num_groups = len(ranks)
        self.in_features = in_features
        self.out_features = out_features
        self.group_dim = out_features // self.num_groups
        self.permutation = permutation
        self.permutation_info = permutation_info
        self.head_num = head_num

        if (self.group_dim * self.num_groups) != self.out_features:
            raise ValueError(
                f"out_features must be divisible by num_groups (got `out_features`: {self.out_features}"
                f" and `num_groups`: {self.num_groups})."
            )

        self.VT = nn.Linear(in_features, sum(ranks), bias=False)
        
        Us = []
        for r in ranks:
            Us.append(nn.Linear(r, self.group_dim, bias=bias))

        self.U = nn.ModuleList(Us)    
        
        
        self.quantized_latents = False
        self.latent_quantizer = None
        
        self.inps = []
        self.outs = []
                
    def add_batch_update_u(self, inp, out):
        # if self.inps == None:
        #     self.inps = inp
        #     self.outs = out
        # else:
        #     self.inps = self.inps + inp
        #     self.outs = self.outs + out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        self.inps.append(inp.sum(dim=0))
        self.outs.append(out.sum(dim=0))
        del inp, out
        torch.cuda.empty_cache()
        
    def forward(self, 
                hidden_states: torch.Tensor):
        if self.permutation:
            low_rank_latents = self.project_to_latent(hidden_states)
            if self.quantized_latents:
                low_rank_latents = self.quantize_latent(low_rank_latents)
            outputs_permutation = self.reconstruct(low_rank_latents)
            outputs_repermutation = apply_inverse_permutation_to_output(outputs_permutation, self.permutation_info, self.head_num)
            outputs = outputs_repermutation

        else:
            low_rank_latents = self.project_to_latent(hidden_states)
            if self.quantized_latents:
                low_rank_latents = self.quantize_latent(low_rank_latents)
            outputs = self.reconstruct(low_rank_latents)
   
        return outputs
    
    def project_to_latent(self, hidden_states:  torch.Tensor):
        """
            hidden_states: Tensor of shape (batch_size, seq_len, in_features)
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                "Input tensor should have dimension 3."
            )
        hidden_states = self.VT(hidden_states)
        """
            hidden_states: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        """
        return hidden_states

    def reconstruct(self, low_rank_latents: torch.Tensor):
        """
            low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        """
        outputs = []
        total_ranks = 0
        for i in range(self.num_groups):
            low_rank_latent = low_rank_latents[:, :, total_ranks: total_ranks+self.ranks[i]]
            outputs.append(self.U[i](low_rank_latent))
            total_ranks += self.ranks[i]

        """
            outputs: Tensor of shape (batch_size, seq_len, out_features)
        """
        return torch.cat(outputs, dim=-1)
    
    def quantize_latent(self, low_rank_latents: torch.Tensor):
        """
            low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        """
        assert self.latent_quantizer is not None, "Latent quantizer is not initialized."
        fake_quantized_low_rank_latents = []
        total_ranks = 0
        for i in range(self.num_groups):
            low_rank_latent = low_rank_latents[:, :, total_ranks: total_ranks+self.ranks[i]]
            fake_quantized_low_rank_latents.append(self.latent_quantizer(low_rank_latent))
            total_ranks += self.ranks[i]

        """
            fake_quantized_low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ...)
        """
        return torch.cat(fake_quantized_low_rank_latents, dim=-1)
    
    # Configure quantization parameters and enable the quantized_latents flag
    def configure_latent_quantizer(self, 
        n_bits: int, 
        group_size: int, 
        sym: bool,
        clip_ratio: float,
        hadamard = False
    ):
        #self.latent_quantizer = Quantizer(n_bits, group_size, sym, clip_ratio, hadamard)
        self.latent_quantizer = Quantizer(n_bits, group_size, sym, clip_ratio)
        if hadamard:
            self.fused_hadamard_matrix()
            
        self.quantized_latents = True
    
    def fused_hadamard_matrix(self):
        total_ranks = 0
        for i in range(self.num_groups):
            # Apply Q to VT
            VT_weight_i = self.VT.weight.data[total_ranks: total_ranks+self.ranks[i], :]
            VT_weight_i = apply_hadamard(VT_weight_i.t())
            self.VT.weight.data[total_ranks: total_ranks+self.ranks[i], :] = VT_weight_i.t()
            # Apply Q^T to U
            
            U_weight_i = self.U[i].weight.data
            U_weight_i = apply_hadamard(U_weight_i)
            self.U[i].weight.data = U_weight_i
            
            total_ranks += self.ranks[i]
    
    # Convert an nn.Linear into a HeadwiseLowRankModule (differs only in whether whitening is used for SVD)
    @staticmethod
    def from_linear_whiten(
        old_module: nn.Linear,
        ranks: list,
    ):   
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)
            # l and r operate on the transposed weight: l reconstructs, r projects to the low-rank space

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module

    @staticmethod
    def from_linear_whiten_pre(
        old_module: nn.Linear,
        ranks: list,
    ):   
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data
        scaling_diag_matrix = old_module.scaling_diag_matrix
        scaling_diag_matrix = scaling_diag_matrix.to(w.device)


        w_scale = torch.matmul(w.to(torch.float32), scaling_diag_matrix.to(torch.float32))
        w_scale = w_scale.half()




        w = w_scale

        w = w.reshape(len(ranks), -1, old_module.in_features)
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_decomposition_from_weight_nowhiten(w[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)
            # l and r operate on the transposed weight: l reconstructs, r projects to the low-rank space

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module

    @staticmethod
    def from_linear_whiten_pre_loss_optimized(
        old_module: nn.Linear,
        ranks: list,
    ):   
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w_ori = old_module.weight.data
        w_ori = w_ori.reshape(len(ranks), -1, old_module.in_features)
        w = old_module.weight.data


        scaling_diag_matrix = old_module.scaling_diag_matrix
        scaling_diag_matrix = scaling_diag_matrix.to(w.device)

        w_scale = torch.matmul(w.to(torch.float32), scaling_diag_matrix.to(torch.float32))
        w_scale = w_scale.half()



        w = w_scale

        w = w.reshape(len(ranks), -1, old_module.in_features)


        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        errors = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_decomposition_from_weight_nowhiten(w[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)
            # l and r operate on the transposed weight: l reconstructs, r projects to the low-rank space
            error = torch.norm(w_ori[i] - l @ r, 'fro')
            errors.append(error)



        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module
    
    @staticmethod
    def from_linear_whiten_reorder(
        old_module: nn.Linear,
        ranks: list,
        head_num,
    ):   

        permutation_info = generate_permutation_info()

        w = old_module.weight.data



        scaling_diag_matrix = old_module.scaling_diag_matrix
        scaling_diag_matrix = scaling_diag_matrix.to(w.device)

        w_scale = torch.matmul(w.to(torch.float32), scaling_diag_matrix.to(torch.float32))
        w_scale = w_scale.half()

        # HSR grouping is derived from the model geometry, not hard-coded:
        #   head_num       = number of (key/value) heads in this projection
        #   num_groups     = len(ranks)  (one low-rank factor per group)
        #   group_size     = heads per group = head_num // num_groups
        num_groups = len(ranks)
        group_size = head_num // num_groups
        cka_matrix = compute_cka_matrix_from_flat(w_scale, head_num, verbose=False)

        group_info = greedy_grouping(cka_matrix, num_groups=num_groups, group_size=group_size)

        permutation_info = generate_permutation(group_info)

        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None, permutation=True, permutation_info=permutation_info, head_num=head_num)




        w_scale_permuted = apply_permutation_to_weight_matrix(w_scale, permutation_info, head_num)

        w_scale_permuted = w_scale_permuted.reshape(len(ranks), -1, old_module.in_features)
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_decomposition_from_weight_nowhiten(w_scale_permuted[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)
            # l and r operate on the transposed weight: l reconstructs, r projects to the low-rank space

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module

    @staticmethod
    def from_linear_adasvd(
        old_module: nn.Linear,
        ranks: list,
        num_iter: int,
        inps,
        outs,
    ):
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        inps = torch.stack(inps)
        outs = torch.stack(outs)
        outs_chunks = torch.chunk(outs, len(ranks), dim=-1)
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_update_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i], inps, outs_chunks[i], num_iter)
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)
            # l and r operate on the transposed weight: l reconstructs, r projects to the low-rank space

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            new_module.U[i].weight.data = new_module.U[i].weight.data.half()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        
        
        new_module.VT.weight.data = VT_weight.half()
        
        return new_module

    @staticmethod
    def from_linear_adasvd_with_permutation(
        old_module: nn.Linear,
        ranks: list,
        num_iter: int,
        inps,
        outs,
        permutation_info,
        head_num,
    ):
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None, permutation=True, permutation_info=permutation_info, head_num=head_num)
        w = old_module.weight.data

        w_permuted = apply_permutation_to_weight_matrix(w, permutation_info, head_num)

        w_permuted = w_permuted.reshape(len(ranks), -1, old_module.in_features)
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        inps = torch.stack(inps)
        outs = torch.stack(outs)
        outs_permuted = apply_permutation_to_x_matrix(outs, permutation_info, head_num)
        outs_chunks = torch.chunk(outs_permuted, len(ranks), dim=-1)

        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_update_decomposition_from_weight(w_permuted[i], old_module.scaling_diag_matrix, ranks[i], inps, outs_chunks[i], num_iter)
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)
            # l and r operate on the transposed weight: l reconstructs, r projects to the low-rank space

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            new_module.U[i].weight.data = new_module.U[i].weight.data.half()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        
        
        new_module.VT.weight.data = VT_weight.half()
        
        return new_module
    
    @staticmethod
    def from_linear(
        old_module: nn.Linear,
        ranks: list,
    ):
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        # Split w evenly into len(ranks) groups; heads in a group share one grouped SVD
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_decomposition_from_weight(w[i], ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]
        # load to VT
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module

    @staticmethod
    def from_linear_whiten_permutation(
        old_module: nn.Linear,
        ranks: list,
        permutation_info,
        head_num,
    ):   
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None, permutation=True, permutation_info=permutation_info, head_num=head_num)
        w = old_module.weight.data
        scaling_diag_matrix = old_module.scaling_diag_matrix
        scaling_diag_matrix = scaling_diag_matrix.to(w.device)
        w_scale = torch.matmul(w.to(torch.float32), scaling_diag_matrix.to(torch.float32))
        w_scale = w_scale.half()
        w = w_scale

        w_permuted = apply_permutation_to_weight_matrix(w, permutation_info, head_num)
        w_permuted = w_permuted.reshape(len(ranks), -1, old_module.in_features)

        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_decomposition_from_weight_nowhiten(w_permuted[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)
            # l and r operate on the transposed weight: l reconstructs, r projects to the low-rank space

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module
    
    @staticmethod
    def from_linear_permutation(
        old_module: nn.Linear,
        ranks: list,
        permutation_info,
        head_num,
    ):
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None, permutation=True, permutation_info=permutation_info, head_num=head_num)
        # Split w evenly into len(ranks) groups; heads in a group share one grouped SVD


        w = old_module.weight.data
        # scaling_diag_matrix = old_module.scaling_diag_matrix

        # scaling_diag_matrix = scaling_diag_matrix.to(w.device)

        # w = torch.matmul(w.to(torch.float32), scaling_diag_matrix.to(torch.float32))

        # w = w.half()

        w_permuted = apply_permutation_to_weight_matrix(w, permutation_info, head_num)

        w_permuted = w_permuted.reshape(len(ranks), -1, old_module.in_features)
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        wl = []
        wr = []
        for i in range(len(ranks)):
            # l, r = _per_head_decomposition_from_weight(w_permuted[i], ranks[i])
            l, r = _per_head_decomposition_from_weight(w_permuted[i], ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]
        # load to VT
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module