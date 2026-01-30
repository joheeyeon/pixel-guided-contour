"""
Pure PyTorch implementation of extreme_utils functions
This replaces the C++ extension with native PyTorch operations
"""
import torch


def roll_array(array, shift):
    """
    Roll array elements along the last dimension
    
    Args:
        array: input tensor
        shift: (N,) tensor of shift amounts for each element
        
    Returns:
        rolled tensor
    """
    if array.numel() == 0:
        return array
    
    rolled_array = torch.zeros_like(array)
    for i, s in enumerate(shift):
        rolled_array[i] = torch.roll(array[i], s.item(), dims=0)
    
    return rolled_array


def calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num):
    """
    Adjust edge numbers to ensure total equals p_num
    
    Args:
        edge_num: (B, N, V) number of points per edge
        edge_num_sum: (B, N) sum of edge numbers
        edge_idx_sort: (B, N, V) sorted indices by edge_num (descending)
        p_num: target total number of points
    """
    if len(edge_num.shape) == 2:
        # (B, V) -> (B, 1, V)
        batch_size = edge_num.shape[0]
        num_polys = 1
        edge_num = edge_num.unsqueeze(1)
        edge_num_sum = edge_num_sum.unsqueeze(1)
        edge_idx_sort = edge_idx_sort.unsqueeze(1)
    else:
        # (B, N, V)
        batch_size = edge_num.shape[0]
        num_polys = edge_num.shape[1]
    
    for b in range(batch_size):
        for n in range(num_polys):
            if edge_num.shape[1] == 0:
                continue
            current_sum = edge_num_sum[b, n].item() if num_polys > 1 else edge_num_sum[b].item()
            diff = int(p_num - current_sum)
            
            if diff == 0:
                continue
            
            # Adjust edge numbers
            if diff > 0:
                # Need to add points
                for i in range(diff):
                    idx_tensor = edge_idx_sort[b, n, i % edge_idx_sort.shape[-1]]
                    if idx_tensor.numel() == 1:
                        idx = idx_tensor.item()
                        edge_num[b, n, idx] += 1
            else:
                # Need to remove points (rare case)
                for i in range(-diff):
                    idx_tensor = edge_idx_sort[b, n, -(i % edge_idx_sort.shape[-1]) - 1]
                    if idx_tensor.numel() == 1:
                        idx = idx_tensor.item()
                        if edge_num[b, n, idx] > 1:
                            edge_num[b, n, idx] -= 1


def calculate_wnp(edge_num, edge_start_idx, p_num):
    """
    Calculate weights and indices for linear interpolation
    
    Args:
        edge_num: (B, N, V) or (B, V) number of points per edge
        edge_start_idx: (B, N, V) or (B, V) starting index for each edge
        p_num: target number of points
        
    Returns:
        weight: (B, N, p_num, 1) or (B, p_num, 1) interpolation weights
        ind: (B, N, p_num, 2) or (B, p_num, 2) vertex indices for interpolation
    """
    device = edge_num.device
    dtype = edge_num.dtype
    
    if len(edge_num.shape) == 2:
        # (B, V) shape
        batch_size, num_vertices = edge_num.shape
        num_polys = None
        edge_num = edge_num.unsqueeze(1)  # (B, 1, V)
        edge_start_idx = edge_start_idx.unsqueeze(1)  # (B, 1, V)
    else:
        # (B, N, V) shape
        batch_size, num_polys, num_vertices = edge_num.shape
    
    # Initialize output tensors
    weight = torch.zeros(batch_size, 1 if num_polys is None else num_polys, p_num, 1, device=device)
    ind = torch.zeros(batch_size, 1 if num_polys is None else num_polys, p_num, 2, device=device, dtype=torch.long)
    
    for b in range(batch_size):
        for n in range(1 if num_polys is None else num_polys):
            for v in range(num_vertices):
                num_pts = edge_num[b, n, v].item()
                start = edge_start_idx[b, n, v].item()
                
                if num_pts == 0:
                    continue
                    
                v_next = (v + 1) % num_vertices
                
                for i in range(int(num_pts)):
                    idx = int(start + i)
                    if idx >= p_num:
                        continue
                        
                    if num_pts == 1:
                        # Single point at midpoint
                        weight[b, n, idx, 0] = 0.5
                    else:
                        # Linear interpolation weight
                        weight[b, n, idx, 0] = i / (num_pts - 1.0) if num_pts > 1 else 0.0
                    
                    ind[b, n, idx, 0] = v
                    ind[b, n, idx, 1] = v_next
    
    if num_polys is None:
        weight = weight.squeeze(1)
        ind = ind.squeeze(1)
    
    return weight, ind


def uniform_upsample_pure(poly, p_num):
    """
    Pure PyTorch implementation of uniform_upsample
    
    Args:
        poly: (B, N, V, 2) or (B, V, 2) polygon vertices
        p_num: target number of points
        
    Returns:
        upsampled polygon with p_num vertices
    """
    # Handle different input shapes
    if len(poly.shape) == 3:
        # (B, V, 2) -> (B, N=1, V, 2)
        poly = poly.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, num_polys, num_verts, _ = poly.shape
    
    # Calculate edge lengths
    next_poly = torch.roll(poly, -1, dims=2)
    edge_len = (next_poly - poly).pow(2).sum(-1).sqrt()
    
    # Calculate number of points per edge
    edge_len_sum = torch.sum(edge_len, dim=-1, keepdim=True)
    edge_num = torch.round(edge_len * p_num / edge_len_sum).long()
    edge_num = torch.clamp(edge_num, min=1)
    
    # Sort edges by number of points (for adjustment)
    edge_idx_sort = torch.argsort(edge_num, dim=-1, descending=True)
    
    # Adjust edge numbers to sum to p_num
    edge_num_sum = torch.sum(edge_num, dim=-1)
    if edge_num.numel() > 0:
        calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
    
    # Calculate starting indices
    edge_start_idx = torch.cumsum(edge_num, dim=-1) - edge_num
    
    # Calculate weights and indices for interpolation
    weight, ind = calculate_wnp(edge_num, edge_start_idx, p_num)
    
    # Gather vertices for interpolation
    ind1 = ind[..., 0].unsqueeze(-1).expand(-1, -1, -1, 2)  # (B, N, p_num, 2)
    ind2 = ind[..., 1].unsqueeze(-1).expand(-1, -1, -1, 2)  # (B, N, p_num, 2)
    poly1 = poly.gather(2, ind1)
    poly2 = poly.gather(2, ind2)
    
    # Linear interpolation
    poly_upsampled = poly1 * (1 - weight) + poly2 * weight
    
    if squeeze_output:
        poly_upsampled = poly_upsampled.squeeze(1)
    
    return poly_upsampled