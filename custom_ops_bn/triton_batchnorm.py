"""
Triton kernels for batch normalization.

This module provides pure Triton kernel implementations for batch normalization.
The kernels use Welford's online algorithm for numerically stable mean and
variance computation.

For PyTorch integration, use the custom operators registered in ops.py.

Note: For production use, the CUDA implementation is recommended for better
performance and stability.
"""

import triton
import triton.language as tl


@triton.jit
def batchnorm_forward_training_kernel(
    # Pointers
    input_ptr,          # Input tensor pointer
    gamma_ptr,          # Scale parameter pointer
    beta_ptr,           # Shift parameter pointer
    output_ptr,         # Output tensor pointer
    mean_ptr,           # Saved mean pointer
    invstd_ptr,         # Saved inverse std pointer
    running_mean_ptr,   # Running mean pointer (updated in-place)
    running_var_ptr,    # Running variance pointer (updated in-place)
    # Dimensions
    N, C, spatial_dim,  # Batch size, channels, spatial dimension (H*W)
    # Parameters
    momentum,           # Momentum for running stats
    eps,                # Epsilon for numerical stability
    # Strides
    stride_n, stride_c, stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)

    c_off = c.to(tl.int64) * stride_c

    # Initialize Welford's algorithm variables
    mean = tl.zeros([1], dtype=tl.float32)
    m2 = tl.zeros([1], dtype=tl.float32)
    count = 0

    offs = tl.arange(0, BLOCK_SIZE)

    # Pass 1: Compute mean and variance using Welford's algorithm
    for n in range(0, N):
        base = n.to(tl.int64) * stride_n + c_off
        for start in range(0, spatial_dim, BLOCK_SIZE):
            s = start + offs
            mask = s < spatial_dim
            idx = base + s * stride_s
            x = tl.load(input_ptr + idx, mask=mask, other=0.0)
            count += tl.sum(mask.to(tl.int32), axis=0)
            delta = x - mean
            mean += tl.sum(delta * mask.to(tl.float32), axis=0) / count
            delta2 = x - mean
            m2 += tl.sum(delta * delta2 * mask.to(tl.float32), axis=0)

    var = m2 / count
    invstd = 1.0 / tl.sqrt(var + eps)

    # Save mean and invstd for backward pass
    tl.store(mean_ptr + c, mean)
    tl.store(invstd_ptr + c, invstd)

    # Update running statistics
    running_mean = tl.load(running_mean_ptr + c)
    running_var = tl.load(running_var_ptr + c)

    running_mean_new = momentum * running_mean + (1 - momentum) * mean
    running_var_new = momentum * running_var + (1 - momentum) * var
    
    tl.store(running_mean_ptr + c, running_mean_new)
    tl.store(running_var_ptr + c, running_var_new)

    # normalize and apply scale and shift
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    for n in range(0, N):
        base = n.to(tl.int64) * stride_n + c_off
        for start in range(0, spatial_dim, BLOCK_SIZE):
            s = start + offs
            mask = s < spatial_dim


@triton.jit
def batchnorm_forward_inference_kernel(
    # Pointers
    input_ptr,          # Input tensor pointer
    gamma_ptr,          # Scale parameter pointer
    beta_ptr,           # Shift parameter pointer
    mean_ptr,           # Running mean pointer
    var_ptr,            # Running variance pointer
    output_ptr,         # Output tensor pointer
    invstd_ptr,         # Saved inverse std pointer (for backward)
    # Dimensions
    N, C, spatial_dim,  # Batch size, channels, spatial dimension
    # Parameters
    eps,                # Epsilon for numerical stability
    # Strides
    stride_n, stride_c, stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)

    c_off = c.to(tl.int64) * stride_c

    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    invstd = 1.0 / tl.sqrt(var + eps)

    # Save invstd for backward pass
    tl.store(invstd_ptr + c, invstd)

    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    offs = tl.arange(0, BLOCK_SIZE)

    for n in range(0, N):
        base = n.to(tl.int64) * stride_n + c_off
        for start in range(0, spatial_dim, BLOCK_SIZE):
            s = start + offs
            mask = s < spatial_dim
            idx = base + s * stride_s
            x = tl.load(input_ptr + idx, mask=mask, other=0.0)
            x_hat = (x - mean) * invstd
            y = gamma * x_hat + beta
            tl.store(output_ptr + idx, y, mask=mask)



@triton.jit
def batchnorm_backward_kernel(
    # Pointers
    grad_output_ptr,    # Gradient from next layer
    input_ptr,          # Original input
    gamma_ptr,          # Scale parameter
    mean_ptr,           # Saved mean from forward
    invstd_ptr,         # Saved inverse std from forward
    grad_input_ptr,     # Gradient w.r.t. input (output)
    grad_gamma_ptr,     # Gradient w.r.t. gamma (output)
    grad_beta_ptr,      # Gradient w.r.t. beta (output)
    # Dimensions
    N, C, spatial_dim,
    # Strides
    stride_n, stride_c, stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)

    c_off = c.to(tl.int64) * stride_c

    mean = tl.load(mean_ptr + c)
    invstd = tl.load(invstd_ptr + c)
    gamma = tl.load(gamma_ptr + c)
    M = (N * spatial_dim) * 1.0

    offs = tl.arange(0, BLOCK_SIZE)

    # Pass 1: sum(dy) and sum(dy * x_hat)
    acc_dy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_dy_xhat = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for n in range(0, N):
        base = n.to(tl.int64) * stride_n + c_off
        for start in range(0, spatial_dim, BLOCK_SIZE):
            s = start + offs
            mask = s < spatial_dim
            idx = base + s * stride_s
            dy = tl.load(grad_output_ptr + idx, mask=mask, other=0.0)
            x = tl.load(input_ptr + idx, mask=mask, other=0.0)
            x_hat = (x - mean) * invstd
            acc_dy += dy
            acc_dy_xhat += dy * x_hat

    sum_dy = tl.sum(acc_dy, axis=0)
    sum_dy_xhat = tl.sum(acc_dy_xhat, axis=0)

    tl.store(grad_beta_ptr + c, sum_dy)
    tl.store(grad_gamma_ptr + c, sum_dy_xhat)

    # Pass 2: grad_input
    scale = gamma * invstd / M

    for n in range(0, N):
        base = n.to(tl.int64) * stride_n + c_off
        for start in range(0, spatial_dim, BLOCK_SIZE):
            s = start + offs
            mask = s < spatial_dim
            idx = base + s * stride_s
            dy = tl.load(grad_output_ptr + idx, mask=mask, other=0.0)
            x = tl.load(input_ptr + idx, mask=mask, other=0.0)
            x_hat = (x - mean) * invstd
            grad_x = scale * (M * dy - sum_dy - x_hat * sum_dy_xhat)
            tl.store(grad_input_ptr + idx, grad_x, mask=mask)

