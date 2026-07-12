# swamp-gpu/kernels/attention.mojo
# Mojo 1.0.0b2 — FlashAttention-style sparse attention (CPU AVX-512)
# Tile-based online softmax + weighted sum. Zero CUDA.

from math import exp, sqrt, max, abs_f

# ---------------------------------------------------------------------------
# 4-bit KV block: per block of 32 values → d(f16)+dmin(f16)+16 nibbles = 20 bytes
# ---------------------------------------------------------------------------
comptime BLOCK_D: Int = 32
comptime BLOCK_BYTES: Int = 20  # d(2)+dmin(2)+nibbles(16)
comptime TILE_KV: Int = 64      # positions per tile (cache-friendly)
comptime HEAD_DIM: Int = 64

# ---------------------------------------------------------------------------
# Dequantize one 4-bit KV block and dot with Q
# ---------------------------------------------------------------------------
@always_inline
def dequant_dot(
    q: Pointer[Float32],       # [BLOCK_D] query
    k_q4: Pointer[Float32],   # {d,dmin,nibbles} (stored as f32 for simplicity)
    dim: Int,                   # actual dimension (<= BLOCK_D)
) -> Float32:
    var d = k_q4.load(0)
    var dmin = k_q4.load(1)
    var result: Float32 = 0.0
    for i in range(dim):
        var nib_ptr = k_q4.offset(2 + i // 2)
        var shift = (i % 2) * 4
        var nib = (nib_ptr[].to_int() >> shift) & 0x0F
        var k_val = d * Float32(nib) + dmin
        result += k_val * q.load(i)
    return result


# ---------------------------------------------------------------------------
# FlashAttention: online softmax + weighted sum for one head
# ---------------------------------------------------------------------------
@always_inline
def flash_attn_head(
    q_head: Pointer[Float32],       # [head_dim]
    k_q4: Pointer[Float32],         # [n_positions, head_dim_q4]
    v_q4: Pointer[Float32],
    positions: Pointer[Int32],       # [n_sparse]
    n_positions: Int,
    head_dim: Int,
    scale: Float32,
    out: Pointer[Float32],           # [head_dim]
):
    # Tile over sparse positions
    var m: Float32 = -1e10   # running max (online softmax)
    var d: Float32 = 0.0     # running denominator
    var blocks = (head_dim + BLOCK_D - 1) // BLOCK_D
    var blk_bytes = blocks * BLOCK_BYTES

    # Per-head output buffer
    for d_i in range(head_dim):
        out.store(d_i, 0.0)

    var t_start = 0
    while t_start < n_positions:
        var t_end = min(t_start + TILE_KV, n_positions)
        var n_tile = t_end - t_start

        # Score tile
        var max_score: Float32 = -1e10
        var scores = Pointer[Float32].alloc(n_tile)

        for pos_idx in range(n_tile):
            var t_pos = positions.load(t_start + pos_idx)
            var k_off = t_pos * blk_bytes
            var dot: Float32 = 0.0
            for blk in range(blocks):
                var kd = k_q4.load(k_off + blk * BLOCK_BYTES)
                var kdmin = k_q4.load(k_off + blk * BLOCK_BYTES + 1)
                var qq = q_head.offset(blk * BLOCK_D)
                var kv = k_q4.offset(k_off + blk * BLOCK_BYTES)
                for i in range(BLOCK_D):
                    if blk * BLOCK_D + i >= head_dim: break
                    var nib_ptr = kv.offset(2 + i // 2)
                    var shift = (i % 2) * 4
                    var nib = (nib_ptr[].to_int() >> shift) & 0x0F
                    var k_val = kd * Float32(nib) + kdmin
                    dot += k_val * qq.load(i)
            var sc = dot * scale
            scores.store(pos_idx, sc)
            if sc > max_score: max_score = sc

        # Online softmax merge
        var tile_d: Float32 = 0.0
        for pos_idx in range(n_tile):
            tile_d += exp(scores.load(pos_idx) - max_score)

        var new_m = max(m, max_score)
        d = d * exp(m - new_m) + tile_d * exp(max_score - new_m)
        m = new_m

        # Weighted sum of V for this tile
        for pos_idx in range(n_tile):
            var w = exp(scores.load(pos_idx) - m) / d
            if w < 1e-8: continue
            var t_pos = positions.load(t_start + pos_idx)
            var v_off = t_pos * blk_bytes
            for blk in range(blocks):
                var vd = v_q4.load(v_off + blk * BLOCK_BYTES)
                var vdmin = v_q4.load(v_off + blk * BLOCK_BYTES + 1)
                vv = v_q4.offset(v_off + blk * BLOCK_BYTES)
                for i in range(BLOCK_D):
                    if blk * BLOCK_D + i >= head_dim: break
                    var nib_ptr = vv.offset(2 + i // 2)
                    var shift = (i % 2) * 4
                    var nib = (nib_ptr[].to_int() >> shift) & 0x0F
                    var v_val = vd * Float32(nib) + vdmin
                    var out_i = blk * BLOCK_D + i
                    out.store(out_i, out.load(out_i) + v_val * w)

        scores.free()


# ---------------------------------------------------------------------------
# Host-callable entry point
# ---------------------------------------------------------------------------
@export
def sparse_attention_q4(
    q_ptr: Int,                 # raw pointer (passed as Int for FFI)
    k_q4_ptr: Int,
    v_q4_ptr: Int,
    positions_ptr: Int,
    out_ptr: Int,
    n_heads: Int,
    head_dim: Int,
    n_sparse: Int,
    window: Int,
    global_stride: Int,
):
    var q = Pointer[Float32](q_ptr)
    var k_q4 = Pointer[Float32](k_q4_ptr)
    var v_q4 = Pointer[Float32](v_q4_ptr)
    var positions = Pointer[Int32](positions_ptr)
    var output = Pointer[Float32](out_ptr)
    var scale = 1.0 / sqrt(Float32(head_dim))

    for h in range(n_heads):
        var q_off = h * head_dim
        var out_off = h * head_dim
        flash_attn_head(
            q.offset(q_off),
            k_q4, v_q4,
            positions, n_sparse,
            head_dim, scale,
            output.offset(out_off),
        )
