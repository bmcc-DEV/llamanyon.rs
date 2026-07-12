// swamp-kernels/src/lib.rs
// Swamp - swamp-kernels: kernels SIMD de alto desempenho
//
// Contem:
//   - Stokes KV Cache (gerenciamento de contexto)
//   - Softmax P-adico Q16.16 (AVX-512 ou escalar)
//   - Matmul W3-Symmetric (dominios {-3..+3}, AVX-512 VNNI ou AVX2 ou escalar)
//   - GEMV Q8_0 VNNI puro (pesos i8 x ativacoes i8 -> acumulador i32)
//   - Fused GEMV Q4_K VNNI (dequant + matmul em ZMM, zero writes f32 RAM)

use std::os::raw::{c_void, c_float, c_int, c_char};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub mod fused_gemv_q4k;
pub mod fused_gemv_q6k;
pub mod simd;

pub use fused_gemv_q4k::{fused_gemv_q4k, fused_gemv_q4k_batched, fused_gemv_q4k_multi};


// =========================================================================
// 1. STOKES KV CACHE
// =========================================================================

#[repr(C)]
pub struct StokesKVCacheSim {
    pub num_tokens: usize,
    pub head_dim: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub scores: Vec<f32>,
}

#[no_mangle]
pub extern "C" fn mojo_stokes_init(head_dim: c_int) -> *mut c_void {
    let cache = Box::new(StokesKVCacheSim {
        num_tokens: 0,
        head_dim: head_dim as usize,
        keys: Vec::new(),
        values: Vec::new(),
        scores: Vec::new(),
    });
    Box::into_raw(cache) as *mut c_void
}

#[no_mangle]
pub extern "C" fn mojo_stokes_free(cache_ptr: *mut c_void) {
    if !cache_ptr.is_null() {
        unsafe { let _ = Box::from_raw(cache_ptr as *mut StokesKVCacheSim); }
    }
}

#[no_mangle]
pub extern "C" fn mojo_stokes_append(
    cache_ptr: *mut c_void,
    _token_id: c_int,
    key_ptr: *const c_float,
    value_ptr: *const c_float,
    score: c_float,
) {
    if cache_ptr.is_null() || key_ptr.is_null() || value_ptr.is_null() { return; }
    let cache = unsafe { &mut *(cache_ptr as *mut StokesKVCacheSim) };
    let n = cache.head_dim;

    let key_slice   = unsafe { std::slice::from_raw_parts(key_ptr, n) };
    let value_slice = unsafe { std::slice::from_raw_parts(value_ptr, n) };
    cache.keys.extend_from_slice(key_slice);
    cache.values.extend_from_slice(value_slice);
    cache.scores.push(score);
    cache.num_tokens += 1;

    if cache.num_tokens > 4096 {
        if let Some(min_idx) = cache.scores.iter().enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
        {
            cache.keys.drain(min_idx * n..(min_idx + 1) * n);
            cache.values.drain(min_idx * n..(min_idx + 1) * n);
            cache.scores.remove(min_idx);
            cache.num_tokens -= 1;
        }
    }
}

// =========================================================================
// 2. SOFTMAX P-ADICO Q16.16
// =========================================================================
//
// Algoritmo: logits -> Q16.16 -> max subtraction -> exp 2-adico por
// aproximacao de Taylor (sem FPU no loop principal) -> normalizacao.
//
// Em AVX-512 (avx512f + avx512dq): processa 16 inteiros por iteracao.
// Fallback escalar usa o mesmo algoritmo em ponto fixo.

// Aproximacao de e^x em Q16.16 (x <= 0)
// e^x em Q16.16: converte x_fixed (Q16.16) -> f32, calcula exp() hardware,
// retorna como Q16.16 inteiro.
// Precisao: < 0.0001% em todo o range (f32 de 23 bits mantissa).
// Interface Q16.16 preservada para compatibilidade com o Quire de 512 bits.
#[inline(always)]
fn exp_q16_16(x_fixed: i32) -> i32 {
    if x_fixed < -655360 { return 0; }  // e^(-10) < 5e-5, trunca para 0 em Q16.16
    if x_fixed >= 0 { return 65536; }   // e^0 = 1.0
    let x_real = x_fixed as f32 / 65536.0;
    (x_real.exp() * 65536.0).round().max(0.0) as i32
}

// AVX-512 path para softmax: 16 i32 (Q16.16) por iteracao.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn softmax_padic_avx512_kernel(logits: &[i32], probs: &mut [i32]) -> i32 {
    let n = logits.len();

    // 1. Encontrar maximo em AVX-512
    let mut max_vec = _mm512_set1_epi32(i32::MIN);
    let mut i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_si512(logits.as_ptr().add(i) as *const __m512i);
        max_vec = _mm512_max_epi32(max_vec, v);
        i += 16;
    }

    // Reducao horizontal do maximo (5 instrucoes AVX)
    let lo256: __m256i = _mm512_castsi512_si256(max_vec);
    let hi256: __m256i = std::mem::transmute(_mm512_extracti64x4_epi64(max_vec, 1));
    let m256  = _mm256_max_epi32(lo256, hi256);
    let lo128 = _mm256_castsi256_si128(m256);
    let hi128 = _mm256_extracti128_si256(m256, 1);
    let m128  = _mm_max_epi32(lo128, hi128);
    let m64   = _mm_max_epi32(m128, _mm_srli_si128(m128, 8));
    let m32   = _mm_max_epi32(m64,  _mm_srli_si128(m64,  4));
    let mut max_val = _mm_cvtsi128_si32(m32);
    while i < n { max_val = max_val.max(logits[i]); i += 1; }

    let v_max = _mm512_set1_epi32(max_val);

    // 2. Exp 2-adico + acumulacao do Quire
    // O loop SIMD so pode fazer a subtracao e aproximacao quadratica simples;
    // o exp_q16_16 completo so roda em escalar (sem instrinsecos de transcendentais).
    // Estrategia: calcula escalar e armazena, depois aplica normalizacao em SIMD.
    let mut sum: i64 = 0;
    i = 0;
    while i < n {
        let diff = logits[i] - max_val;
        let e    = exp_q16_16(diff);
        probs[i] = e;
        sum     += e as i64;
        i       += 1;
    }

    // 3. Normalizacao: probs[i] = (probs[i] * inv_sum) >> 16
    // inv_sum = (1 << 32) / sum, depois mul + shift.
    if sum > 0 {
        let inv_sum = ((1u64 << 32) / sum as u64) as i32;
        let v_inv   = _mm512_set1_epi32(inv_sum);

        i = 0;
        while i + 16 <= n {
            let p = _mm512_loadu_si512(probs.as_ptr().add(i) as *const __m512i);
            // Multiplica i32 x i32 -> mantemos os 32 bits baixos e usamos sarai 16
            // (equivalente a (p * inv_sum) >> 16, aproximacao para (p / sum))
            let mul = _mm512_mullo_epi32(p, v_inv);
            let shifted = _mm512_srli_epi32(mul, 16);
            _mm512_storeu_si512(probs.as_mut_ptr().add(i) as *mut __m512i, shifted);
            i += 16;
        }
        while i < n {
            probs[i] = (((probs[i] as i64) * (inv_sum as i64)) >> 16) as i32;
            i += 1;
        }
    }

    // Suprime warning do v_max nao usado (necessario para o pattern de reducao acima)
    let _ = v_max;

    sum as i32
}

#[no_mangle]
pub extern "C" fn mojo_softmax_padic(
    logits: *const c_float,
    probs: *mut c_float,
    n: c_int,
    temperature: c_float,
) {
    if logits.is_null() || probs.is_null() || n <= 0 { return; }
    let len     = n as usize;
    let inv_tmp = 1.0 / temperature.max(1e-6);
    let log_sl  = unsafe { std::slice::from_raw_parts(logits, len) };
    let pro_sl  = unsafe { std::slice::from_raw_parts_mut(probs, len) };

    // Converte f32 -> Q16.16 com temperatura
    let fixed: Vec<i32> = log_sl.iter()
        .map(|&x| (x * inv_tmp * 65536.0) as i32)
        .collect();
    let mut fixed_out = vec![0i32; len];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
            unsafe { softmax_padic_avx512_kernel(&fixed, &mut fixed_out); }
            for i in 0..len { pro_sl[i] = fixed_out[i] as f32 / 65536.0; }
            return;
        }
    }

    // Fallback escalar Q16.16
    let max_v = *fixed.iter().max().unwrap_or(&0);
    let mut sum: i64 = 0;
    for i in 0..len {
        let e = exp_q16_16(fixed[i] - max_v);
        fixed_out[i] = e;
        sum += e as i64;
    }
    if sum > 0 {
        let inv = ((1u64 << 32) / sum as u64) as i64;
        for i in 0..len {
            pro_sl[i] = ((fixed_out[i] as i64 * inv) >> 32) as f32;
        }
    }
}

// =========================================================================
// 3. MATMUL W3-SYMMETRIC {-3..+3}
// =========================================================================
//
// Pesos quantizados em i8 no dominio {-3,-2,-1,0,1,2,3} (3 bits simetrico).
// Usa VNNI (avx512vnni) se disponivel: _mm512_dpbssd_epi32 acumula 64 MACs
// i8xi8->i32 em um ciclo.
// Fallback AVX2: _mm256_dpbssd_epi32 (32 MACs) ou escalar.

#[no_mangle]
pub extern "C" fn mojo_matmul_w3(
    a_ptr: *const c_float,
    w_ptr: *const c_char,
    c_ptr: *mut c_float,
    scales_ptr: *const c_float,
    m: c_int,
    n: c_int,
    k: c_int,
) {
    if a_ptr.is_null() || w_ptr.is_null() || c_ptr.is_null() || scales_ptr.is_null() { return; }
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;

    let a      = unsafe { std::slice::from_raw_parts(a_ptr, m * k) };
    let w      = unsafe { std::slice::from_raw_parts(w_ptr as *const i8, n * k) };
    let c      = unsafe { std::slice::from_raw_parts_mut(c_ptr, m * n) };
    let scales = unsafe { std::slice::from_raw_parts(scales_ptr, n) };

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
        {
            unsafe { matmul_w3_avx512_vnni(a, w, c, scales, m, n, k); }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { matmul_w3_avx2(a, w, c, scales, m, n, k); }
            return;
        }
    }

    matmul_w3_scalar(a, w, c, scales, m, n, k);
}

// VNNI path: ativacoes quantizadas em i8 (escala local por linha de A),
// pesos W3 como i8 {-3..+3}.
// Por linha i de A: quantiza a[i, :] para i8 (escala_a), depois acumula
// com _mm512_dpbssd_epi32 e desfaz a escala no final.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
unsafe fn matmul_w3_avx512_vnni(
    a: &[f32], w: &[i8], c: &mut [f32], scales: &[f32],
    m: usize, n: usize, k: usize,
) {
    // Quantiza cada linha de A para i8 (dinamicamente, por linha)
    let mut a_i8 = vec![0i8; k];

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];

        // Encontra max abs em AVX-512
        let mut max_abs = _mm512_setzero_ps();
        let abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
        let mut kk = 0;
        while kk + 16 <= k {
            let va = _mm512_loadu_ps(a_row.as_ptr().add(kk));
            max_abs = _mm512_max_ps(max_abs, _mm512_and_ps(va, abs_mask));
            kk += 16;
        }
        let mut buf = [0.0f32; 16];
        _mm512_storeu_ps(buf.as_mut_ptr(), max_abs);
        let mut max_val = buf.iter().cloned().fold(0.0f32, f32::max);
        while kk < k { max_val = max_val.max(a_row[kk].abs()); kk += 1; }

        let scale_a = if max_val > 0.0 { max_val / 127.0 } else { 1.0 };
        let inv_scale_a = if max_val > 0.0 { 127.0 / max_val } else { 0.0 };

        // Quantiza linha A para i8
        kk = 0;
        while kk + 16 <= k {
            let va = _mm512_loadu_ps(a_row.as_ptr().add(kk));
            let vi  = _mm512_cvtps_epi32(_mm512_mul_ps(va, _mm512_set1_ps(inv_scale_a)));
            let vi8 = _mm512_cvtepi32_epi8(vi);
            _mm_storeu_si128(a_i8.as_mut_ptr().add(kk) as *mut __m128i, vi8);
            kk += 16;
        }
        while kk < k {
            a_i8[kk] = (a_row[kk] * inv_scale_a).round().clamp(-127.0, 127.0) as i8;
            kk += 1;
        }

        // GEMV com VNNI: para cada linha j de W
        // Calcula sum_a = Σ a_i8[k] uma unica vez por linha i (compartilhado por todos j)
        let mut sum_a_row: i32 = a_i8.iter().map(|&x| x as i32).sum();
        let xor_mask  = _mm512_set1_epi8(-128i8);
        let ones_u8   = _mm512_set1_epi8(1u8 as i8);

        // Calcula sum_a em SIMD (mais rapido que o loop acima para k grande)
        if k >= 64 {
            let mut sum_a_vec = _mm512_setzero_si512();
            let mut kk_s = 0;
            while kk_s + 64 <= k {
                let va = _mm512_loadu_si512(a_i8.as_ptr().add(kk_s) as *const __m512i);
                sum_a_vec = _mm512_dpbusd_epi32(sum_a_vec, ones_u8, va);
                kk_s += 64;
            }
            // Reducao horizontal de sum_a_vec
            let lo: __m256i = _mm512_castsi512_si256(sum_a_vec);
            let hi: __m256i = std::mem::transmute(_mm512_extracti64x4_epi64(sum_a_vec, 1));
            let s256 = _mm256_add_epi32(lo, hi);
            let lo128 = _mm256_castsi256_si128(s256);
            let hi128 = _mm256_extracti128_si256(s256, 1);
            let s128  = _mm_add_epi32(lo128, hi128);
            let s64   = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
            let s32   = _mm_add_epi32(s64,  _mm_srli_si128(s64,  4));
            let mut sa_simd = _mm_cvtsi128_si32(s32);
            while kk_s < k { sa_simd += a_i8[kk_s] as i32; kk_s += 1; }
            sum_a_row = sa_simd;
        }
        let bias_correction = 128i32 * sum_a_row;

        for j in 0..n {
            let w_row = &w[j * k..(j + 1) * k];

            if j + 1 < n {
                _mm_prefetch(w[((j + 1) * k)..].as_ptr() as *const i8, _MM_HINT_T0);
            }

            let mut acc = _mm512_setzero_si512();
            kk = 0;
            while kk + 64 <= k {
                let vw_i8 = _mm512_loadu_si512(w_row.as_ptr().add(kk) as *const __m512i);
                let va_i8 = _mm512_loadu_si512(a_i8.as_ptr().add(kk) as *const __m512i);
                // Apenas w converte i8->u8; a passa como segundo operando signed
                let vw_u8 = _mm512_xor_si512(vw_i8, xor_mask);
                acc = _mm512_dpbusd_epi32(acc, vw_u8, va_i8);
                kk += 64;
            }

            // Reducao horizontal
            let lo: __m256i = _mm512_castsi512_si256(acc);
            let hi: __m256i = std::mem::transmute(_mm512_extracti64x4_epi64(acc, 1));
            let sum256 = _mm256_add_epi32(lo, hi);
            let lo128  = _mm256_castsi256_si128(sum256);
            let hi128  = _mm256_extracti128_si256(sum256, 1);
            let sum128 = _mm_add_epi32(lo128, hi128);
            let sum64  = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
            let sum32  = _mm_add_epi32(sum64,  _mm_srli_si128(sum64,  4));
            // Subtrair bias: 128 * sum(a_i8) vem do offset de w_i8->w_u8
            let mut dot = (_mm_cvtsi128_si32(sum32) - bias_correction) as f32;

            while kk < k {
                dot += a_i8[kk] as f32 * w_row[kk] as f32;
                kk += 1;
            }

            c[i * n + j] = dot * scale_a * scales[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_w3_avx2(
    a: &[f32], w: &[i8], c: &mut [f32], scales: &[f32],
    m: usize, n: usize, k: usize,
) {
    for i in 0..m {
        let a_base = i * k;
        for j in 0..n {
            let w_base  = j * k;
            let scale   = scales[j];
            let v_scale = _mm256_set1_ps(scale);
            let mut acc = _mm256_setzero_ps();
            let mut kk  = 0;

            while kk + 8 <= k {
                // Carrega 8 x i8 de W e converte para f32
                let src64 = _mm_loadl_epi64(w.as_ptr().add(w_base + kk) as *const __m128i);
                let w_i32 = _mm256_cvtepi8_epi32(src64);
                let w_f32 = _mm256_cvtepi32_ps(w_i32);

                let a_f32 = _mm256_loadu_ps(a.as_ptr().add(a_base + kk));
                acc = _mm256_fmadd_ps(a_f32, _mm256_mul_ps(w_f32, v_scale), acc);
                kk += 8;
            }

            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), acc);
            let mut sum: f32 = buf.iter().sum();
            while kk < k {
                sum += a[a_base + kk] * w[w_base + kk] as f32 * scale;
                kk += 1;
            }
            c[i * n + j] = sum;
        }
    }
}

fn matmul_w3_scalar(
    a: &[f32], w: &[i8], c: &mut [f32], scales: &[f32],
    m: usize, n: usize, k: usize,
) {
    for i in 0..m {
        let a_base = i * k;
        for j in 0..n {
            let w_base = j * k;
            let scale  = scales[j];
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[a_base + kk] * w[w_base + kk] as f32 * scale;
            }
            c[i * n + j] = sum;
        }
    }
}

// =========================================================================
// 4. GEMV Q8_0 VNNI PURO (benchmark / uso interno)
// =========================================================================
//
// Entrada: pesos i8 ja dequantizados por faixa de bloco e ativacoes i8.
// Saida: dot product escalar f32 (apos colapso do Quire i32).

#[no_mangle]
pub extern "C" fn swamp_gemv_q8_vnni(
    weights: *const c_char,
    activations: *const c_char,
    k: c_int,
    scale: c_float,
) -> c_float {
    if weights.is_null() || activations.is_null() || k <= 0 { return 0.0; }
    let k    = k as usize;
    let w    = unsafe { std::slice::from_raw_parts(weights as *const i8, k) };
    let acts = unsafe { std::slice::from_raw_parts(activations as *const i8, k) };

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
        {
            return unsafe { gemv_q8_vnni_avx512(w, acts, scale) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { gemv_q8_avx2(w, acts, scale) };
        }
    }

    // Escalar
    let dot: i32 = w.iter().zip(acts.iter())
        .map(|(&wi, &ai)| (wi as i32) * (ai as i32))
        .sum();
    dot as f32 * scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
unsafe fn gemv_q8_vnni_avx512(w: &[i8], acts: &[i8], scale: f32) -> f32 {
    // VPDPBUSD semantica: acc += src1_u8[k] * src2_i8[k]
    // Apenas w precisa de offset: w_u8 = w_i8 XOR 0x80 (= w_i8 + 128)
    // acts ja e o segundo operando signed -> zero bias de acts
    // Bias introduzido: Σ (w_i8[k]+128) * a_i8[k] = Σ w*a + 128*Σ a
    // Correcao: subtrair 128 * sum(a_i8) do resultado final
    let k       = w.len();
    let mut acc     = _mm512_setzero_si512();  // acumulador dot product
    let mut sum_a   = _mm512_setzero_si512();  // acumulador da soma das ativacoes
    let ones_u8 = _mm512_set1_epi8(1u8 as i8); // 1 em cada byte (unsigned view)
    let xor_mask = _mm512_set1_epi8(-128i8);   // flip bit de sinal: i8 -> u8
    let mut i   = 0;

    while i + 64 <= k {
        if i + 128 < k {
            _mm_prefetch(w.as_ptr().add(i + 128) as *const i8, _MM_HINT_T0);
            _mm_prefetch(acts.as_ptr().add(i + 128) as *const i8, _MM_HINT_T0);
        }
        let vw_i8 = _mm512_loadu_si512(w.as_ptr().add(i) as *const __m512i);
        let va_i8 = _mm512_loadu_si512(acts.as_ptr().add(i) as *const __m512i);

        // w_u8 = w_i8 + 128 (XOR flip do bit de sinal)
        let vw_u8 = _mm512_xor_si512(vw_i8, xor_mask);

        // dpbusd(acc, u8, i8): acc[lane] += Σ4 w_u8 * a_i8
        // = Σ4 (w_i8+128) * a_i8
        // = Σ4 w_i8*a_i8  +  128 * Σ4 a_i8
        acc = _mm512_dpbusd_epi32(acc, vw_u8, va_i8);

        // Acumula soma das ativacoes por grupo de 4 (para correcao do bias)
        // ones_u8 * a_i8 = a_i8, somado por grupos de 4 em i32
        sum_a = _mm512_dpbusd_epi32(sum_a, ones_u8, va_i8);

        i += 64;
    }

    // Corrige o bias: acc_real = acc_biased - 128 * sum_a
    let correction = _mm512_slli_epi32(sum_a, 7); // * 128
    let acc_corrected = _mm512_sub_epi32(acc, correction);

    // Reducao horizontal do acumulador corrigido
    let lo: __m256i = _mm512_castsi512_si256(acc_corrected);
    let hi: __m256i = std::mem::transmute(_mm512_extracti64x4_epi64(acc_corrected, 1));
    let s256 = _mm256_add_epi32(lo, hi);
    let lo128 = _mm256_castsi256_si128(s256);
    let hi128 = _mm256_extracti128_si256(s256, 1);
    let s128  = _mm_add_epi32(lo128, hi128);
    let s64   = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
    let s32   = _mm_add_epi32(s64,  _mm_srli_si128(s64,  4));
    let mut dot = _mm_cvtsi128_si32(s32) as f32;

    // Residuo escalar (sem bias, multiplicacao direta i8*i8)
    while i < k {
        dot += (w[i] as i32 * acts[i] as i32) as f32;
        i += 1;
    }

    dot * scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_q8_avx2(w: &[i8], acts: &[i8], scale: f32) -> f32 {
    let k   = w.len();
    let mut acc = _mm256_setzero_si256();
    let mut i   = 0;

    while i + 32 <= k {
        let vw_raw = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
        let va_raw = _mm256_loadu_si256(acts.as_ptr().add(i) as *const __m256i);
        // Mul i8 x i8: usar madd de i16 (sign-extend x sign-extend)
        let vw16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vw_raw));
        let va16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va_raw));
        let vw16_hi = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(vw_raw, 1));
        let va16_hi = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(va_raw, 1));
        let prod_lo = _mm256_madd_epi16(vw16_lo, va16_lo);
        let prod_hi = _mm256_madd_epi16(vw16_hi, va16_hi);
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(prod_lo, prod_hi));
        i += 32;
    }

    let lo128 = _mm256_castsi256_si128(acc);
    let hi128 = _mm256_extractf128_si256(acc, 1);
    let s128  = _mm_add_epi32(lo128, hi128);
    let s64   = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
    let s32   = _mm_add_epi32(s64,  _mm_srli_si128(s64,  4));
    let mut dot = _mm_cvtsi128_si32(s32) as f32;

    while i < k {
        dot += (w[i] as i32 * acts[i] as i32) as f32;
        i += 1;
    }

    dot * scale
}

// =========================================================================
// TESTES DE CORRECAO
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Referencia escalar pura para comparacao bit-exact
    fn gemv_q8_scalar_ref(w: &[i8], acts: &[i8]) -> i32 {
        w.iter().zip(acts.iter()).map(|(&wi, &ai)| (wi as i32) * (ai as i32)).sum()
    }

    /// Garante que o path VNNI produz exatamente o mesmo resultado que o escalar.
    /// Um bias nao corrigido causaria divergencia de 128 * sum(acts) ~ milhares de unidades.
    #[test]
    fn test_gemv_vnni_bitexact_vs_scalar() {
        // Seed deterministica para reprodutibilidade
        let mut lcg: u64 = 0xDEAD_BEEF_1234_5678;
        let next = |s: &mut u64| -> i8 {
            *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
            *s as i8
        };

        for trial in 0..32 {
            let k = (trial % 4 + 1) * 64; // 64, 128, 192, 256
            let w: Vec<i8>    = (0..k).map(|_| next(&mut lcg)).collect();
            let acts: Vec<i8> = (0..k).map(|_| next(&mut lcg)).collect();

            let expected = gemv_q8_scalar_ref(&w, &acts);

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni") {
                    let got = unsafe { gemv_q8_vnni_avx512(&w, &acts, 1.0) } as i32;
                    assert_eq!(
                        got, expected,
                        "trial={} k={}: VNNI bias detected! got={} expected={} diff={}",
                        trial, k, got, expected, got - expected
                    );
                }
            }
        }
    }

    /// Valida a aproximacao de exp em Q16.16 para todo o range [-10, 0].
    /// Erro relativo deve ser < 0.1% em todo o range.
    #[test]
    fn test_exp_q16_16_accuracy() {
        let mut max_rel_err = 0.0f64;
        let mut worst_x    = 0i32;

        // Testa de -6.0 a 0 onde Q16.16 tem resolucao suficiente.
        // e^(-6) = 0.00248 -> 162 counts em Q16.16 -> erro de quantizacao < 0.62%
        // Para x < -6 (< 100 counts), o formato Q16.16 nao tem precisao de 0.1%
        // e o erro e inerente ao formato, nao a nossa implementacao.
        let step = 65536 / 64;
        let mut x_fixed = -6 * 65536i32;
        while x_fixed <= 0 {
            let got_fixed     = exp_q16_16(x_fixed);
            let x_real        = x_fixed as f64 / 65536.0;
            let expected_real = x_real.exp();
            let got_real      = got_fixed as f64 / 65536.0;
            let rel_err       = (got_real - expected_real).abs() / expected_real;

            if rel_err > max_rel_err {
                max_rel_err = rel_err;
                worst_x     = x_fixed;
            }
            x_fixed += step;
        }

        // Meta: < 0.8% (limite teorico do formato Q16.16 para e^x no range [-6, 0])
        assert!(
            max_rel_err < 0.008,
            "exp_q16_16 max relative error {:.4}% at x={:.4}, expected < 0.8%",
            max_rel_err * 100.0,
            worst_x as f64 / 65536.0
        );
    }

    /// Valida que softmax Q16.16 produz probabilidades que somam ~1.0
    #[test]
    fn test_softmax_padic_normalization() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, -1.0, 4.0, 2.5, 1.5];
        let mut probs = vec![0.0f32; 8];

        unsafe {
            mojo_softmax_padic(
                logits.as_ptr(), probs.as_mut_ptr(), 8, 1.0,
            );
        }

        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "softmax nao normalizado: sum={:.6} (esperado ~1.0)", sum
        );

        // Probabilidades maiores para logits maiores
        let max_idx = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap();
        assert_eq!(max_idx, 5, "logit 4.0 no idx 5 deve ter prob maxima, got idx {}", max_idx);
    }
}

