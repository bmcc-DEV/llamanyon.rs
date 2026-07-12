// swamp-engine/src/cache.rs
// TieredPagedKVCache: KV cache com hierarquia de 3 niveis
//   Hot  (RAM, Q4) — formato primario, ~9 KB/token para 1B model
//   FP32 (RAM)     — cache de dequantizacao, alocado sob demanda para full attention
//   Cold (NVMe)    — pagina serializada em Q4
//
// Block-size adaptativo por zona de distancia:
//   Zona 0 (pos 0..4096):     block_size=32  (granularidade fina)
//   Zona 1 (pos 4096..131072): block_size=128
//   Zona 2 (pos 131072+):      block_size=1024 (compressao 32x)

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::{File, OpenOptions};
use std::io::{Write, Read, Seek, SeekFrom};
use std::path::Path;

const DEFAULT_BLOCK_SIZE: usize = 32;
const DEFAULT_COLD_PAGES: usize = 65536;

const ZONE_STARTS: [usize; 3] = [0, 4096, 131072];
const ZONE_BLOCK_SIZES: [usize; 3] = [32, 128, 1024];

#[derive(Clone, Copy, PartialEq, Eq)]
enum PageState {
    Hot,
    Cold(u64),
}

struct CacheZone {
    block_size: usize,
    zone_start: usize,
    // Per-page arrays (indexed by local page id within this zone)
    k_pages: Vec<*mut f32>,
    v_pages: Vec<*mut f32>,
    k_q4_pages: Vec<*mut u8>,
    v_q4_pages: Vec<*mut u8>,
    page_capacities: Vec<(Layout, usize)>,
    page_state: Vec<PageState>,
    lru_prev: Vec<Option<usize>>,
    lru_next: Vec<Option<usize>>,
    lru_head: Option<usize>,
    lru_tail: Option<usize>,
    hot_count: usize,
    // Entropy-guided eviction: accumulated attention-importance per page.
    // Updated by attention_tracked() — higher = more likely to be kept hot.
    entropy_importance: Vec<f64>,
}

impl CacheZone {
    fn new(block_size: usize, zone_start: usize, num_pages: usize) -> Self {
        Self {
            block_size,
            zone_start,
            k_pages: Vec::with_capacity(num_pages),
            v_pages: Vec::with_capacity(num_pages),
            k_q4_pages: Vec::with_capacity(num_pages),
            v_q4_pages: Vec::with_capacity(num_pages),
            page_capacities: Vec::with_capacity(num_pages),
            page_state: Vec::with_capacity(num_pages),
            lru_prev: Vec::with_capacity(num_pages),
            lru_next: Vec::with_capacity(num_pages),
            lru_head: None,
            lru_tail: None,
            hot_count: 0,
            entropy_importance: Vec::with_capacity(num_pages),
        }
    }
}

pub struct PagedKVCache {
    zones: Vec<CacheZone>,
    // Flat page id -> zone mapping (pre-computed)
    zone_page_offsets: Vec<usize>, // flat page id of first page in each zone
    total_pages: usize,

    // Cold storage (NVMe) — circular pre-allocated file
    cold_file: Option<File>,
    cold_dir: String,
    cold_capacity: usize,
    cold_slot_used: Vec<bool>,
    cold_write_cursor: usize,

    // Config
    ram_page_limit: usize,
    num_layers: usize,
    num_kv_heads: usize,
    pub max_seq_len: usize,
    pub block_size: usize,
    head_dim: usize,
    current_pos: usize,

    // Cached per-zone page sizes (computed once for speed)
    f32_page_sizes: Vec<(Layout, usize)>,
    q4_page_sizes: Vec<usize>,
    q4_layouts: Vec<Layout>,
}

unsafe impl Send for PagedKVCache {}
unsafe impl Sync for PagedKVCache {}

impl PagedKVCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Self {
        Self::with_block_size(num_layers, num_kv_heads, max_seq_len, head_dim, DEFAULT_BLOCK_SIZE)
    }

    pub fn with_block_size(
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        _block_size: usize,
    ) -> Self {
        let zones = Self::build_zones(num_layers, num_kv_heads, max_seq_len, head_dim);
        let total_pages: usize = zones.iter().map(|z| z.k_pages.capacity()).sum();
        let zone_page_offsets = Self::compute_zone_offsets(&zones);

        let q4_page_size = num_layers * num_kv_heads * 32 * ((head_dim + 31) / 32) * 20;

        let ram_page_limit = std::env::var("SWAMP_CACHE_RAM_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| (mb * 1024 * 1024).max(q4_page_size) / q4_page_size)
            .unwrap_or(usize::MAX);

        let cold_dir = std::env::var("SWAMP_CACHE_COLD_DIR")
            .unwrap_or_else(|_| "/tmp/swamp_cache".to_string());

        let cold_capacity = std::env::var("SWAMP_CACHE_COLD_PAGES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| {
                if ram_page_limit == usize::MAX {
                    DEFAULT_COLD_PAGES
                } else {
                    total_pages.saturating_sub(ram_page_limit).min(DEFAULT_COLD_PAGES)
                }
            });

        let f32_page_sizes = (0..3).map(|zi| {
            let bs = ZONE_BLOCK_SIZES[zi];
            let count = num_layers * num_kv_heads * bs * head_dim;
            let byte_size = count * std::mem::size_of::<f32>();
            let layout = Layout::from_size_align(byte_size, 64).unwrap();
            (layout, byte_size)
        }).collect();

        let q4_page_sizes: Vec<usize> = (0..3).map(|zi| {
            let bs = ZONE_BLOCK_SIZES[zi];
            let blocks_per_head = (head_dim + 31) / 32;
            let bytes_per_head = blocks_per_head * 20;
            num_layers * num_kv_heads * bs * bytes_per_head
        }).collect();

        let q4_layouts = q4_page_sizes.iter().map(|&sz| {
            Layout::from_size_align(sz, 64).unwrap()
        }).collect();

        Self {
            zones,
            zone_page_offsets,
            total_pages,
            cold_file: None,
            cold_dir,
            cold_capacity,
            cold_slot_used: Vec::new(),
            cold_write_cursor: 0,
            ram_page_limit,
            num_layers,
            num_kv_heads,
            max_seq_len,
            block_size: ZONE_BLOCK_SIZES[0],
            head_dim,
            current_pos: 0,
            f32_page_sizes,
            q4_page_sizes,
            q4_layouts,
        }
    }

    fn build_zones(num_layers: usize, num_kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Vec<CacheZone> {
        let _ = num_layers;
        let _ = num_kv_heads;
        let _ = head_dim;
        let mut zones = Vec::with_capacity(3);
        for zi in 0..3 {
            let bs = ZONE_BLOCK_SIZES[zi];
            let start = ZONE_STARTS[zi];
            let end = if zi + 1 < 3 { ZONE_STARTS[zi + 1] } else { max_seq_len };
            let coverage = if end > start { end - start } else { 0 };
            let np = if coverage > 0 { (coverage + bs - 1) / bs } else { 0 };
            zones.push(CacheZone::new(bs, start, np));
        }
        zones
    }

    fn compute_zone_offsets(zones: &[CacheZone]) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(zones.len());
        let mut acc = 0;
        for z in zones {
            offsets.push(acc);
            acc += z.k_pages.capacity();
        }
        offsets
    }

    fn zone_index(&self, pos: usize) -> usize {
        if pos < ZONE_STARTS[1] { 0 }
        else if pos < ZONE_STARTS[2] { 1 }
        else { 2 }
    }

    fn local_page_id(&self, zi: usize, pos: usize) -> usize {
        let zone = &self.zones[zi];
        (pos - zone.zone_start) / zone.block_size
    }

    fn flat_page_id(&self, zi: usize, local: usize) -> usize {
        self.zone_page_offsets[zi] + local
    }

    fn resolve_flat(&self, flat_id: usize) -> (usize, usize) {
        for zi in (0..self.zones.len()).rev() {
            if flat_id >= self.zone_page_offsets[zi] {
                return (zi, flat_id - self.zone_page_offsets[zi]);
            }
        }
        panic!("flat page id {} out of range (total {})", flat_id, self.total_pages);
    }

    fn block_size_for_zone(&self, zi: usize) -> usize {
        self.zones[zi].block_size
    }

    pub fn block_size_for(&self, pos: usize) -> usize {
        ZONE_BLOCK_SIZES[self.zone_index(pos)]
    }

    fn f32_page_byte_size_for_zone(&self, zi: usize) -> (Layout, usize) {
        self.f32_page_sizes[zi]
    }

    fn q4_page_byte_size_for_zone(&self, zi: usize) -> usize {
        self.q4_page_sizes[zi]
    }

    fn q4_layout_for_zone(&self, zi: usize) -> Layout {
        self.q4_layouts[zi]
    }

    fn init_cold_file(&mut self) {
        if self.cold_file.is_some() { return; }
        if self.cold_capacity == 0 { return; }

        let path = Path::new(&self.cold_dir);
        let _ = std::fs::create_dir_all(path);
        let cold_path = path.join(format!("kv_cache_{}.bin", std::process::id()));

        // Slot size is based on zone 2 (largest block_size = 1024)
        // This ensures any zone's page fits in a cold slot
        let max_q4 = self.q4_page_sizes.iter().max().copied().unwrap_or(0);
        let slot_size = max_q4 * 2;
        let file_size = self.cold_capacity * slot_size;

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&cold_path)
            .expect("Failed to create KV cold storage file");
        file.set_len(file_size as u64).expect("Failed to pre-allocate cold storage");

        self.cold_slot_used = vec![false; self.cold_capacity];
        self.cold_write_cursor = 0;
        self.cold_file = Some(file);
    }

    fn alloc_hot_page_in_zone(&mut self, zi: usize) -> usize {
        let q4_layout = self.q4_layout_for_zone(zi);
        let k_q4_ptr = unsafe { alloc_zeroed(q4_layout) } as *mut u8;
        let v_q4_ptr = unsafe { alloc_zeroed(q4_layout) } as *mut u8;
        let zone = &mut self.zones[zi];
        if k_q4_ptr.is_null() || v_q4_ptr.is_null() {
            panic!("PagedKVCache OOM at zone {} page {}", zi, zone.k_pages.len());
        }

        let local_id = zone.k_pages.len();
        zone.k_pages.push(std::ptr::null_mut());
        zone.v_pages.push(std::ptr::null_mut());
        zone.k_q4_pages.push(k_q4_ptr);
        zone.v_q4_pages.push(v_q4_ptr);
        zone.page_capacities.push((q4_layout, self.q4_page_sizes[zi]));
        zone.page_state.push(PageState::Hot);
        zone.lru_prev.push(None);
        zone.lru_next.push(None);
        zone.entropy_importance.push(0.0);
        zone.hot_count += 1;

        zone.lru_next[local_id] = zone.lru_head;
        if let Some(h) = zone.lru_head {
            zone.lru_prev[h] = Some(local_id);
        }
        zone.lru_head = Some(local_id);
        if zone.lru_tail.is_none() {
            zone.lru_tail = Some(local_id);
        }

        local_id
    }

    fn touch_lru_in_zone(&mut self, zi: usize, local_id: usize) {
        let zone = &mut self.zones[zi];
        if zone.lru_head == Some(local_id) { return; }
        let prev = zone.lru_prev[local_id];
        let next = zone.lru_next[local_id];
        if let Some(p) = prev { zone.lru_next[p] = next; }
        if let Some(n) = next { zone.lru_prev[n] = prev; }
        if zone.lru_tail == Some(local_id) { zone.lru_tail = prev; }
        zone.lru_prev[local_id] = None;
        zone.lru_next[local_id] = zone.lru_head;
        if let Some(h) = zone.lru_head { zone.lru_prev[h] = Some(local_id); }
        zone.lru_head = Some(local_id);
        if zone.lru_tail.is_none() { zone.lru_tail = Some(local_id); }
    }

    fn remove_from_lru_in_zone(&mut self, zi: usize, local_id: usize) {
        let zone = &mut self.zones[zi];
        let prev = zone.lru_prev[local_id];
        let next = zone.lru_next[local_id];
        if let Some(p) = prev { zone.lru_next[p] = next; }
        if let Some(n) = next { zone.lru_prev[n] = prev; }
        if zone.lru_head == Some(local_id) { zone.lru_head = next; }
        if zone.lru_tail == Some(local_id) { zone.lru_tail = prev; }
        zone.lru_prev[local_id] = None;
        zone.lru_next[local_id] = None;
    }

    fn ensure_f32_page_in_zone(&mut self, zi: usize, local_id: usize) {
        if !self.zones[zi].k_pages[local_id].is_null() { return; }

        let (layout, byte_size) = self.f32_page_byte_size_for_zone(zi);
        let k_ptr = unsafe { alloc_zeroed(layout) } as *mut f32;
        let v_ptr = unsafe { alloc_zeroed(layout) } as *mut f32;
        if k_ptr.is_null() || v_ptr.is_null() {
            panic!("PagedKVCache OOM on f32 page alloc zone {} page {}", zi, local_id);
        }

        let blocks_per_head = (self.head_dim + 31) / 32;
        let q4_blk_size = 20;
        let bs = self.zones[zi].block_size;
        let q4_bytes_per_slot = blocks_per_head * q4_blk_size;

        // Extract raw pointers before the loop to avoid repeated borrow
        let k_q4 = self.zones[zi].k_q4_pages[local_id];
        let v_q4 = self.zones[zi].v_q4_pages[local_id];

        for layer in 0..self.num_layers {
            for kv_head in 0..self.num_kv_heads {
                for slot in 0..bs {
                    let q4_base = unsafe {
                        k_q4.add(
                            (layer * self.num_kv_heads + kv_head) * bs * q4_bytes_per_slot + slot * q4_bytes_per_slot
                        )
                    };
                    let f32_off = ((layer * self.num_kv_heads + kv_head) * bs + slot) * self.head_dim;
                    let f32_k = unsafe { std::slice::from_raw_parts_mut(k_ptr.add(f32_off), self.head_dim) };
                    let f32_v = unsafe { std::slice::from_raw_parts_mut(v_ptr.add(f32_off), self.head_dim) };
                    let q4_v_base = unsafe {
                        v_q4.add(
                            (layer * self.num_kv_heads + kv_head) * bs * q4_bytes_per_slot + slot * q4_bytes_per_slot
                        )
                    };
                    for blk in 0..blocks_per_head {
                        let blk_off = blk * q4_blk_size;
                        unsafe fn dequant_block(q4_src: *const u8, f32_dst: &mut [f32], head_dim: usize, blk: usize, blk_off: usize) {
                            let d = half::f16::from_le_bytes([
                                *q4_src.add(blk_off),
                                *q4_src.add(blk_off + 1),
                            ]).to_f32();
                            let dmin = half::f16::from_le_bytes([
                                *q4_src.add(blk_off + 2),
                                *q4_src.add(blk_off + 3),
                            ]).to_f32();
                            let nib_base = q4_src.add(blk_off + 4);
                            for i in 0..32 {
                                if blk * 32 + i >= head_dim { break; }
                                let nib = if i % 2 == 0 {
                                    *nib_base.add(i / 2) & 0x0F
                                } else {
                                    *nib_base.add(i / 2) >> 4
                                };
                                f32_dst[blk * 32 + i] = d * nib as f32 + dmin;
                            }
                        }
                        unsafe { dequant_block(q4_base, f32_k, self.head_dim, blk, blk_off); }
                        unsafe { dequant_block(q4_v_base, f32_v, self.head_dim, blk, blk_off); }
                    }
                }
            }
        }

        self.zones[zi].k_pages[local_id] = k_ptr;
        self.zones[zi].v_pages[local_id] = v_ptr;
        self.zones[zi].page_capacities[local_id] = (layout, byte_size);
    }

    fn evict_one_page_from_zone(&mut self, zi: usize) {
        // Entropy-guided eviction: among the LRU tail-end candidate pages,
        // evict the one with lowest accumulated attention importance.
        let evict = {
            let zone = &self.zones[zi];
            let tail = match zone.lru_tail {
                Some(id) => id,
                None => return,
            };
            // Scan up to 8 pages from the LRU tail (least recently used end)
            let mut best = tail;
            let mut best_importance = zone.entropy_importance[tail];
            let mut cursor = zone.lru_prev[tail];
            for _ in 0..7 {
                match cursor {
                    Some(id) => {
                        let imp = zone.entropy_importance[id];
                        if imp < best_importance {
                            best_importance = imp;
                            best = id;
                        }
                        cursor = zone.lru_prev[id];
                    }
                    None => break,
                }
            }
            best
        };

        let q4_bytes = self.q4_page_byte_size_for_zone(zi);
        let q4_layout = self.q4_layout_for_zone(zi);
        self.init_cold_file();

        if let Some(ref mut file) = self.cold_file {
            let num_slots = self.cold_capacity;
            let slot = (0..num_slots)
                .find(|&i| !self.cold_slot_used[(self.cold_write_cursor + i) % num_slots])
                .map(|i| (self.cold_write_cursor + i) % num_slots)
                .unwrap_or_else(|| {
                    let old = self.cold_write_cursor;
                    self.cold_write_cursor = (self.cold_write_cursor + 1) % num_slots;
                    old
                });

            let slot_size = q4_bytes * 2;
            let offset = (slot * slot_size) as u64;
            file.seek(SeekFrom::Start(offset)).expect("cold store seek failed");
            unsafe {
                let k_slice = std::slice::from_raw_parts(self.zones[zi].k_q4_pages[evict] as *const u8, q4_bytes);
                let v_slice = std::slice::from_raw_parts(self.zones[zi].v_q4_pages[evict] as *const u8, q4_bytes);
                file.write_all(k_slice).expect("cold store write failed");
                file.write_all(v_slice).expect("cold store write failed");
            }

            self.cold_slot_used[slot] = true;
            self.cold_write_cursor = (slot + 1) % num_slots;
            self.zones[zi].page_state[evict] = PageState::Cold(slot as u64);
            self.zones[zi].hot_count -= 1;
            self.remove_from_lru_in_zone(zi, evict);

            unsafe {
                dealloc(self.zones[zi].k_q4_pages[evict] as *mut u8, q4_layout);
                dealloc(self.zones[zi].v_q4_pages[evict] as *mut u8, q4_layout);
            }
            self.zones[zi].k_q4_pages[evict] = std::ptr::null_mut();
            self.zones[zi].v_q4_pages[evict] = std::ptr::null_mut();
            if !self.zones[zi].k_pages[evict].is_null() {
                let (layout, _) = self.f32_page_byte_size_for_zone(zi);
                unsafe {
                    dealloc(self.zones[zi].k_pages[evict] as *mut u8, layout);
                    dealloc(self.zones[zi].v_pages[evict] as *mut u8, layout);
                }
                self.zones[zi].k_pages[evict] = std::ptr::null_mut();
                self.zones[zi].v_pages[evict] = std::ptr::null_mut();
            }
        }
    }

    fn reload_cold_page_in_zone(&mut self, zi: usize, local_id: usize) {
        let slot = match self.zones[zi].page_state[local_id] {
            PageState::Cold(off) => off as usize,
            _ => return,
        };

        let q4_bytes = self.q4_page_byte_size_for_zone(zi);
        let q4_layout = self.q4_layout_for_zone(zi);

        if self.zones[zi].hot_count >= self.ram_page_limit {
            self.evict_one_page_from_zone(zi);
        }

        if self.zones[zi].k_q4_pages[local_id].is_null() {
            let k_q4_ptr = unsafe { alloc_zeroed(q4_layout) } as *mut u8;
            let v_q4_ptr = unsafe { alloc_zeroed(q4_layout) } as *mut u8;
            if k_q4_ptr.is_null() || v_q4_ptr.is_null() {
                panic!("PagedKVCache OOM on cold reload zone {} page {}", zi, local_id);
            }
            self.zones[zi].k_q4_pages[local_id] = k_q4_ptr;
            self.zones[zi].v_q4_pages[local_id] = v_q4_ptr;
        }

        if let Some(ref mut file) = self.cold_file {
            let offset = (slot * q4_bytes * 2) as u64;
            let mut buf = vec![0u8; q4_bytes * 2];
            file.seek(SeekFrom::Start(offset)).expect("cold store seek failed");
            file.read_exact(&mut buf).expect("cold store read failed");
            unsafe {
                std::ptr::copy_nonoverlapping(buf.as_ptr(), self.zones[zi].k_q4_pages[local_id], q4_bytes);
                std::ptr::copy_nonoverlapping(buf.as_ptr().add(q4_bytes), self.zones[zi].v_q4_pages[local_id], q4_bytes);
            }
            self.cold_slot_used[slot] = false;
        }

        self.zones[zi].page_state[local_id] = PageState::Hot;
        self.zones[zi].hot_count += 1;

        let zone = &mut self.zones[zi];
        zone.lru_prev[local_id] = None;
        zone.lru_next[local_id] = zone.lru_head;
        if let Some(h) = zone.lru_head { zone.lru_prev[h] = Some(local_id); }
        zone.lru_head = Some(local_id);
        if zone.lru_tail.is_none() { zone.lru_tail = Some(local_id); }
    }

    fn ensure_page(&mut self, flat_id: usize) {
        let (zi, local_id) = self.resolve_flat(flat_id);

        while local_id >= self.zones[zi].k_pages.len() {
            if self.zones[zi].hot_count >= self.ram_page_limit {
                self.evict_one_page_from_zone(zi);
            }
            self.alloc_hot_page_in_zone(zi);
        }

        match self.zones[zi].page_state[local_id] {
            PageState::Cold(_) => self.reload_cold_page_in_zone(zi, local_id),
            PageState::Hot => {}
        }

        self.touch_lru_in_zone(zi, local_id);
    }

    pub fn ensure_pages_hot(&mut self, start_page: usize, end_page: usize) {
        for pid in start_page..=end_page {
            self.ensure_page(pid);
        }
    }

    pub fn ensure_pages_hot_f32(&mut self, start_page: usize, end_page: usize) {
        for pid in start_page..=end_page {
            self.ensure_page(pid);
            let (zi, local_id) = self.resolve_flat(pid);
            self.ensure_f32_page_in_zone(zi, local_id);
        }
    }

    /// Record attention importance for a position: higher weight means this
    /// position receives focused attention (low entropy) and should stay hot.
    pub fn record_importance(&mut self, pos: usize, weight: f64) {
        let zi = self.zone_index(pos);
        let local = self.local_page_id(zi, pos);
        if local < self.zones[zi].entropy_importance.len() {
            self.zones[zi].entropy_importance[local] += weight;
        }
    }

    /// Reset all entropy importance scores (e.g. start of a new request).
    pub fn reset_entropy(&mut self) {
        for zone in &mut self.zones {
            for score in &mut zone.entropy_importance {
                *score = 0.0;
            }
        }
    }

    /// Get the current entropy importance for a page covering `pos`.
    pub fn entropy_for(&self, pos: usize) -> f64 {
        let zi = self.zone_index(pos);
        let local = self.local_page_id(zi, pos);
        if local < self.zones[zi].entropy_importance.len() {
            self.zones[zi].entropy_importance[local]
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn page_id(&self, pos: usize) -> usize {
        let zi = self.zone_index(pos);
        let local = self.local_page_id(zi, pos);
        self.flat_page_id(zi, local)
    }

    #[inline(always)]
    pub fn page_start_for(&self, pos: usize) -> usize {
        pos - self.slot_in_page(pos)
    }

    #[inline(always)]
    pub fn slot_in_page(&self, pos: usize) -> usize {
        let zi = self.zone_index(pos);
        let bs = self.zones[zi].block_size;
        let zs = self.zones[zi].zone_start;
        (pos - zs) % bs
    }

    pub fn save(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        let pos = self.current_pos;
        let zi = self.zone_index(pos);
        let bs = self.zones[zi].block_size;
        let zs = self.zones[zi].zone_start;
        let local_pid = (pos - zs) / bs;
        let slot = (pos - zs) % bs;
        let flat_pid = self.flat_page_id(zi, local_pid);
        self.ensure_page(flat_pid);

        let blocks_per_head = (self.head_dim + 31) / 32;
        let q4bph = blocks_per_head * 20;

        for kv_head in 0..self.num_kv_heads {
            let page_off = (layer * self.num_kv_heads + kv_head) * bs * q4bph + slot * q4bph;
            let src_off = kv_head * self.head_dim;
            let k_src = &k[src_off..src_off + self.head_dim];
            let v_src = &v[src_off..src_off + self.head_dim];

            for blk in 0..blocks_per_head {
                let blk_start = blk * 32;
                let blk_end = (blk_start + 32).min(self.head_dim);
                let blk_len = blk_end - blk_start;

                macro_rules! quantize_block {
                    ($src:expr, $q4_base:expr) => {{
                        let (d_max, d_min) = {
                            let mut mx = 0.0f32;
                            let mut mn = f32::MAX;
                            for &val in &$src[blk_start..blk_end] {
                                mx = mx.max(val.abs());
                                mn = mn.min(val);
                            }
                            (mx, mn)
                        };
                        let d = if d_max > 1e-10 { d_max / 7.0 } else { 0.0 };
                        let dmin = d_min;
                        unsafe {
                            std::ptr::write($q4_base as *mut u16, u16::from_le_bytes(half::f16::from_f32(d).to_le_bytes()));
                            std::ptr::write($q4_base.add(2) as *mut u16, u16::from_le_bytes(half::f16::from_f32(dmin).to_le_bytes()));
                            for i in 0..blk_len {
                                let q = if d > 1e-10 {
                                    (($src[blk_start + i] - dmin) / d).round().max(0.0).min(15.0) as u8
                                } else { 0 };
                                let nib_off = $q4_base.add(4 + i / 2);
                                let shift = if i % 2 == 0 { 0 } else { 4 };
                                std::ptr::write(nib_off, std::ptr::read(nib_off) | (q << shift));
                            }
                        }
                    }};
                }

                let k_q4_base = unsafe { self.zones[zi].k_q4_pages[local_pid].add(page_off + blk * 20) };
                quantize_block!(k_src, k_q4_base);
                let v_q4_base = unsafe { self.zones[zi].v_q4_pages[local_pid].add(page_off + blk * 20) };
                quantize_block!(v_src, v_q4_base);
            }

            if !self.zones[zi].k_pages[local_pid].is_null() {
                let f32_off = ((layer * self.num_kv_heads + kv_head) * bs + slot) * self.head_dim;
                unsafe {
                    std::ptr::copy_nonoverlapping(k.as_ptr().add(src_off), self.zones[zi].k_pages[local_pid].add(f32_off), self.head_dim);
                    std::ptr::copy_nonoverlapping(v.as_ptr().add(src_off), self.zones[zi].v_pages[local_pid].add(f32_off), self.head_dim);
                }
            }
        }
    }

    pub unsafe fn get_k_unchecked(&self, layer: usize, kv_head: usize, pos: usize) -> &[f32] {
        let zi = self.zone_index(pos);
        let bs = self.zones[zi].block_size;
        let zs = self.zones[zi].zone_start;
        let local_pid = (pos - zs) / bs;
        let slot = (pos - zs) % bs;
        assert!(!self.zones[zi].k_pages[local_pid].is_null(),
            "get_k_unchecked: FP32 not allocated for zone {} page {}. Call ensure_pages_hot_f32() first", zi, local_pid);
        let off = ((layer * self.num_kv_heads + kv_head) * bs + slot) * self.head_dim;
        std::slice::from_raw_parts(self.zones[zi].k_pages[local_pid].add(off), self.head_dim)
    }

    pub unsafe fn get_v_unchecked(&self, layer: usize, kv_head: usize, pos: usize) -> &[f32] {
        let zi = self.zone_index(pos);
        let bs = self.zones[zi].block_size;
        let zs = self.zones[zi].zone_start;
        let local_pid = (pos - zs) / bs;
        let slot = (pos - zs) % bs;
        assert!(!self.zones[zi].v_pages[local_pid].is_null(),
            "get_v_unchecked: FP32 not allocated for zone {} page {}. Call ensure_pages_hot_f32() first", zi, local_pid);
        let off = ((layer * self.num_kv_heads + kv_head) * bs + slot) * self.head_dim;
        std::slice::from_raw_parts(self.zones[zi].v_pages[local_pid].add(off), self.head_dim)
    }

    #[inline(always)]
    pub fn k_page_ptr(&self, layer: usize, kv_head: usize, page_id: usize) -> *const f32 {
        let (zi, local_id) = self.resolve_flat(page_id);
        assert!(!self.zones[zi].k_pages[local_id].is_null(),
            "k_page_ptr: FP32 not allocated for zone {} page {}", zi, local_id);
        let bs = self.zones[zi].block_size;
        let off = ((layer * self.num_kv_heads + kv_head) * bs) * self.head_dim;
        unsafe { self.zones[zi].k_pages[local_id].add(off) as *const f32 }
    }

    #[inline(always)]
    pub fn v_page_ptr(&self, layer: usize, kv_head: usize, page_id: usize) -> *const f32 {
        let (zi, local_id) = self.resolve_flat(page_id);
        assert!(!self.zones[zi].v_pages[local_id].is_null(),
            "v_page_ptr: FP32 not allocated for zone {} page {}", zi, local_id);
        let bs = self.zones[zi].block_size;
        let off = ((layer * self.num_kv_heads + kv_head) * bs) * self.head_dim;
        unsafe { self.zones[zi].v_pages[local_id].add(off) as *const f32 }
    }

    #[inline(always)]
    pub fn k_q4_page_ptr(&self, layer: usize, kv_head: usize, page_id: usize) -> *const u8 {
        let (zi, local_id) = self.resolve_flat(page_id);
        assert!(!self.zones[zi].k_q4_pages[local_id].is_null(),
            "k_q4_page_ptr on evicted zone {} page {}", zi, local_id);
        let bs = self.zones[zi].block_size;
        let blocks_per_head = (self.head_dim + 31) / 32;
        let bytes_per_head = blocks_per_head * 20;
        let off = (layer * self.num_kv_heads + kv_head) * bs * bytes_per_head;
        unsafe { self.zones[zi].k_q4_pages[local_id].add(off) as *const u8 }
    }

    #[inline(always)]
    pub fn v_q4_page_ptr(&self, layer: usize, kv_head: usize, page_id: usize) -> *const u8 {
        let (zi, local_id) = self.resolve_flat(page_id);
        assert!(!self.zones[zi].v_q4_pages[local_id].is_null(),
            "v_q4_page_ptr on evicted zone {} page {}", zi, local_id);
        let bs = self.zones[zi].block_size;
        let blocks_per_head = (self.head_dim + 31) / 32;
        let bytes_per_head = blocks_per_head * 20;
        let off = (layer * self.num_kv_heads + kv_head) * bs * bytes_per_head;
        unsafe { self.zones[zi].v_q4_pages[local_id].add(off) as *const u8 }
    }

    pub fn save_at(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let zi = self.zone_index(pos);
        let bs = self.zones[zi].block_size;
        let zs = self.zones[zi].zone_start;
        let local_pid = (pos - zs) / bs;
        let slot = (pos - zs) % bs;
        let flat_pid = self.flat_page_id(zi, local_pid);
        self.ensure_page(flat_pid);

        let blocks_per_head = (self.head_dim + 31) / 32;
        let q4bph = blocks_per_head * 20;

        for kv_head in 0..self.num_kv_heads {
            let page_off = (layer * self.num_kv_heads + kv_head) * bs * q4bph + slot * q4bph;
            let src_off = kv_head * self.head_dim;
            let k_src = &k[src_off..src_off + self.head_dim];
            let v_src = &v[src_off..src_off + self.head_dim];

            for blk in 0..blocks_per_head {
                let blk_start = blk * 32;
                let blk_end = (blk_start + 32).min(self.head_dim);
                let blk_len = blk_end - blk_start;

                let (d_max_k, d_min_k) = k_src[blk_start..blk_end].iter()
                    .fold((0.0f32, f32::MAX), |(mx, mn), &v| (mx.max(v.abs()), mn.min(v)));
                let d_k = if d_max_k > 1e-10 { d_max_k / 7.0 } else { 0.0 };
                let dmin_k = d_min_k;

                let q4_base = unsafe { self.zones[zi].k_q4_pages[local_pid].add(page_off + blk * 20) };
                unsafe {
                    std::ptr::write(q4_base as *mut u16, u16::from_le_bytes(half::f16::from_f32(d_k).to_le_bytes()));
                    std::ptr::write(q4_base.add(2) as *mut u16, u16::from_le_bytes(half::f16::from_f32(dmin_k).to_le_bytes()));
                    for i in 0..blk_len {
                        let q = if d_k > 1e-10 {
                            ((k_src[blk_start + i] - dmin_k) / d_k).round().max(0.0).min(15.0) as u8
                        } else { 0 };
                        let nib_off = q4_base.add(4 + i / 2);
                        let shift = if i % 2 == 0 { 0 } else { 4 };
                        std::ptr::write(nib_off, std::ptr::read(nib_off) | (q << shift));
                    }
                }

                let (d_max_v, d_min_v) = v_src[blk_start..blk_end].iter()
                    .fold((0.0f32, f32::MAX), |(mx, mn), &v| (mx.max(v.abs()), mn.min(v)));
                let d_v = if d_max_v > 1e-10 { d_max_v / 7.0 } else { 0.0 };
                let dmin_v = d_min_v;

                let v_q4_base = unsafe { self.zones[zi].v_q4_pages[local_pid].add(page_off + blk * 20) };
                unsafe {
                    std::ptr::write(v_q4_base as *mut u16, u16::from_le_bytes(half::f16::from_f32(d_v).to_le_bytes()));
                    std::ptr::write(v_q4_base.add(2) as *mut u16, u16::from_le_bytes(half::f16::from_f32(dmin_v).to_le_bytes()));
                    for i in 0..blk_len {
                        let q = if d_v > 1e-10 {
                            ((v_src[blk_start + i] - dmin_v) / d_v).round().max(0.0).min(15.0) as u8
                        } else { 0 };
                        let nib_off = v_q4_base.add(4 + i / 2);
                        let shift = if i % 2 == 0 { 0 } else { 4 };
                        std::ptr::write(nib_off, std::ptr::read(nib_off) | (q << shift));
                    }
                }
            }
        }
    }

    pub fn advance(&mut self) {
        self.current_pos += 1;
    }

    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    pub fn reset(&mut self) {
        self.current_pos = 0;
    }
}

impl Drop for PagedKVCache {
    fn drop(&mut self) {
        for zi in 0..self.zones.len() {
            let q4_layout = self.q4_layout_for_zone(zi);
            for i in 0..self.zones[zi].k_pages.len() {
                if !self.zones[zi].k_pages[i].is_null() {
                    let (f32_layout, _) = self.f32_page_byte_size_for_zone(zi);
                    unsafe {
                        dealloc(self.zones[zi].k_pages[i] as *mut u8, f32_layout);
                        dealloc(self.zones[zi].v_pages[i] as *mut u8, f32_layout);
                    }
                }
                if !self.zones[zi].k_q4_pages[i].is_null() {
                    unsafe {
                        dealloc(self.zones[zi].k_q4_pages[i] as *mut u8, q4_layout);
                        dealloc(self.zones[zi].v_q4_pages[i] as *mut u8, q4_layout);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_cache_basic_rw() {
        let mut cache = PagedKVCache::with_block_size(2, 4, 128, 64, 32);
        let n_layers = 2;
        let n_kv = 4;
        let head_dim = 64;

        for pos in 0..3 {
            let mut k = vec![0.0f32; n_kv * head_dim];
            let mut v = vec![0.0f32; n_kv * head_dim];
            for h in 0..n_kv {
                k[h * head_dim] = (pos * 100 + h) as f32;
                v[h * head_dim] = (pos * 1000 + h) as f32;
            }
            for l in 0..n_layers {
                cache.save(l, &k, &v);
            }
            cache.advance();
        }

        assert_eq!(cache.current_pos(), 3);
        cache.ensure_pages_hot_f32(0, 0);

        for l in 0..n_layers {
            for h in 0..n_kv {
                for pos in 0..3 {
                    let k_slice = unsafe { cache.get_k_unchecked(l, h, pos) };
                    let v_slice = unsafe { cache.get_v_unchecked(l, h, pos) };
                    let expected_k = (pos * 100 + h) as f32;
                    let expected_v = (pos * 1000 + h) as f32;
                    assert!((k_slice[0] - expected_k).abs() < 1.0,
                        "K mismatch layer={} head={} pos={}: got {} expected {}", l, h, pos, k_slice[0], expected_k);
                    assert!((v_slice[0] - expected_v).abs() < 1.0,
                        "V mismatch layer={} head={} pos={}: got {} expected {}", l, h, pos, v_slice[0], expected_v);
                }
            }
        }
    }

    #[test]
    fn test_paged_cache_multiple_pages() {
        let mut cache = PagedKVCache::with_block_size(1, 1, 1024, 64, 32);
        let n_tokens = 32;

        for pos in 0..n_tokens {
            let k = vec![pos as f32; 64];
            let v = vec![(pos * 2) as f32; 64];
            cache.save(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.current_pos(), n_tokens);

        let last_pid = cache.page_id(n_tokens - 1);
        cache.ensure_pages_hot_f32(0, last_pid);

        for pos in 0..n_tokens {
            let k = unsafe { cache.get_k_unchecked(0, 0, pos) };
            let v = unsafe { cache.get_v_unchecked(0, 0, pos) };
            assert_eq!(k[0], pos as f32, "pos={}", pos);
            assert_eq!(v[0], (pos * 2) as f32);
        }

        for pos in 0..n_tokens {
            let pid = cache.page_id(pos);
            let k_ptr = cache.k_page_ptr(0, 0, pid);
            let v_ptr = cache.v_page_ptr(0, 0, pid);
            unsafe {
                let bs = cache.block_size_for(pos);
                assert!((*k_ptr - (pos - pos % bs) as f32).abs() < 0.5, "K pid={} pos={}", pid, pos);
                assert!((*v_ptr - (pos - pos % bs) as f32 * 2.0).abs() < 0.5, "V pid={} pos={}", pid, pos);
            }
        }
    }

    #[test]
    fn test_paged_cache_reset() {
        let mut cache = PagedKVCache::with_block_size(1, 1, 1024, 64, 32);
        for pos in 0..10 {
            let k = vec![pos as f32; 64];
            let v = vec![pos as f32; 64];
            cache.save(0, &k, &v);
            cache.advance();
        }

        cache.reset();
        assert_eq!(cache.current_pos(), 0);

        let k = vec![42.0f32; 64];
        let v = vec![42.0f32; 64];
        cache.save(0, &k, &v);
        cache.advance();
        cache.ensure_pages_hot_f32(0, 0);

        let k_read = unsafe { cache.get_k_unchecked(0, 0, 0) };
        assert!((k_read[0] - 42.0).abs() < 0.5, "got {}", k_read[0]);
    }

    #[test]
    fn test_tiered_eviction_basic() {
        std::env::set_var("SWAMP_CACHE_RAM_MB", "0");
        let mut cache = PagedKVCache::with_block_size(1, 1, 128, 64, 32);
        cache.ram_page_limit = 2;

        let n_tokens = 12;
        for pos in 0..n_tokens {
            let k = vec![pos as f32; 64];
            let v = vec![(pos * 2) as f32; 64];
            cache.save(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.current_pos(), n_tokens);

        let last_pid = cache.page_id(n_tokens - 1);
        cache.ensure_pages_hot_f32(last_pid, last_pid);
        let k2 = unsafe { cache.get_k_unchecked(0, 0, 8) };
        assert!((k2[0] - 8.0).abs() < 0.5, "Page 2: got {}", k2[0]);

        cache.ensure_pages_hot_f32(0, 0);
        let k0 = unsafe { cache.get_k_unchecked(0, 0, 0) };
        assert!((k0[0] - 0.0).abs() < 0.5, "Page 0: got {}", k0[0]);

        std::env::remove_var("SWAMP_CACHE_RAM_MB");
    }

    #[test]
    fn test_tiered_no_eviction_within_limit() {
        let mut cache = PagedKVCache::with_block_size(1, 1, 128, 64, 32);
        cache.ram_page_limit = 10;

        let n_tokens = 8;
        for pos in 0..n_tokens {
            let k = vec![pos as f32; 64];
            let v = vec![(pos * 2) as f32; 64];
            cache.save(0, &k, &v);
            cache.advance();
        }

        let last_pid = cache.page_id(n_tokens - 1);
        cache.ensure_pages_hot_f32(0, last_pid);
        for pos in 0..n_tokens {
            let k = unsafe { cache.get_k_unchecked(0, 0, pos) };
            assert!((k[0] - pos as f32).abs() < 0.5, "pos={} got {}", pos, k[0]);
        }
    }

    #[test]
    fn test_zone_boundaries() {
        let mut cache = PagedKVCache::with_block_size(1, 1, 140000, 64, 32);
        assert_eq!(cache.block_size_for(0), 32);
        assert_eq!(cache.block_size_for(4095), 32);
        assert_eq!(cache.block_size_for(4096), 128);
        assert_eq!(cache.block_size_for(131071), 128);
        assert_eq!(cache.block_size_for(131072), 1024);

        let n_tokens = 140000;
        for pos in 0..n_tokens {
            let k = vec![pos as f32; 64];
            let v = vec![(pos * 2) as f32; 64];
            cache.save(0, &k, &v);
            cache.advance();
        }

        // Verify page IDs across zones
        assert_eq!(cache.page_id(0), 0);
        assert_eq!(cache.page_id(31), 0); // zone 0, page 0
        assert_eq!(cache.page_id(32), 1); // zone 0, page 1
        assert_eq!(cache.page_id(4095), 127); // zone 0, last page
        assert_eq!(cache.page_id(4096), 128 + 0); // zone 1, page 0
        assert_eq!(cache.page_id(4223), 128 + 0); // zone 1, page 0 (128 positions)
        assert_eq!(cache.page_id(4224), 128 + 1); // zone 1, page 1
        assert_eq!(cache.page_id(131071), 128 + 991); // zone 1, page 991
        assert_eq!(cache.page_id(131072), 128 + 992); // zone 2, page 0
        assert_eq!(cache.page_id(132095), 128 + 992); // zone 2, page 0 (1024 positions)
        assert_eq!(cache.page_id(132096), 128 + 993); // zone 2, page 1
    }

    #[test]
    fn test_zone_read_write() {
        let mut cache = PagedKVCache::with_block_size(2, 2, 200000, 64, 32);

        // Write to zone 0
        for pos in 0..100 {
            let k = vec![pos as f32; 128];
            let v = vec![(pos * 3) as f32; 128];
            for l in 0..2 { cache.save(l, &k, &v); }
            cache.advance();
        }

        // Write to zone 1
        for pos in 100..4200 {
            let k = vec![pos as f32; 128];
            let v = vec![(pos * 3) as f32; 128];
            for l in 0..2 { cache.save(l, &k, &v); }
            cache.advance();
        }

        assert_eq!(cache.current_pos(), 4200);

        // Verify reads across zones
        // Q4 stores d/dmin as f16; at magnitude ~4000 f16 step=2, at ~12000
        // step=8, causing dmin rounding error up to ~4 for v values.
        // Use generous tolerance 10 to isolate zone logic from quantization.
        let last_pid = cache.page_id(4199);
        cache.ensure_pages_hot_f32(0, last_pid);

        for pos in [0, 99, 100, 4096, 4198] {
            let expected_k = pos as f32;
            let expected_v = (pos * 3) as f32;
            let k = unsafe { cache.get_k_unchecked(0, 0, pos) };
            assert!((k[0] - expected_k).abs() < 10.0, "pos={} k got {} expected {}", pos, k[0], expected_k);
            let v = unsafe { cache.get_v_unchecked(0, 1, pos) };
            assert!((v[0] - expected_v).abs() < 10.0, "pos={} v got {} expected {}", pos, v[0], expected_v);
        }
    }
}
