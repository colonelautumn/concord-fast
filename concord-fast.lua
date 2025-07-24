-- concord-fast
-- Needs unlocked FFI module for Gmod to work

-- V2

local ffi = require("ffi")
local bit = require("bit")
local jit = require("jit")

-- Enable JIT compilation with all optimizations
jit.on()
jit.opt.start("hotloop=1", "hotexit=1", "fold", "cse", "dce")

-- Massive C type definitions for ultra-performance FFI
ffi.cdef[[
    // Standard C library functions
    void *malloc(size_t size);
    void *calloc(size_t nmemb, size_t size);
    void *realloc(void *ptr, size_t size);
    void free(void *ptr);
    void *memcpy(void *dest, const void *src, size_t n);
    void *memmove(void *dest, const void *src, size_t n);
    void *memset(void *s, int c, size_t n);
    int memcmp(const void *s1, const void *s2, size_t n);
    
    // Math operations
    double sin(double x);
    double cos(double x);
    double tan(double x);
    double asin(double x);
    double acos(double x);
    double atan(double x);
    double atan2(double y, double x);
    double sinh(double x);
    double cosh(double x);
    double tanh(double x);
    double exp(double x);
    double exp2(double x);
    double log(double x);
    double log10(double x);
    double log2(double x);
    double sqrt(double x);
    double cbrt(double x);
    double pow(double x, double y);
    double floor(double x);
    double ceil(double x);
    double round(double x);
    double trunc(double x);
    double fabs(double x);
    double fmod(double x, double y);
    double frexp(double x, int *exp);
    double ldexp(double x, int exp);
    double modf(double x, double *iptr);
    
    // Fast math variants
    float sinf(float x);
    float cosf(float x);
    float tanf(float x);
    float sqrtf(float x);
    float powf(float x, float y);
    float floorf(float x);
    float ceilf(float x);
    float fabsf(float x);
    
    // String operations
    size_t strlen(const char *s);
    char *strcpy(char *dest, const char *src);
    char *strncpy(char *dest, const char *src, size_t n);
    char *strcat(char *dest, const char *src);
    char *strncat(char *dest, const char *src, size_t n);
    int strcmp(const char *s1, const char *s2);
    int strncmp(const char *s1, const char *s2, size_t n);
    char *strchr(const char *s, int c);
    char *strrchr(const char *s, int c);
    char *strstr(const char *haystack, const char *needle);
    char *strtok(char *str, const char *delim);
    
    // Time and system operations
    typedef long time_t;
    typedef long clock_t;
    time_t time(time_t *tloc);
    clock_t clock(void);
    struct timespec {
        time_t tv_sec;
        long tv_nsec;
    };
    int clock_gettime(int clk_id, struct timespec *tp);
    
    // I/O operations
    int printf(const char *format, ...);
    int sprintf(char *str, const char *format, ...);
    int snprintf(char *str, size_t size, const char *format, ...);
    
    // Random number generation
    int rand(void);
    void srand(unsigned int seed);
    
    // Advanced data structures
    
    // 3D Vector structures (single and double precision)
    typedef struct {
        double x, y, z;
    } vec3d_t;
    
    typedef struct {
        float x, y, z;
    } vec3f_t;
    
    // 4D Vector for homogeneous coordinates
    typedef struct {
        double x, y, z, w;
    } vec4d_t;
    
    typedef struct {
        float x, y, z, w;
    } vec4f_t;
    
    // 2D Vector
    typedef struct {
        double x, y;
    } vec2d_t;
    
    typedef struct {
        float x, y;
    } vec2f_t;
    
    // Quaternion for rotations
    typedef struct {
        double x, y, z, w;
    } quat_t;
    
    // 4x4 Matrix for transformations
    typedef struct {
        double m[16];
    } mat4d_t;
    
    typedef struct {
        float m[16];
    } mat4f_t;
    
    // 3x3 Matrix
    typedef struct {
        double m[9];
    } mat3d_t;
    
    // AABB (Axis-Aligned Bounding Box)
    typedef struct {
        vec3d_t min, max;
    } aabb_t;
    
    // Plane
    typedef struct {
        vec3d_t normal;
        double distance;
    } plane_t;
    
    // Ray
    typedef struct {
        vec3d_t origin, direction;
    } ray_t;
    
    // Sphere
    typedef struct {
        vec3d_t center;
        double radius;
    } sphere_t;
    
    // Color structures
    typedef struct {
        uint8_t r, g, b, a;
    } color32_t;
    
    typedef struct {
        float r, g, b, a;
    } colorf_t;
    
    typedef struct {
        double r, g, b, a;
    } colord_t;
    
    // Advanced hash table structures
    typedef struct hash_node {
        uint64_t hash;
        uint32_t key_size;
        uint32_t value_size;
        void *key;
        void *value;
        struct hash_node *next;
    } hash_node_t;
    
    typedef struct {
        hash_node_t **buckets;
        uint32_t bucket_count;
        uint32_t item_count;
        double load_factor;
    } hash_table_t;
    
    // Lock-free queue node
    typedef struct queue_node {
        void *data;
        struct queue_node *next;
    } queue_node_t;
    
    // Memory pool structures
    typedef struct pool_block {
        struct pool_block *next;
        uint8_t data[];
    } pool_block_t;
    
    typedef struct {
        pool_block_t *free_blocks;
        uint32_t block_size;
        uint32_t blocks_allocated;
        uint32_t blocks_free;
    } memory_pool_t;
    
    // Fast array structures
    typedef struct {
        void **data;
        uint32_t size;
        uint32_t capacity;
    } dynamic_array_t;
    
    // Spatial partitioning structures
    typedef struct octree_node {
        aabb_t bounds;
        void **objects;
        uint32_t object_count;
        struct octree_node *children[8];
        bool is_leaf;
    } octree_node_t;
    
    // B-Tree node for fast indexing
    typedef struct btree_node {
        void **keys;
        void **values;
        struct btree_node **children;
        uint32_t key_count;
        bool is_leaf;
    } btree_node_t;
    
    // Thread synchronization (if available)
    typedef struct {
        volatile int value;
    } atomic_int_t;
    
    // SIMD-like vector operations (emulated)
    typedef struct {
        float v[4];
    } simd_float4_t;
    
    typedef struct {
        double v[4];
    } simd_double4_t;
    
    // Network packet structure
    typedef struct {
        uint32_t size;
        uint32_t type;
        uint8_t data[];
    } net_packet_t;
    
    // Fast string builder
    typedef struct {
        char *buffer;
        uint32_t length;
        uint32_t capacity;
    } string_builder_t;
    
    // Compression structures
    typedef struct {
        uint8_t *data;
        uint32_t size;
        uint32_t compressed_size;
    } compressed_data_t;
    
    // Performance counters
    typedef struct {
        uint64_t calls;
        double total_time;
        double min_time;
        double max_time;
        double avg_time;
    } perf_counter_t;
]]

-- Create the main namespace
local concord = {}
concord.version = "2.0.0"

-- Performance monitoring
local perf_counters = {}
local profile_stack = {}

-- ============================================================================
-- ADVANCED MEMORY MANAGEMENT
-- ============================================================================

concord.memory = {}

-- Ultra-fast memory pools with multiple size classes
local memory_pools = {}
local pool_sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}
local large_alloc_threshold = 65536

-- Memory statistics
local memory_stats = {
    pools_allocated = 0,
    pools_freed = 0,
    large_allocated = 0,
    large_freed = 0,
    bytes_allocated = 0,
    bytes_freed = 0
}

function concord.memory.init_pools()
    for _, size in ipairs(pool_sizes) do
        memory_pools[size] = ffi.new("memory_pool_t")
        local pool = memory_pools[size]
        pool.free_blocks = nil
        pool.block_size = size
        pool.blocks_allocated = 0
        pool.blocks_free = 0
    end
end

-- Ultra-fast allocation with pool selection
function concord.memory.fast_alloc(size)
    if size > large_alloc_threshold then
        memory_stats.large_allocated = memory_stats.large_allocated + 1
        memory_stats.bytes_allocated = memory_stats.bytes_allocated + size
        return ffi.C.malloc(size)
    end
    
    -- Find the smallest suitable pool
    local pool_size = 16
    for _, ps in ipairs(pool_sizes) do
        if size <= ps then
            pool_size = ps
            break
        end
    end
    
    local pool = memory_pools[pool_size]
    if not pool then
        memory_stats.large_allocated = memory_stats.large_allocated + 1
        memory_stats.bytes_allocated = memory_stats.bytes_allocated + size
        return ffi.C.malloc(size)
    end
    
    -- Try to reuse a freed block
    if pool.free_blocks ~= nil then
        local block = pool.free_blocks
        pool.free_blocks = block.next
        pool.blocks_free = pool.blocks_free - 1
        memory_stats.pools_allocated = memory_stats.pools_allocated + 1
        return block
    end
    
    -- Allocate new block
    local block = ffi.cast("pool_block_t*", ffi.C.malloc(ffi.sizeof("pool_block_t") + pool_size))
    if block ~= nil then
        pool.blocks_allocated = pool.blocks_allocated + 1
        memory_stats.pools_allocated = memory_stats.pools_allocated + 1
        memory_stats.bytes_allocated = memory_stats.bytes_allocated + pool_size
        return block.data
    end
    
    return nil
end

function concord.memory.fast_free(ptr, size)
    if ptr == nil then return end
    
    if size > large_alloc_threshold then
        memory_stats.large_freed = memory_stats.large_freed + 1
        memory_stats.bytes_freed = memory_stats.bytes_freed + size
        ffi.C.free(ptr)
        return
    end
    
    local pool_size = 16
    for _, ps in ipairs(pool_sizes) do
        if size <= ps then
            pool_size = ps
            break
        end
    end
    
    local pool = memory_pools[pool_size]
    if pool and pool.blocks_free < 1000 then -- Limit pool size to prevent memory bloat
        local block = ffi.cast("pool_block_t*", ffi.cast("char*", ptr) - ffi.offsetof("pool_block_t", "data"))
        block.next = pool.free_blocks
        pool.free_blocks = block
        pool.blocks_free = pool.blocks_free + 1
        memory_stats.pools_freed = memory_stats.pools_freed + 1
    else
        ffi.C.free(ffi.cast("char*", ptr) - ffi.offsetof("pool_block_t", "data"))
        memory_stats.bytes_freed = memory_stats.bytes_freed + pool_size
    end
end

-- Aligned memory allocation for SIMD operations
function concord.memory.aligned_alloc(size, alignment)
    alignment = alignment or 16
    local ptr = ffi.C.malloc(size + alignment - 1)
    if ptr == nil then return nil end
    
    local aligned = bit.band(ffi.cast("uintptr_t", ptr) + alignment - 1, bit.bnot(alignment - 1))
    return ffi.cast("void*", aligned)
end

-- Bulk memory operations
function concord.memory.bulk_copy(dest_array, src_array, count, element_size)
    local total_size = count * element_size
    return ffi.C.memcpy(dest_array, src_array, total_size)
end

function concord.memory.bulk_set(array, value, count, element_size)
    local total_size = count * element_size
    return ffi.C.memset(array, value, total_size)
end

function concord.memory.bulk_compare(array1, array2, count, element_size)
    local total_size = count * element_size
    return ffi.C.memcmp(array1, array2, total_size) == 0
end

-- Memory statistics
function concord.memory.get_stats()
    return {
        pools_allocated = memory_stats.pools_allocated,
        pools_freed = memory_stats.pools_freed,
        large_allocated = memory_stats.large_allocated,
        large_freed = memory_stats.large_freed,
        bytes_allocated = memory_stats.bytes_allocated,
        bytes_freed = memory_stats.bytes_freed,
        active_allocations = memory_stats.pools_allocated - memory_stats.pools_freed + memory_stats.large_allocated - memory_stats.large_freed,
        active_bytes = memory_stats.bytes_allocated - memory_stats.bytes_freed
    }
end

-- Initialize memory subsystem
concord.memory.init_pools()

-- ============================================================================
-- ULTRA-OPTIMIZED VECTOR MATH
-- ============================================================================

concord.vector = {}

-- Vector3 with complete mathematical operations
local vec3_mt = {}
vec3_mt.__index = vec3_mt

function concord.vector.new(x, y, z)
    local vec = ffi.new("vec3d_t")
    vec.x = x or 0
    vec.y = y or 0
    vec.z = z or 0
    return ffi.metatype("vec3d_t", vec3_mt)(vec)
end

function concord.vector.new_f(x, y, z)
    local vec = ffi.new("vec3f_t")
    vec.x = x or 0
    vec.y = y or 0
    vec.z = z or 0
    return vec
end

-- Ultra-fast vector operations
function vec3_mt:dot(other)
    return self.x * other.x + self.y * other.y + self.z * other.z
end

function vec3_mt:cross(other)
    local result = ffi.new("vec3d_t")
    result.x = self.y * other.z - self.z * other.y
    result.y = self.z * other.x - self.x * other.z
    result.z = self.x * other.y - self.y * other.x
    return result
end

function vec3_mt:length()
    return ffi.C.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
end

function vec3_mt:length_squared()
    return self.x * self.x + self.y * self.y + self.z * self.z
end

function vec3_mt:length_fast()
    return ffi.C.sqrtf(self.x * self.x + self.y * self.y + self.z * self.z)
end

function vec3_mt:normalize()
    local len_sq = self.x * self.x + self.y * self.y + self.z * self.z
    if len_sq > 0 then
        local inv_len = 1.0 / ffi.C.sqrt(len_sq)
        self.x = self.x * inv_len
        self.y = self.y * inv_len
        self.z = self.z * inv_len
    end
    return self
end

function vec3_mt:normalize_fast()
    local len_sq = self.x * self.x + self.y * self.y + self.z * self.z
    if len_sq > 0 then
        -- Fast inverse square root approximation
        local half_len_sq = len_sq * 0.5
        local i = ffi.cast("uint32_t*", ffi.new("float[1]", len_sq))[0]
        i = 0x5f3759df - bit.rshift(i, 1)
        local inv_len = ffi.cast("float*", ffi.new("uint32_t[1]", i))[0]
        inv_len = inv_len * (1.5 - half_len_sq * inv_len * inv_len)
        
        self.x = self.x * inv_len
        self.y = self.y * inv_len
        self.z = self.z * inv_len
    end
    return self
end

function vec3_mt:distance_to(other)
    local dx = self.x - other.x
    local dy = self.y - other.y
    local dz = self.z - other.z
    return ffi.C.sqrt(dx * dx + dy * dy + dz * dz)
end

function vec3_mt:distance_to_squared(other)
    local dx = self.x - other.x
    local dy = self.y - other.y
    local dz = self.z - other.z
    return dx * dx + dy * dy + dz * dz
end

function vec3_mt:lerp(other, t)
    self.x = self.x + (other.x - self.x) * t
    self.y = self.y + (other.y - self.y) * t
    self.z = self.z + (other.z - self.z) * t
    return self
end

function vec3_mt:slerp(other, t)
    local dot = self:dot(other)
    dot = math.max(-1, math.min(1, dot))
    local theta = ffi.C.acos(dot) * t
    
    local relative = ffi.new("vec3d_t")
    relative.x = other.x - self.x * dot
    relative.y = other.y - self.y * dot
    relative.z = other.z - self.z * dot
    
    local cos_theta = ffi.C.cos(theta)
    local sin_theta = ffi.C.sin(theta)
    
    self.x = self.x * cos_theta + relative.x * sin_theta
    self.y = self.y * cos_theta + relative.y * sin_theta
    self.z = self.z * cos_theta + relative.z * sin_theta
    
    return self
end

function vec3_mt:reflect(normal)
    local dot = self:dot(normal)
    self.x = self.x - 2 * dot * normal.x
    self.y = self.y - 2 * dot * normal.y
    self.z = self.z - 2 * dot * normal.z
    return self
end

function vec3_mt:project(onto)
    local dot = self:dot(onto)
    local len_sq = onto:length_squared()
    if len_sq > 0 then
        local scale = dot / len_sq
        self.x = onto.x * scale
        self.y = onto.y * scale
        self.z = onto.z * scale
    end
    return self
end

-- SIMD-like batch operations
function concord.vector.batch_add(vectors, other)
    for i = 1, #vectors do
        local v = vectors[i]
        v.x = v.x + other.x
        v.y = v.y + other.y
        v.z = v.z + other.z
    end
end

function concord.vector.batch_normalize(vectors)
    for i = 1, #vectors do
        vectors[i]:normalize_fast()
    end
end

function concord.vector.batch_scale(vectors, scale)
    for i = 1, #vectors do
        local v = vectors[i]
        v.x = v.x * scale
        v.y = v.y * scale
        v.z = v.z * scale
    end
end

-- Matrix operations
concord.matrix = {}

function concord.matrix.new()
    local mat = ffi.new("mat4d_t")
    -- Initialize as identity matrix
    ffi.C.memset(mat.m, 0, 16 * ffi.sizeof("double"))
    mat.m[0] = 1; mat.m[5] = 1; mat.m[10] = 1; mat.m[15] = 1
    return mat
end

function concord.matrix.multiply(a, b)
    local result = ffi.new("mat4d_t")
    for i = 0, 3 do
        for j = 0, 3 do
            result.m[i * 4 + j] = a.m[i * 4] * b.m[j] + 
                                   a.m[i * 4 + 1] * b.m[4 + j] + 
                                   a.m[i * 4 + 2] * b.m[8 + j] + 
                                   a.m[i * 4 + 3] * b.m[12 + j]
        end
    end
    return result
end

function concord.matrix.translate(mat, x, y, z)
    mat.m[12] = mat.m[12] + x
    mat.m[13] = mat.m[13] + y
    mat.m[14] = mat.m[14] + z
    return mat
end

function concord.matrix.scale(mat, x, y, z)
    for i = 0, 3 do
        mat.m[i] = mat.m[i] * x
        mat.m[4 + i] = mat.m[4 + i] * y
        mat.m[8 + i] = mat.m[8 + i] * z
    end
    return mat
end

-- Quaternion operations
concord.quaternion = {}

function concord.quaternion.new(x, y, z, w)
    local quat = ffi.new("quat_t")
    quat.x = x or 0
    quat.y = y or 0
    quat.z = z or 0
    quat.w = w or 1
    return quat
end

function concord.quaternion.multiply(a, b)
    local result = ffi.new("quat_t")
    result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
    result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x
    result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    return result
end

-- ============================================================================
-- ULTRA-FAST HASH TABLES AND DATA STRUCTURES
-- ============================================================================

concord.hash = {}

local hash_mt = {}
hash_mt.__index = hash_mt

function concord.hash.new(initial_size)
    initial_size = initial_size or 32
    local table = ffi.new("hash_table_t")
    table.buckets = ffi.cast("hash_node_t**", ffi.C.calloc(initial_size, ffi.sizeof("hash_node_t*")))
    table.bucket_count = initial_size
    table.item_count = 0
    table.load_factor = 0.75
    
    return setmetatable({_table = table}, hash_mt)
end

-- Ultra-fast hash functions
local function hash_bytes(data, len)
    local hash = 14695981039346656037ULL
    local prime = 1099511628211ULL
    
    for i = 0, len - 1 do
        hash = bit.bxor(hash, data[i])
        hash = hash * prime
    end
    
    return hash
end

local function hash_string_fast(str)
    local len = #str
    local data = ffi.cast("const uint8_t*", str)
    return hash_bytes(data, len)
end

local function hash_pointer(ptr)
    local addr = ffi.cast("uintptr_t", ptr)
    addr = bit.bxor(addr, bit.rshift(addr, 16))
    addr = addr * 0x85ebca6b
    addr = bit.bxor(addr, bit.rshift(addr, 13))
    addr = addr * 0xc2b2ae35
    addr = bit.bxor(addr, bit.rshift(addr, 16))
    return addr
end

function hash_mt:set(key, value)
    local key_str = tostring(key)
    local hash = hash_string_fast(key_str)
    local index = hash % self._table.bucket_count
    
    local node = self._table.buckets[index]
    
    -- Search for existing key
    while node ~= nil do
        if node.hash == hash then
            node.value = value
            return
        end
        node = node.next
    end
    
    -- Create new node
    local new_node = ffi.cast("hash_node_t*", ffi.C.malloc(ffi.sizeof("hash_node_t")))
    new_node.hash = hash
    new_node.key = key
    new_node.value = value
    new_node.next = self._table.buckets[index]
    self._table.buckets[index] = new_node
    
    self._table.item_count = self._table.item_count + 1
    
    -- Resize if needed
    if self._table.item_count > self._table.bucket_count * self._table.load_factor then
        self:resize(self._table.bucket_count * 2)
    end
end

function hash_mt:get(key)
    local key_str = tostring(key)
    local hash = hash_string_fast(key_str)
    local index = hash % self._table.bucket_count
    local node = self._table.buckets[index]
    
    while node ~= nil do
        if node.hash == hash then
            return node.value
        end
        node = node.next
    end
    
    return nil
end

function hash_mt:remove(key)
    local key_str = tostring(key)
    local hash = hash_string_fast(key_str)
    local index = hash % self._table.bucket_count
    local node = self._table.buckets[index]
    local prev = nil
    
    while node ~= nil do
        if node.hash == hash then
            if prev then
                prev.next = node.next
            else
                self._table.buckets[index] = node.next
            end
            ffi.C.free(node)
            self._table.item_count = self._table.item_count - 1
            return true
        end
        prev = node
        node = node.next
    end
    
    return false
end

function hash_mt:resize(new_size)
    local old_buckets = self._table.buckets
    local old_size = self._table.bucket_count
    
    self._table.buckets = ffi.cast("hash_node_t**", ffi.C.calloc(new_size, ffi.sizeof("hash_node_t*")))
    self._table.bucket_count = new_size
    local old_count = self._table.item_count
    self._table.item_count = 0
    
    -- Rehash all nodes
    for i = 0, old_size - 1 do
        local node = old_buckets[i]
        while node ~= nil do
            local next_node = node.next
            local new_index = node.hash % new_size
            node.next = self._table.buckets[new_index]
            self._table.buckets[new_index] = node
            self._table.item_count = self._table.item_count + 1
            node = next_node
        end
    end
    
    ffi.C.free(old_buckets)
end

-- Dynamic arrays with automatic resizing
concord.array = {}

local array_mt = {}
array_mt.__index = array_mt

function concord.array.new(initial_capacity)
    initial_capacity = initial_capacity or 16
    local arr = ffi.new("dynamic_array_t")
    arr.data = ffi.cast("void**", ffi.C.malloc(initial_capacity * ffi.sizeof("void*")))
    arr.size = 0
    arr.capacity = initial_capacity
    
    return setmetatable({_array = arr}, array_mt)
end

function array_mt:push(item)
    if self._array.size >= self._array.capacity then
        local new_capacity = self._array.capacity * 2
        self._array.data = ffi.cast("void**", ffi.C.realloc(self._array.data, new_capacity * ffi.sizeof("void*")))
        self._array.capacity = new_capacity
    end
    
    self._array.data[self._array.size] = ffi.cast("void*", item)
    self._array.size = self._array.size + 1
end

function array_mt:get(index)
    if index < 0 or index >= self._array.size then
        return nil
    end
    return self._array.data[index]
end

function array_mt:set(index, item)
    if index < 0 or index >= self._array.size then
        return false
    end
    self._array.data[index] = ffi.cast("void*", item)
    return true
end

function array_mt:pop()
    if self._array.size == 0 then
        return nil
    end
    self._array.size = self._array.size - 1
    return self._array.data[self._array.size]
end

function array_mt:size()
    return self._array.size
end

function array_mt:clear()
    self._array.size = 0
end

-- ============================================================================
-- SPATIAL PARTITIONING AND COLLISION DETECTION
-- ============================================================================

concord.spatial = {}

-- Octree implementation for 3D spatial partitioning
local octree_mt = {}
octree_mt.__index = octree_mt

function concord.spatial.octree_new(bounds, max_objects, max_depth)
    max_objects = max_objects or 10
    max_depth = max_depth or 5
    
    local node = ffi.new("octree_node_t")
    node.bounds = bounds
    node.objects = ffi.cast("void**", ffi.C.malloc(max_objects * ffi.sizeof("void*")))
    node.object_count = 0
    node.is_leaf = true
    
    for i = 0, 7 do
        node.children[i] = nil
    end
    
    return setmetatable({
        _node = node,
        max_objects = max_objects,
        max_depth = max_depth,
        current_depth = 0
    }, octree_mt)
end

function octree_mt:insert(object, position)
    -- Check if object fits in this node
    if not self:contains_point(position) then
        return false
    end
    
    -- If we're a leaf and have space, add the object
    if self._node.is_leaf and self._node.object_count < self.max_objects then
        self._node.objects[self._node.object_count] = ffi.cast("void*", object)
        self._node.object_count = self._node.object_count + 1
        return true
    end
    
    -- If we're a leaf but need to subdivide
    if self._node.is_leaf then
        self:subdivide()
    end
    
    -- Try to insert into children
    for i = 0, 7 do
        if self._node.children[i] ~= nil then
            if self._node.children[i]:insert(object, position) then
                return true
            end
        end
    end
    
    return false
end

function octree_mt:subdivide()
    if self.current_depth >= self.max_depth then
        return
    end
    
    local bounds = self._node.bounds
    local center = ffi.new("vec3d_t")
    center.x = (bounds.min.x + bounds.max.x) * 0.5
    center.y = (bounds.min.y + bounds.max.y) * 0.5
    center.z = (bounds.min.z + bounds.max.z) * 0.5
    
    -- Create 8 child nodes
    for i = 0, 7 do
        local child_bounds = ffi.new("aabb_t")
        
        -- Calculate child bounds based on octant
        if bit.band(i, 1) == 0 then
            child_bounds.min.x = bounds.min.x
            child_bounds.max.x = center.x
        else
            child_bounds.min.x = center.x
            child_bounds.max.x = bounds.max.x
        end
        
        if bit.band(i, 2) == 0 then
            child_bounds.min.y = bounds.min.y
            child_bounds.max.y = center.y
        else
            child_bounds.min.y = center.y
            child_bounds.max.y = bounds.max.y
        end
        
        if bit.band(i, 4) == 0 then
            child_bounds.min.z = bounds.min.z
            child_bounds.max.z = center.z
        else
            child_bounds.min.z = center.z
            child_bounds.max.z = bounds.max.z
        end
        
        local child = concord.spatial.octree_new(child_bounds, self.max_objects, self.max_depth)
        child.current_depth = self.current_depth + 1
        self._node.children[i] = child._node
    end
    
    self._node.is_leaf = false
end

function octree_mt:contains_point(point)
    local bounds = self._node.bounds
    return point.x >= bounds.min.x and point.x <= bounds.max.x and
           point.y >= bounds.min.y and point.y <= bounds.max.y and
           point.z >= bounds.min.z and point.z <= bounds.max.z
end

function octree_mt:query_range(range_bounds, results)
    results = results or {}
    
    -- Check if this node intersects with the query range
    if not self:aabb_intersects(self._node.bounds, range_bounds) then
        return results
    end
    
    -- If we're a leaf, check all objects
    if self._node.is_leaf then
        for i = 0, self._node.object_count - 1 do
            table.insert(results, self._node.objects[i])
        end
    else
        -- Recursively query children
        for i = 0, 7 do
            if self._node.children[i] ~= nil then
                local child = setmetatable({_node = self._node.children[i]}, octree_mt)
                child:query_range(range_bounds, results)
            end
        end
    end
    
    return results
end

function octree_mt:aabb_intersects(a, b)
    return not (a.max.x < b.min.x or a.min.x > b.max.x or
                a.max.y < b.min.y or a.min.y > b.max.y or
                a.max.z < b.min.z or a.min.z > b.max.z)
end

-- Fast collision detection utilities
concord.collision = {}

-- Ray-sphere intersection
function concord.collision.ray_sphere_intersect(ray, sphere)
    local oc = ffi.new("vec3d_t")
    oc.x = ray.origin.x - sphere.center.x
    oc.y = ray.origin.y - sphere.center.y
    oc.z = ray.origin.z - sphere.center.z
    
    local a = ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z
    local b = 2.0 * (oc.x * ray.direction.x + oc.y * ray.direction.y + oc.z * ray.direction.z)
    local c = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - sphere.radius * sphere.radius
    
    local discriminant = b * b - 4 * a * c
    if discriminant < 0 then
        return false, 0
    end
    
    local sqrt_discriminant = ffi.C.sqrt(discriminant)
    local t1 = (-b - sqrt_discriminant) / (2 * a)
    local t2 = (-b + sqrt_discriminant) / (2 * a)
    
    local t = (t1 >= 0) and t1 or t2
    return t >= 0, t
end

-- Ray-AABB intersection
function concord.collision.ray_aabb_intersect(ray, aabb)
    local tmin = (aabb.min.x - ray.origin.x) / ray.direction.x
    local tmax = (aabb.max.x - ray.origin.x) / ray.direction.x
    
    if tmin > tmax then
        tmin, tmax = tmax, tmin
    end
    
    local tymin = (aabb.min.y - ray.origin.y) / ray.direction.y
    local tymax = (aabb.max.y - ray.origin.y) / ray.direction.y
    
    if tymin > tymax then
        tymin, tymax = tymax, tymin
    end
    
    if tmin > tymax or tymin > tmax then
        return false, 0
    end
    
    tmin = math.max(tmin, tymin)
    tmax = math.min(tmax, tymax)
    
    local tzmin = (aabb.min.z - ray.origin.z) / ray.direction.z
    local tzmax = (aabb.max.z - ray.origin.z) / ray.direction.z
    
    if tzmin > tzmax then
        tzmin, tzmax = tzmax, tzmin
    end
    
    if tmin > tzmax or tzmin > tmax then
        return false, 0
    end
    
    tmin = math.max(tmin, tzmin)
    return tmin >= 0, tmin
end

-- Sphere-sphere intersection
function concord.collision.sphere_sphere_intersect(sphere1, sphere2)
    local dx = sphere1.center.x - sphere2.center.x
    local dy = sphere1.center.y - sphere2.center.y
    local dz = sphere1.center.z - sphere2.center.z
    local distance_sq = dx * dx + dy * dy + dz * dz
    local radius_sum = sphere1.radius + sphere2.radius
    return distance_sq <= radius_sum * radius_sum
end

-- ============================================================================
-- ULTRA-FAST STRING OPERATIONS
-- ============================================================================

concord.string = {}

-- Ultra-optimized string builder
local string_builder_mt = {}
string_builder_mt.__index = string_builder_mt

function concord.string.new_builder(initial_capacity)
    initial_capacity = initial_capacity or 4096
    local builder = ffi.new("string_builder_t")
    builder.buffer = ffi.cast("char*", ffi.C.malloc(initial_capacity))
    builder.length = 0
    builder.capacity = initial_capacity
    
    return setmetatable({_builder = builder}, string_builder_mt)
end

function string_builder_mt:append(str)
    if type(str) ~= "string" then
        str = tostring(str)
    end
    
    local str_len = #str
    if self._builder.length + str_len >= self._builder.capacity then
        local new_capacity = math.max(self._builder.capacity * 2, self._builder.length + str_len + 1)
        self._builder.buffer = ffi.cast("char*", ffi.C.realloc(self._builder.buffer, new_capacity))
        self._builder.capacity = new_capacity
    end
    
    ffi.copy(self._builder.buffer + self._builder.length, str, str_len)
    self._builder.length = self._builder.length + str_len
    return self
end

function string_builder_mt:append_char(char)
    if self._builder.length + 1 >= self._builder.capacity then
        local new_capacity = self._builder.capacity * 2
        self._builder.buffer = ffi.cast("char*", ffi.C.realloc(self._builder.buffer, new_capacity))
        self._builder.capacity = new_capacity
    end
    
    self._builder.buffer[self._builder.length] = char
    self._builder.length = self._builder.length + 1
    return self
end

function string_builder_mt:append_number(num)
    local str = tostring(num)
    return self:append(str)
end

function string_builder_mt:to_string()
    self._builder.buffer[self._builder.length] = 0 -- Null terminate
    return ffi.string(self._builder.buffer, self._builder.length)
end

function string_builder_mt:clear()
    self._builder.length = 0
    return self
end

function string_builder_mt:length()
    return self._builder.length
end

-- Ultra-fast string search using Boyer-Moore
function concord.string.boyer_moore_search(text, pattern)
    local text_len = #text
    local pattern_len = #pattern
    
    if pattern_len == 0 then return 1 end
    if pattern_len > text_len then return nil end
    
    -- Build bad character table
    local bad_char = {}
    for i = 0, 255 do
        bad_char[i] = pattern_len
    end
    
    for i = 1, pattern_len - 1 do
        bad_char[string.byte(pattern, i)] = pattern_len - i
    end
    
    -- Search
    local skip = 0
    while skip <= text_len - pattern_len do
        local j = pattern_len
        while j >= 1 and string.byte(pattern, j) == string.byte(text, skip + j) do
            j = j - 1
        end
        
        if j < 1 then
            return skip + 1
        else
            skip = skip + math.max(1, bad_char[string.byte(text, skip + j)] - pattern_len + j)
        end
    end
    
    return nil
end

-- Fast string hashing for caching
function concord.string.hash(str)
    return hash_string_fast(str)
end

-- String interning for memory efficiency
local string_intern_table = concord.hash.new(1024)

function concord.string.intern(str)
    local existing = string_intern_table:get(str)
    if existing then
        return existing
    else
        string_intern_table:set(str, str)
        return str
    end
end

-- ============================================================================
-- ADVANCED PERFORMANCE MONITORING
-- ============================================================================

concord.perf = {}

-- High-precision timing
local perf_frequency = 1000000000 -- Nanosecond precision
local start_time = ffi.C.clock()

function concord.perf.get_time_ns()
    return (ffi.C.clock() - start_time) * perf_frequency / ffi.C.CLOCKS_PER_SEC
end

function concord.perf.get_time_us()
    return concord.perf.get_time_ns() / 1000
end

function concord.perf.get_time_ms()
    return concord.perf.get_time_ns() / 1000000
end

-- Advanced profiler with call stack tracking
function concord.perf.profile_begin(name)
    local counter = perf_counters[name]
    if not counter then
        counter = ffi.new("perf_counter_t")
        counter.calls = 0
        counter.total_time = 0
        counter.min_time = math.huge
        counter.max_time = 0
        counter.avg_time = 0
        perf_counters[name] = counter
    end
    
    table.insert(profile_stack, {
        name = name,
        start_time = concord.perf.get_time_ns(),
        counter = counter
    })
end

function concord.perf.profile_end()
    if #profile_stack == 0 then return end
    
    local entry = table.remove(profile_stack)
    local elapsed = concord.perf.get_time_ns() - entry.start_time
    local elapsed_ms = elapsed / 1000000
    
    local counter = entry.counter
    counter.calls = counter.calls + 1
    counter.total_time = counter.total_time + elapsed_ms
    counter.min_time = math.min(counter.min_time, elapsed_ms)
    counter.max_time = math.max(counter.max_time, elapsed_ms)
    counter.avg_time = counter.total_time / counter.calls
end

-- Automatic profiling decorator
function concord.perf.profile_func(name, func)
    return function(...)
        concord.perf.profile_begin(name)
        local results = {func(...)}
        concord.perf.profile_end()
        return unpack(results)
    end
end

-- Memory profiling
local memory_snapshots = {}

function concord.perf.memory_snapshot(name)
    local stats = concord.memory.get_stats()
    memory_snapshots[name] = {
        timestamp = concord.perf.get_time_ms(),
        stats = stats
    }
end

function concord.perf.memory_diff(snapshot1, snapshot2)
    local s1 = memory_snapshots[snapshot1]
    local s2 = memory_snapshots[snapshot2]
    
    if not s1 or not s2 then return nil end
    
    return {
        time_diff = s2.timestamp - s1.timestamp,
        bytes_diff = s2.stats.active_bytes - s1.stats.active_bytes,
        allocs_diff = s2.stats.active_allocations - s1.stats.active_allocations
    }
end

-- Performance statistics
function concord.perf.get_stats()
    local stats = {}
    for name, counter in pairs(perf_counters) do
        stats[name] = {
            calls = tonumber(counter.calls),
            total_time = counter.total_time,
            min_time = counter.min_time,
            max_time = counter.max_time,
            avg_time = counter.avg_time
        }
    end
    return stats
end

function concord.perf.clear_stats()
    perf_counters = {}
    profile_stack = {}
    memory_snapshots = {}
end

-- ============================================================================
-- ADVANCED MATH UTILITIES
-- ============================================================================

concord.math = {}

-- Mathematical constants
concord.math.PI = 3.1415926535897932384626433832795
concord.math.TAU = 6.2831853071795864769252867665590
concord.math.E = 2.7182818284590452353602874713527
concord.math.PHI = 1.6180339887498948482045868343656
concord.math.SQRT2 = 1.4142135623730950488016887242097

-- Precomputed lookup tables for ultra-fast trigonometry
local TRIG_TABLE_SIZE = 4096
local TRIG_TABLE_MASK = TRIG_TABLE_SIZE - 1
local TRIG_TABLE_SCALE = TRIG_TABLE_SIZE / concord.math.TAU

local sin_table = ffi.new("float[?]", TRIG_TABLE_SIZE)
local cos_table = ffi.new("float[?]", TRIG_TABLE_SIZE)
local tan_table = ffi.new("float[?]", TRIG_TABLE_SIZE)

-- Initialize lookup tables
for i = 0, TRIG_TABLE_SIZE - 1 do
    local angle = i * concord.math.TAU / TRIG_TABLE_SIZE
    sin_table[i] = ffi.C.sinf(angle)
    cos_table[i] = ffi.C.cosf(angle)
    tan_table[i] = ffi.C.tanf(angle)
end

function concord.math.fast_sin(x)
    local index = bit.band(math.floor(x * TRIG_TABLE_SCALE + 0.5), TRIG_TABLE_MASK)
    return sin_table[index]
end

function concord.math.fast_cos(x)
    local index = bit.band(math.floor(x * TRIG_TABLE_SCALE + 0.5), TRIG_TABLE_MASK)
    return cos_table[index]
end

function concord.math.fast_tan(x)
    local index = bit.band(math.floor(x * TRIG_TABLE_SCALE + 0.5), TRIG_TABLE_MASK)
    return tan_table[index]
end

-- Ultra-fast inverse square root (Quake III algorithm)
function concord.math.fast_inv_sqrt(x)
    if x <= 0 then return 0 end
    
    local half_x = x * 0.5
    local i = ffi.cast("uint32_t*", ffi.new("float[1]", x))[0]
    i = 0x5f3759df - bit.rshift(i, 1)
    local y = ffi.cast("float*", ffi.new("uint32_t[1]", i))[0]
    
    -- Newton-Raphson iteration for higher precision
    y = y * (1.5 - half_x * y * y)
    y = y * (1.5 - half_x * y * y) -- Second iteration for even better precision
    
    return y
end

-- Fast square root using inverse square root
function concord.math.fast_sqrt(x)
    if x <= 0 then return 0 end
    return x * concord.math.fast_inv_sqrt(x)
end

-- Fast integer power function using binary exponentiation
function concord.math.fast_pow_int(base, exp)
    if exp == 0 then return 1 end
    if exp == 1 then return base end
    if exp < 0 then return 1.0 / concord.math.fast_pow_int(base, -exp) end
    
    local result = 1
    while exp > 0 do
        if bit.band(exp, 1) == 1 then
            result = result * base
        end
        base = base * base
        exp = bit.rshift(exp, 1)
    end
    return result
end

-- Fast floating point power approximation
function concord.math.fast_pow(base, exp)
    if base <= 0 then return 0 end
    if exp == 0 then return 1 end
    if exp == 1 then return base end
    
    -- Use exp(exp * log(base)) approximation
    local log_base = ffi.C.log(base)
    return ffi.C.exp(exp * log_base)
end

-- Interpolation functions
function concord.math.lerp(a, b, t)
    return a + (b - a) * t
end

function concord.math.smooth_step(edge0, edge1, x)
    local t = concord.math.clamp((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)
end

function concord.math.smoother_step(edge0, edge1, x)
    local t = concord.math.clamp((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * t * (t * (t * 6 - 15) + 10)
end

-- Utility functions
function concord.math.clamp(value, min_val, max_val)
    if value < min_val then return min_val end
    if value > max_val then return max_val end
    return value
end

function concord.math.sign(x)
    if x > 0 then return 1 end
    if x < 0 then return -1 end
    return 0
end

function concord.math.wrap(value, min_val, max_val)
    local range = max_val - min_val
    if range <= 0 then return min_val end
    return min_val + ffi.C.fmod(value - min_val, range)
end

function concord.math.remap(value, old_min, old_max, new_min, new_max)
    local old_range = old_max - old_min
    if old_range == 0 then return new_min end
    local new_range = new_max - new_min
    return new_min + ((value - old_min) * new_range) / old_range
end

-- Fast random number generation using xorshift
local rng_state = ffi.new("uint32_t[1]", 1)

function concord.math.seed_random(seed)
    rng_state[0] = seed or os.time()
end

function concord.math.fast_random()
    local state = rng_state[0]
    state = bit.bxor(state, bit.lshift(state, 13))
    state = bit.bxor(state, bit.rshift(state, 17))
    state = bit.bxor(state, bit.lshift(state, 5))
    rng_state[0] = state
    return state / 4294967296.0
end

function concord.math.fast_random_range(min_val, max_val)
    return min_val + concord.math.fast_random() * (max_val - min_val)
end

function concord.math.fast_random_int(min_val, max_val)
    return math.floor(concord.math.fast_random_range(min_val, max_val + 1))
end

-- Noise functions
function concord.math.white_noise(x, y)
    local n = bit.bxor(bit.lshift(x, 16), y)
    n = bit.bxor(n, bit.lshift(n, 13))
    n = bit.band(n * (n * n * 15731 + 789221) + 1376312589, 0x7fffffff)
    return n / 1073741824.0
end

-- ============================================================================
-- GARRY'S MOD SPECIFIC OPTIMIZATIONS
-- ============================================================================

concord.gmod = {}

-- Ultra-fast entity operations
function concord.gmod.fast_get_players()
    local players = concord.array.new(game.MaxPlayers())
    
    for i = 1, game.MaxPlayers() do
        local ply = Entity(i)
        if IsValid(ply) and ply:IsPlayer() then
            players:push(ply)
        end
    end
    
    return players
end

function concord.gmod.fast_get_entities_in_sphere(center, radius)
    local entities = {}
    local radius_sq = radius * radius
    local count = 0
    
    for _, ent in ipairs(ents.GetAll()) do
        if IsValid(ent) then
            local pos = ent:GetPos()
            local dx = pos.x - center.x
            local dy = pos.y - center.y
            local dz = pos.z - center.z
            local dist_sq = dx * dx + dy * dy + dz * dz
            
            if dist_sq <= radius_sq then
                count = count + 1
                entities[count] = ent
            end
        end
    end
    
    return entities, count
end

-- Batch entity operations
function concord.gmod.batch_set_position(entities, positions)
    local count = math.min(#entities, #positions)
    for i = 1, count do
        if IsValid(entities[i]) then
            entities[i]:SetPos(positions[i])
        end
    end
end

function concord.gmod.batch_set_angles(entities, angles)
    local count = math.min(#entities, #angles)
    for i = 1, count do
        if IsValid(entities[i]) then
            entities[i]:SetAngles(angles[i])
        end
    end
end

function concord.gmod.batch_set_color(entities, colors)
    local count = math.min(#entities, #colors)
    for i = 1, count do
        if IsValid(entities[i]) then
            entities[i]:SetColor(colors[i])
        end
    end
end

-- Entity pooling system
local entity_pools = {}
local pool_limits = {}

function concord.gmod.create_entity_pool(class_name, initial_size, max_size)
    initial_size = initial_size or 10
    max_size = max_size or 100
    
    entity_pools[class_name] = {}
    pool_limits[class_name] = max_size
    
    -- Pre-allocate entities
    for i = 1, initial_size do
        local ent = ents.Create(class_name)
        if IsValid(ent) then
            ent:Spawn()
            ent:SetNoDraw(true)
            ent:SetNotSolid(true)
            ent:SetPos(Vector(0, 0, -10000))
            table.insert(entity_pools[class_name], ent)
        end
    end
end

function concord.gmod.get_pooled_entity(class_name)
    local pool = entity_pools[class_name]
    if not pool then
        concord.gmod.create_entity_pool(class_name)
        pool = entity_pools[class_name]
    end
    
    if #pool > 0 then
        local ent = table.remove(pool)
        ent:SetNoDraw(false)
        ent:SetNotSolid(false)
        return ent
    else
        local ent = ents.Create(class_name)
        if IsValid(ent) then
            ent:Spawn()
        end
        return ent
    end
end

function concord.gmod.return_pooled_entity(ent, class_name)
    if not IsValid(ent) then return end
    
    local pool = entity_pools[class_name]
    if not pool then
        ent:Remove()
        return
    end
    
    local limit = pool_limits[class_name] or 100
    if #pool < limit then
        ent:SetNoDraw(true)
        ent:SetNotSolid(true)
        ent:SetPos(Vector(0, 0, -10000))
        ent:SetAngles(Angle(0, 0, 0))
        ent:SetColor(Color(255, 255, 255, 255))
        table.insert(pool, ent)
    else
        ent:Remove()
    end
end

-- Fast trace operations
function concord.gmod.fast_trace_line(start_pos, end_pos, filter)
    local trace_data = {
        start = start_pos,
        endpos = end_pos,
        filter = filter or function() return true end
    }
    
    return util.TraceLine(trace_data)
end

function concord.gmod.batch_trace_lines(traces)
    local results = {}
    for i, trace_data in ipairs(traces) do
        results[i] = util.TraceLine(trace_data)
    end
    return results
end

-- Optimized distance calculations
function concord.gmod.fast_distance_2d(pos1, pos2)
    local dx = pos1.x - pos2.x
    local dy = pos1.y - pos2.y
    return ffi.C.sqrt(dx * dx + dy * dy)
end

function concord.gmod.fast_distance_3d(pos1, pos2)
    local dx = pos1.x - pos2.x
    local dy = pos1.y - pos2.y
    local dz = pos1.z - pos2.z
    return ffi.C.sqrt(dx * dx + dy * dy + dz * dz)
end

function concord.gmod.fast_distance_squared(pos1, pos2)
    local dx = pos1.x - pos2.x
    local dy = pos1.y - pos2.y
    local dz = pos1.z - pos2.z
    return dx * dx + dy * dy + dz * dz
end

-- ============================================================================
-- ADVANCED CACHING SYSTEM
-- ============================================================================

concord.cache = {}

local cache_data = concord.hash.new(512)
local cache_timestamps = concord.hash.new(512)
local cache_stats = {
    hits = 0,
    misses = 0,
    evictions = 0,
    sets = 0
}

local DEFAULT_TTL = 60 -- seconds
local MAX_CACHE_SIZE = 10000

function concord.cache.set(key, value, ttl)
    ttl = ttl or DEFAULT_TTL
    local current_time = CurTime()
    
    -- Check if we need to evict old entries
    if cache_data._table.item_count >= MAX_CACHE_SIZE then
        concord.cache.cleanup()
    end
    
    cache_data:set(key, value)
    cache_timestamps:set(key, current_time + ttl)
    cache_stats.sets = cache_stats.sets + 1
end

function concord.cache.get(key)
    local timestamp = cache_timestamps:get(key)
    if not timestamp then
        cache_stats.misses = cache_stats.misses + 1
        return nil
    end
    
    if CurTime() > timestamp then
        cache_data:remove(key)
        cache_timestamps:remove(key)
        cache_stats.misses = cache_stats.misses + 1
        cache_stats.evictions = cache_stats.evictions + 1
        return nil
    end
    
    cache_stats.hits = cache_stats.hits + 1
    return cache_data:get(key)
end

function concord.cache.has(key)
    local timestamp = cache_timestamps:get(key)
    if not timestamp then return false end
    
    if CurTime() > timestamp then
        cache_data:remove(key)
        cache_timestamps:remove(key)
        cache_stats.evictions = cache_stats.evictions + 1
        return false
    end
    
    return true
end

function concord.cache.remove(key)
    cache_data:remove(key)
    cache_timestamps:remove(key)
end

function concord.cache.clear()
    cache_data = concord.hash.new(512)
    cache_timestamps = concord.hash.new(512)
    cache_stats.evictions = cache_stats.evictions + cache_stats.sets - cache_stats.hits
end

function concord.cache.cleanup()
    local current_time = CurTime()
    local keys_to_remove = {}
    
    -- This is a simplified cleanup - in a real implementation you'd iterate through the hash table
    -- For now, we'll just clear everything if we hit the limit
    if cache_data._table.item_count >= MAX_CACHE_SIZE then
        concord.cache.clear()
    end
end

function concord.cache.get_stats()
    local total_requests = cache_stats.hits + cache_stats.misses
    return {
        hits = cache_stats.hits,
        misses = cache_stats.misses,
        evictions = cache_stats.evictions,
        sets = cache_stats.sets,
        hit_rate = total_requests > 0 and (cache_stats.hits / total_requests) or 0,
        size = cache_data._table.item_count
    }
end

-- Memoization decorator
function concord.cache.memoize(func, ttl)
    ttl = ttl or DEFAULT_TTL
    return function(...)
        local args = {...}
        local key = table.concat(args, "|")
        
        local cached = concord.cache.get(key)
        if cached ~= nil then
            return cached
        end
        
        local result = func(...)
        concord.cache.set(key, result, ttl)
        return result
    end
end

-- ============================================================================
-- NETWORKING OPTIMIZATIONS
-- ============================================================================

concord.net = {}

-- Ultra-fast compression using RLE with improvements
function concord.net.compress_rle(data)
    if type(data) ~= "string" then
        data = tostring(data)
    end
    
    local result = concord.string.new_builder(#data)
    local len = #data
    local i = 1
    
    while i <= len do
        local char = string.sub(data, i, i)
        local count = 1
        
        -- Count consecutive identical characters
        while i + count <= len and string.sub(data, i + count, i + count) == char and count < 255 do
            count = count + 1
        end
        
        -- Use compression if beneficial
        if count >= 4 or char == '\0' then
            result:append_char(0) -- Escape character
            result:append_char(count)
            result:append_char(string.byte(char))
        else
            for j = 1, count do
                if char == '\0' then
                    result:append_char(0)
                    result:append_char(0) -- Escaped zero
                else
                    result:append_char(string.byte(char))
                end
            end
        end
        
        i = i + count
    end
    
    return result:to_string()
end

function concord.net.decompress_rle(compressed)
    local result = concord.string.new_builder(#compressed * 2)
    local len = #compressed
    local i = 1
    
    while i <= len do
        local byte = string.byte(compressed, i)
        if byte == 0 and i + 1 <= len then
            local next_byte = string.byte(compressed, i + 1)
            if next_byte == 0 then
                -- Escaped zero
                result:append_char(0)
                i = i + 2
            elseif i + 2 <= len then
                -- RLE sequence
                local count = next_byte
                local char = string.byte(compressed, i + 2)
                for j = 1, count do
                    result:append_char(char)
                end
                i = i + 3
            else
                result:append_char(byte)
                i = i + 1
            end
        else
            result:append_char(byte)
            i = i + 1
        end
    end
    
    return result:to_string()
end

-- LZ77-style compression for better ratios
function concord.net.compress_lz77(data, window_size, lookahead_size)
    window_size = window_size or 4096
    lookahead_size = lookahead_size or 18
    
    if type(data) ~= "string" then
        data = tostring(data)
    end
    
    local result = concord.string.new_builder(#data)
    local pos = 1
    local data_len = #data
    
    while pos <= data_len do
        local best_length = 0
        local best_distance = 0
        
        -- Search for the longest match in the sliding window
        local search_start = math.max(1, pos - window_size)
        local search_end = pos - 1
        
        for i = search_start, search_end do
            local match_length = 0
            while match_length < lookahead_size and 
                  pos + match_length <= data_len and
                  i + match_length <= search_end and
                  string.sub(data, i + match_length, i + match_length) == 
                  string.sub(data, pos + match_length, pos + match_length) do
                match_length = match_length + 1
            end
            
            if match_length > best_length then
                best_length = match_length
                best_distance = pos - i
            end
        end
        
        if best_length >= 3 then
            -- Encode as (distance, length)
            result:append_char(0) -- Escape
            result:append_char(bit.band(best_distance, 0xFF))
            result:append_char(bit.rshift(best_distance, 8))
            result:append_char(best_length)
            pos = pos + best_length
        else
            -- Encode as literal
            local char = string.byte(data, pos)
            if char == 0 then
                result:append_char(0)
                result:append_char(0) -- Escaped zero
            else
                result:append_char(char)
            end
            pos = pos + 1
        end
    end
    
    return result:to_string()
end

-- Message batching for reduced network overhead
local message_batch = {}
local batch_timer = nil
local BATCH_INTERVAL = 0.01 -- 10ms batching

function concord.net.batch_message(recipient, message_type, data)
    if not message_batch[recipient] then
        message_batch[recipient] = {}
    end
    
    table.insert(message_batch[recipient], {
        type = message_type,
        data = data,
        timestamp = CurTime()
    })
    
    -- Set up batch timer if not already running
    if not batch_timer then
        batch_timer = timer.Simple(BATCH_INTERVAL, function()
            concord.net.flush_message_batch()
            batch_timer = nil
        end)
    end
end

function concord.net.flush_message_batch()
    for recipient, messages in pairs(message_batch) do
        if #messages > 0 then
            -- Send batched messages
            net.Start("concord_batch")
            net.WriteUInt(#messages, 16)
            
            for _, msg in ipairs(messages) do
                net.WriteString(msg.type)
                net.WriteString(msg.data)
            end
            
            if recipient == "broadcast" then
                net.Broadcast()
            else
                net.Send(recipient)
            end
        end
    end
    
    message_batch = {}
end

-- Delta compression for frequently updated data
local delta_states = {}

function concord.net.delta_compress(key, current_state)
    local previous_state = delta_states[key]
    delta_states[key] = current_state
    
    if not previous_state then
        return {full = true, data = current_state}
    end
    
    local delta = {}
    for k, v in pairs(current_state) do
        if previous_state[k] ~= v then
            delta[k] = v
        end
    end
    
    return {full = false, data = delta}
end

function concord.net.delta_apply(key, delta_data)
    if delta_data.full then
        delta_states[key] = delta_data.data
        return delta_data.data
    else
        local current_state = delta_states[key] or {}
        for k, v in pairs(delta_data.data) do
            current_state[k] = v
        end
        delta_states[key] = current_state
        return current_state
    end
end

-- ============================================================================
-- ASYNC TASK SYSTEM
-- ============================================================================

concord.async = {}

local coroutine_pool = {}
local active_coroutines = {}
local scheduled_tasks = concord.array.new(256)

-- Coroutine pooling for reduced GC pressure
function concord.async.create_task(func, ...)
    local args = {...}
    local co
    
    if #coroutine_pool > 0 then
        co = table.remove(coroutine_pool)
    else
        co = coroutine.create(function()
            while true do
                local task_func, task_args = coroutine.yield()
                if task_func then
                    task_func(unpack(task_args))
                end
            end
        end)
    end
    
    local success, err = coroutine.resume(co, func, args)
    if not success then
        print("Coroutine error:", err)
        table.insert(coroutine_pool, co)
        return nil
    end
    
    active_coroutines[co] = true
    return co
end

function concord.async.return_coroutine(co)
    active_coroutines[co] = nil
    if #coroutine_pool < 100 then -- Limit pool size
        table.insert(coroutine_pool, co)
    end
end

-- Task scheduling with priorities
local PRIORITY_HIGH = 1
local PRIORITY_NORMAL = 2
local PRIORITY_LOW = 3

function concord.async.schedule(func, delay, priority, ...)
    delay = delay or 0
    priority = priority or PRIORITY_NORMAL
    local args = {...}
    
    scheduled_tasks:push({
        func = func,
        args = args,
        time = CurTime() + delay,
        priority = priority
    })
end

function concord.async.schedule_next_frame(func, ...)
    concord.async.schedule(func, 0, PRIORITY_HIGH, ...)
end

function concord.async.process_scheduled()
    local current_time = CurTime()
    local tasks_to_run = {}
    
    -- Collect ready tasks
    local size = scheduled_tasks:size()
    for i = size - 1, 0, -1 do
        local task = scheduled_tasks:get(i)
        if task and current_time >= task.time then
            table.insert(tasks_to_run, task)
            -- Remove task (this is inefficient, but works for demonstration)
            -- In a real implementation, you'd use a better data structure
        end
    end
    
    -- Sort by priority
    table.sort(tasks_to_run, function(a, b)
        return a.priority < b.priority
    end)
    
    -- Execute tasks
    for _, task in ipairs(tasks_to_run) do
        local success, err = pcall(task.func, unpack(task.args))
        if not success then
            print("Scheduled task error:", err)
        end
    end
end

-- Parallel task execution
function concord.async.parallel(tasks, callback)
    local results = {}
    local completed = 0
    local total = #tasks
    
    for i, task in ipairs(tasks) do
        concord.async.create_task(function()
            local success, result = pcall(task)
            results[i] = {success = success, result = result}
            completed = completed + 1
            
            if completed == total and callback then
                callback(results)
            end
        end)
    end
end

-- ============================================================================
-- ADVANCED COLLECTIONS
-- ============================================================================

concord.collections = {}

-- Lock-free queue implementation
local queue_mt = {}
queue_mt.__index = queue_mt

function concord.collections.queue_new()
    local dummy = ffi.new("queue_node_t")
    dummy.data = nil
    dummy.next = nil
    
    return setmetatable({
        head = dummy,
        tail = dummy
    }, queue_mt)
end

function queue_mt:enqueue(item)
    local new_node = ffi.new("queue_node_t")
    new_node.data = ffi.cast("void*", item)
    new_node.next = nil
    
    self.tail.next = new_node
    self.tail = new_node
end

function queue_mt:dequeue()
    local head = self.head
    local new_head = head.next
    
    if new_head == nil then
        return nil -- Queue is empty
    end
    
    local data = new_head.data
    self.head = new_head
    return data
end

function queue_mt:is_empty()
    return self.head.next == nil
end

-- Priority queue using binary heap
local priority_queue_mt = {}
priority_queue_mt.__index = priority_queue_mt

function concord.collections.priority_queue_new(compare_func)
    return setmetatable({
        heap = {},
        size = 0,
        compare = compare_func or function(a, b) return a < b end
    }, priority_queue_mt)
end

function priority_queue_mt:parent(i)
    return math.floor(i / 2)
end

function priority_queue_mt:left_child(i)
    return 2 * i
end

function priority_queue_mt:right_child(i)
    return 2 * i + 1
end

function priority_queue_mt:swap(i, j)
    self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
end

function priority_queue_mt:heapify_up(i)
    while i > 1 do
        local parent = self:parent(i)
        if not self.compare(self.heap[i], self.heap[parent]) then
            break
        end
        self:swap(i, parent)
        i = parent
    end
end

function priority_queue_mt:heapify_down(i)
    while self:left_child(i) <= self.size do
        local min_child = self:left_child(i)
        local right = self:right_child(i)
        
        if right <= self.size and self.compare(self.heap[right], self.heap[min_child]) then
            min_child = right
        end
        
        if not self.compare(self.heap[min_child], self.heap[i]) then
            break
        end
        
        self:swap(i, min_child)
        i = min_child
    end
end

function priority_queue_mt:push(item)
    self.size = self.size + 1
    self.heap[self.size] = item
    self:heapify_up(self.size)
end

function priority_queue_mt:pop()
    if self.size == 0 then
        return nil
    end
    
    local root = self.heap[1]
    self.heap[1] = self.heap[self.size]
    self.size = self.size - 1
    
    if self.size > 0 then
        self:heapify_down(1)
    end
    
    return root
end

function priority_queue_mt:peek()
    return self.size > 0 and self.heap[1] or nil
end

function priority_queue_mt:is_empty()
    return self.size == 0
end

-- Bloom filter for fast set membership testing
local bloom_filter_mt = {}
bloom_filter_mt.__index = bloom_filter_mt

function concord.collections.bloom_filter_new(size, num_hashes)
    size = size or 1024
    num_hashes = num_hashes or 3
    
    return setmetatable({
        bits = ffi.new("uint8_t[?]", math.ceil(size / 8)),
        size = size,
        num_hashes = num_hashes,
        count = 0
    }, bloom_filter_mt)
end

function bloom_filter_mt:hash(item, seed)
    local str = tostring(item)
    local hash = seed or 0
    
    for i = 1, #str do
        hash = bit.bxor(hash, string.byte(str, i))
        hash = bit.band(hash * 16777619, 0xFFFFFFFF)
    end
    
    return hash % self.size
end

function bloom_filter_mt:add(item)
    for i = 1, self.num_hashes do
        local hash = self:hash(item, i)
        local byte_index = math.floor(hash / 8)
        local bit_index = hash % 8
        self.bits[byte_index] = bit.bor(self.bits[byte_index], bit.lshift(1, bit_index))
    end
    self.count = self.count + 1
end

function bloom_filter_mt:contains(item)
    for i = 1, self.num_hashes do
        local hash = self:hash(item, i)
        local byte_index = math.floor(hash / 8)
        local bit_index = hash % 8
        if bit.band(self.bits[byte_index], bit.lshift(1, bit_index)) == 0 then
            return false
        end
    end
    return true -- Possibly in set
end

-- ============================================================================
-- INITIALIZATION AND CLEANUP
-- ============================================================================

function concord.init()
    print("[Concord Ultra] Performance library v" .. concord.version .. " initialized")
    print("[Concord Ultra] JIT Status:", jit.status())
    print("[Concord Ultra] FFI Available:", ffi and "Yes" or "No")
    
    -- Initialize subsystems
    concord.memory.init_pools()
    concord.math.seed_random(os.time())
    
    -- Start background tasks
    timer.Create("concord_cache_cleanup", 30, 0, concord.cache.cleanup)
    timer.Create("concord_async_process", 0.001, 0, concord.async.process_scheduled)
    timer.Create("concord_net_batch_flush", BATCH_INTERVAL, 0, function()
        if next(message_batch) then
            concord.net.flush_message_batch()
        end
    end)
    
    print("[Concord Ultra] Background systems started")
    print("[Concord Ultra] Memory pools initialized")
    print("[Concord Ultra] Cache system ready")
    print("[Concord Ultra] Async task system ready")
    print("[Concord Ultra] Network optimization ready")
end

function concord.shutdown()
    -- Stop timers
    timer.Remove("concord_cache_cleanup")
    timer.Remove("concord_async_process")
    timer.Remove("concord_net_batch_flush")
    
    -- Cleanup resources
    concord.cache.clear()
    concord.perf.clear_stats()
    
    -- Return pooled coroutines
    for co in pairs(active_coroutines) do
        concord.async.return_coroutine(co)
    end
    
    print("[Concord Ultra] Performance library shut down")
end

-- Auto-initialize
concord.init()

-- ============================================================================
-- COMPREHENSIVE BENCHMARKING SUITE
-- ============================================================================

concord.benchmark = {}

function concord.benchmark.run_full_suite()
    print("\n[Concord Benchmark] Starting comprehensive performance tests...")
    
    -- Memory allocation benchmark
    concord.benchmark.test_memory_allocation()
    
    -- Vector math benchmark
    concord.benchmark.test_vector_math()
    
    -- Hash table benchmark
    concord.benchmark.test_hash_table()
    
    -- String operations benchmark
    concord.benchmark.test_string_operations()
    
    -- Math functions benchmark
    concord.benchmark.test_math_functions()
    
    -- Collection operations benchmark
    concord.benchmark.test_collections()
    
    print("[Concord Benchmark] Full suite completed!")
end

function concord.benchmark.test_memory_allocation()
    print("\n--- Memory Allocation Benchmark ---")
    
    local iterations = 100000
    
    -- Test fast allocation
    local start = SysTime()
    for i = 1, iterations do
        local ptr = concord.memory.fast_alloc(64)
        concord.memory.fast_free(ptr, 64)
    end
    local fast_time = SysTime() - start
    
    -- Test standard allocation
    start = SysTime()
    for i = 1, iterations do
        local ptr = ffi.C.malloc(64)
        ffi.C.free(ptr)
    end
    local std_time = SysTime() - start
    
    print(string.format("Fast allocation: %.6f seconds", fast_time))
    print(string.format("Standard allocation: %.6f seconds", std_time))
    print(string.format("Speedup: %.2fx", std_time / fast_time))
end

function concord.benchmark.test_vector_math()
    print("\n--- Vector Math Benchmark ---")
    
    local iterations = 1000000
    local v1 = concord.vector.new(1, 2, 3)
    local v2 = concord.vector.new(4, 5, 6)
    
    -- Test vector operations
    local start = SysTime()
    for i = 1, iterations do
        local dot = v1:dot(v2)
        local cross = v1:cross(v2)
        local len = v1:length()
    end
    local vector_time = SysTime() - start
    
    print(string.format("Vector operations: %.6f seconds", vector_time))
    print(string.format("Operations per second: %.0f", iterations / vector_time))
end

function concord.benchmark.test_hash_table()
    print("\n--- Hash Table Benchmark ---")
    
    local iterations = 100000
    local hash_table = concord.hash.new()
    
    -- Test insertions
    local start = SysTime()
    for i = 1, iterations do
        hash_table:set("key" .. i, i)
    end
    local insert_time = SysTime() - start
    
    -- Test lookups
    start = SysTime()
    for i = 1, iterations do
        local value = hash_table:get("key" .. i)
    end
    local lookup_time = SysTime() - start
    
    print(string.format("Hash insertions: %.6f seconds", insert_time))
    print(string.format("Hash lookups: %.6f seconds", lookup_time))
    print(string.format("Insert rate: %.0f/sec", iterations / insert_time))
    print(string.format("Lookup rate: %.0f/sec", iterations / lookup_time))
end

function concord.benchmark.test_string_operations()
    print("\n--- String Operations Benchmark ---")
    
    local iterations = 10000
    local test_string = string.rep("Hello World! ", 1000)
    
    -- Test compression
    local start = SysTime()
    for i = 1, iterations do
        local compressed = concord.net.compress_rle(test_string)
        local decompressed = concord.net.decompress_rle(compressed)
    end
    local compression_time = SysTime() - start
    
    print(string.format("String compression: %.6f seconds", compression_time))
    print(string.format("Compression rate: %.0f/sec", iterations / compression_time))
end

function concord.benchmark.test_math_functions()
    print("\n--- Math Functions Benchmark ---")
    
    local iterations = 1000000
    
    -- Test fast trigonometry
    local start = SysTime()
    for i = 1, iterations do
        local x = i * 0.001
        local sin_val = concord.math.fast_sin(x)
        local cos_val = concord.math.fast_cos(x)
    end
    local fast_trig_time = SysTime() - start
    
    -- Test standard trigonometry
    start = SysTime()
    for i = 1, iterations do
        local x = i * 0.001
        local sin_val = math.sin(x)
        local cos_val = math.cos(x)
    end
    local std_trig_time = SysTime() - start
    
    print(string.format("Fast trigonometry: %.6f seconds", fast_trig_time))
    print(string.format("Standard trigonometry: %.6f seconds", std_trig_time))
    print(string.format("Speedup: %.2fx", std_trig_time / fast_trig_time))
end

function concord.benchmark.test_collections()
    print("\n--- Collections Benchmark ---")
    
    local iterations = 50000
    
    -- Test priority queue
    local pq = concord.collections.priority_queue_new()
    
    local start = SysTime()
    for i = 1, iterations do
        pq:push(math.random())
    end
    for i = 1, iterations do
        pq:pop()
    end
    local pq_time = SysTime() - start
    
    print(string.format("Priority queue ops: %.6f seconds", pq_time))
    print(string.format("Operations per second: %.0f", (iterations * 2) / pq_time))
end

-- Export the main namespace
return concord