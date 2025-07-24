# Concord Library v2.0.0

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Memory Management](#memory-management)
  - [Vector Math](#vector-math)
  - [Hash Tables & Data Structures](#hash-tables--data-structures)
  - [Spatial Partitioning](#spatial-partitioning)
  - [String Operations](#string-operations)
  - [Performance Monitoring](#performance-monitoring)
  - [Advanced Math](#advanced-math)
  - [Garry's Mod Optimizations](#garrys-mod-optimizations)
  - [Caching System](#caching-system)
  - [Networking](#networking)
  - [Async Tasks](#async-tasks)
  - [Collections](#collections)
- [Benchmarking](#benchmarking)
- [Performance Tips](#performance-tips)
- [License](#license)

## Features ✨

### Core Performance
- **LuaJIT FFI Integration**: Direct C library access for maximum speed
- **JIT Optimization**: Automatic hotloop detection and compilation
- **Memory Pooling**: Ultra-fast allocation with multiple size classes
- **SIMD-like Operations**: Batch processing for vectors and arrays
- **Lock-free Data Structures**: High-performance concurrent collections

### Mathematics & Algorithms
- **Ultra-Fast Vector Math**: Optimized 2D/3D/4D vector operations
- **Matrix Operations**: 4x4 and 3x3 matrix math with transformations
- **Quaternion Math**: Rotation and orientation calculations
- **Fast Trigonometry**: Lookup table-based sin/cos/tan functions
- **Advanced Interpolation**: Linear, smooth step, and spherical interpolation

### Data Structures
- **Hash Tables**: Custom high-performance hash tables with collision handling
- **Dynamic Arrays**: Auto-resizing arrays with pooled memory
- **Octree**: 3D spatial partitioning for collision detection
- **Priority Queues**: Binary heap implementation
- **Bloom Filters**: Probabilistic set membership testing
- **Lock-free Queues**: Thread-safe queue operations

### Garry's Mod Specific
- **Entity Pooling**: Reusable entity management system
- **Batch Operations**: Bulk entity property updates
- **Fast Collision Detection**: Ray-sphere, ray-AABB, sphere-sphere
- **Optimized Distance Calculations**: 2D/3D distance with squared variants
- **Spatial Queries**: Fast entity-in-sphere lookups

### Networking & Compression
- **RLE Compression**: Run-length encoding for data compression
- **LZ77 Compression**: Dictionary-based compression algorithm
- **Message Batching**: Reduced network overhead through batching
- **Delta Compression**: Efficient state synchronization

### Additional Systems
- **Advanced Caching**: TTL-based caching with automatic cleanup
- **Async Task System**: Coroutine pooling and task scheduling
- **Performance Profiling**: Built-in timing and memory profiling
- **String Optimization**: Fast string building and search algorithms

## Installation 📦

1. Download the `concord-fast.lua` file
2. Place it in your Garry's Mod addon's `lua` folder
3. Include it in your script:

```lua
local concord = include("concord-fast.lua")
```

## Quick Start 🚀

```lua
local concord = require("concord-fast")

-- Vector math example
local v1 = concord.vector.new(1, 2, 3)
local v2 = concord.vector.new(4, 5, 6)
local cross = v1:cross(v2)
local dot = v1:dot(v2)

-- Hash table example
local hash = concord.hash.new()
hash:set("player_scores", {player1 = 100, player2 = 200})
local scores = hash:get("player_scores")

-- Performance monitoring
concord.perf.profile_begin("my_function")
-- Your code here
concord.perf.profile_end()

-- Fast math
local fast_sin = concord.math.fast_sin(1.57) -- ≈ 1.0
local fast_sqrt = concord.math.fast_sqrt(16) -- 4.0

-- Entity operations (Garry's Mod)
local players = concord.gmod.fast_get_players()
local entities_in_sphere = concord.gmod.fast_get_entities_in_sphere(Vector(0,0,0), 500)
```

## API Reference 📚

### Memory Management

Ultra-fast memory allocation with pooling system.

```lua
-- Initialize memory pools (done automatically)
concord.memory.init_pools()

-- Fast allocation and deallocation
local ptr = concord.memory.fast_alloc(size)
concord.memory.fast_free(ptr, size)

-- Aligned allocation for SIMD operations
local aligned_ptr = concord.memory.aligned_alloc(size, alignment)

-- Bulk memory operations
concord.memory.bulk_copy(dest_array, src_array, count, element_size)
concord.memory.bulk_set(array, value, count, element_size)
local is_equal = concord.memory.bulk_compare(array1, array2, count, element_size)

-- Memory statistics
local stats = concord.memory.get_stats()
-- Returns: {pools_allocated, pools_freed, large_allocated, large_freed, 
--          bytes_allocated, bytes_freed, active_allocations, active_bytes}
```

### Vector Math

High-performance 3D vector operations with optional single precision.

```lua
-- Create vectors
local vec = concord.vector.new(x, y, z)        -- Double precision
local vec_f = concord.vector.new_f(x, y, z)    -- Single precision (faster)

-- Vector operations
local dot_product = vec1:dot(vec2)
local cross_product = vec1:cross(vec2)
local length = vec:length()
local length_fast = vec:length_fast()           -- Uses single precision
local length_squared = vec:length_squared()     -- Avoids sqrt

-- Normalization
vec:normalize()                                 -- Standard normalization
vec:normalize_fast()                           -- Fast inverse sqrt method

-- Distance calculations
local distance = vec1:distance_to(vec2)
local dist_squared = vec1:distance_to_squared(vec2)

-- Interpolation
vec1:lerp(vec2, t)                             -- Linear interpolation
vec1:slerp(vec2, t)                            -- Spherical interpolation

-- Vector utilities
vec:reflect(normal)                            -- Reflect across normal
vec:project(onto)                              -- Project onto another vector

-- Batch operations for multiple vectors
concord.vector.batch_add(vectors, other_vector)
concord.vector.batch_normalize(vectors)
concord.vector.batch_scale(vectors, scale_factor)
```

### Matrix Operations

4x4 and 3x3 matrix math for transformations.

```lua
-- Create matrix (identity by default)
local mat = concord.matrix.new()

-- Matrix operations
local result = concord.matrix.multiply(mat1, mat2)
concord.matrix.translate(mat, x, y, z)
concord.matrix.scale(mat, x, y, z)

-- Quaternions for rotations
local quat = concord.quaternion.new(x, y, z, w)
local result = concord.quaternion.multiply(quat1, quat2)
```

### Hash Tables & Data Structures

Ultra-fast hash tables with automatic resizing.

```lua
-- Create hash table
local hash = concord.hash.new(initial_size)

-- Basic operations
hash:set(key, value)
local value = hash:get(key)
local success = hash:remove(key)

-- Automatic resizing when load factor exceeded
hash:resize(new_size)

-- Dynamic arrays
local array = concord.array.new(initial_capacity)
array:push(item)
local item = array:pop()
local item = array:get(index)
array:set(index, item)
local size = array:size()
array:clear()
```

### Spatial Partitioning

3D octree for efficient spatial queries and collision detection.

```lua
-- Create octree
local bounds = ffi.new("aabb_t")  -- Define bounding box
local octree = concord.spatial.octree_new(bounds, max_objects, max_depth)

-- Insert objects
local position = concord.vector.new(x, y, z)
octree:insert(object, position)

-- Query objects in range
local query_bounds = ffi.new("aabb_t")  -- Define query area
local results = octree:query_range(query_bounds)

-- Collision detection utilities
local ray = ffi.new("ray_t")
local sphere = ffi.new("sphere_t")
local hit, distance = concord.collision.ray_sphere_intersect(ray, sphere)

local aabb = ffi.new("aabb_t")
local hit, distance = concord.collision.ray_aabb_intersect(ray, aabb)

local intersects = concord.collision.sphere_sphere_intersect(sphere1, sphere2)
```

### String Operations

Ultra-optimized string building and searching.

```lua
-- String builder for efficient concatenation
local builder = concord.string.new_builder(initial_capacity)
builder:append("Hello")
builder:append_char(32) -- Space character
builder:append_number(42)
local result = builder:to_string()
builder:clear()

-- Fast string search (Boyer-Moore algorithm)
local position = concord.string.boyer_moore_search(text, pattern)

-- String hashing
local hash = concord.string.hash(string)

-- String interning for memory efficiency
local interned = concord.string.intern(string)
```

### Performance Monitoring

Advanced profiling with call stack tracking.

```lua
-- Basic profiling
concord.perf.profile_begin("function_name")
-- Your code here
concord.perf.profile_end()

-- Automatic function profiling
local profiled_func = concord.perf.profile_func("my_func", original_function)

-- High-precision timing
local time_ns = concord.perf.get_time_ns()
local time_us = concord.perf.get_time_us()
local time_ms = concord.perf.get_time_ms()

-- Memory profiling
concord.perf.memory_snapshot("snapshot_name")
local diff = concord.perf.memory_diff("snapshot1", "snapshot2")

-- Get performance statistics
local stats = concord.perf.get_stats()
-- Returns detailed timing info for all profiled functions

-- Clear statistics
concord.perf.clear_stats()
```

### Advanced Math

Ultra-fast mathematical operations with lookup tables.

```lua
-- Mathematical constants
concord.math.PI    -- π
concord.math.TAU   -- 2π
concord.math.E     -- Euler's number
concord.math.PHI   -- Golden ratio
concord.math.SQRT2 -- √2

-- Fast trigonometry (lookup table based)
local sin_val = concord.math.fast_sin(angle)
local cos_val = concord.math.fast_cos(angle)
local tan_val = concord.math.fast_tan(angle)

-- Ultra-fast square root and inverse square root
local inv_sqrt = concord.math.fast_inv_sqrt(x)
local sqrt_val = concord.math.fast_sqrt(x)

-- Fast power functions
local result = concord.math.fast_pow_int(base, integer_exponent)
local result = concord.math.fast_pow(base, float_exponent)

-- Interpolation
local lerp_result = concord.math.lerp(a, b, t)
local smooth = concord.math.smooth_step(edge0, edge1, x)
local smoother = concord.math.smoother_step(edge0, edge1, x)

-- Utility functions
local clamped = concord.math.clamp(value, min_val, max_val)
local sign = concord.math.sign(x)
local wrapped = concord.math.wrap(value, min_val, max_val)
local remapped = concord.math.remap(value, old_min, old_max, new_min, new_max)

-- Fast random number generation (xorshift)
concord.math.seed_random(seed)
local random = concord.math.fast_random()                    -- [0, 1)
local ranged = concord.math.fast_random_range(min, max)      -- [min, max)
local int_rand = concord.math.fast_random_int(min, max)      -- [min, max]

-- Noise functions
local noise = concord.math.white_noise(x, y)
```

### Garry's Mod Optimizations

Specific optimizations for Garry's Mod entities and operations.

```lua
-- Fast player retrieval
local players = concord.gmod.fast_get_players()

-- Spatial entity queries
local entities, count = concord.gmod.fast_get_entities_in_sphere(center_vector, radius)

-- Batch entity operations
concord.gmod.batch_set_position(entities, positions)
concord.gmod.batch_set_angles(entities, angles)
concord.gmod.batch_set_color(entities, colors)

-- Entity pooling system
concord.gmod.create_entity_pool("prop_physics", initial_size, max_size)
local entity = concord.gmod.get_pooled_entity("prop_physics")
concord.gmod.return_pooled_entity(entity, "prop_physics")

-- Fast trace operations
local trace = concord.gmod.fast_trace_line(start_pos, end_pos, filter)
local results = concord.gmod.batch_trace_lines(trace_data_array)

-- Optimized distance calculations
local dist_2d = concord.gmod.fast_distance_2d(pos1, pos2)
local dist_3d = concord.gmod.fast_distance_3d(pos1, pos2)
local dist_squared = concord.gmod.fast_distance_squared(pos1, pos2)
```

### Caching System

Advanced caching with TTL and automatic cleanup.

```lua
-- Basic caching operations
concord.cache.set(key, value, ttl_seconds)
local cached_value = concord.cache.get(key)
local exists = concord.cache.has(key)
concord.cache.remove(key)
concord.cache.clear()

-- Manual cleanup of expired entries
concord.cache.cleanup()

-- Cache statistics
local stats = concord.cache.get_stats()
-- Returns: {hits, misses, evictions, sets, hit_rate, size}

-- Function memoization
local memoized_func = concord.cache.memoize(expensive_function, ttl)
```

### Networking

Data compression and message batching for reduced network overhead.

```lua
-- RLE compression
local compressed = concord.net.compress_rle(data)
local decompressed = concord.net.decompress_rle(compressed)

-- LZ77 compression (better ratios)
local lz_compressed = concord.net.compress_lz77(data, window_size, lookahead_size)

-- Message batching
concord.net.batch_message(recipient, message_type, data)
concord.net.flush_message_batch()

-- Delta compression for state updates
local delta = concord.net.delta_compress(key, current_state)
local applied_state = concord.net.delta_apply(key, delta_data)
```

### Async Tasks

Coroutine pooling and task scheduling system.

```lua
-- Create async tasks
local task = concord.async.create_task(function() 
    -- Task code here
end, arg1, arg2)

-- Schedule tasks with delays and priorities
concord.async.schedule(func, delay_seconds, priority, ...)
concord.async.schedule_next_frame(func, ...)

-- Process scheduled tasks (called automatically)
concord.async.process_scheduled()

-- Parallel task execution
concord.async.parallel({task1, task2, task3}, function(results)
    -- Callback when all tasks complete
end)
```

### Collections

Advanced data structures for specialized use cases.

```lua
-- Lock-free queue
local queue = concord.collections.queue_new()
queue:enqueue(item)
local item = queue:dequeue()
local empty = queue:is_empty()

-- Priority queue (binary heap)
local pq = concord.collections.priority_queue_new(compare_function)
pq:push(item)
local top_item = pq:pop()
local peek_item = pq:peek()
local empty = pq:is_empty()

-- Bloom filter for fast set membership testing
local bloom = concord.collections.bloom_filter_new(size, num_hashes)
bloom:add(item)
local possibly_contains = bloom:contains(item)  -- May have false positives
```

## Benchmarking 📊

Run comprehensive performance tests:

```lua
-- Run all benchmarks
concord.benchmark.run_full_suite()

-- Individual benchmark tests
concord.benchmark.test_memory_allocation()
concord.benchmark.test_vector_math()
concord.benchmark.test_hash_table()
concord.benchmark.test_string_operations()
concord.benchmark.test_math_functions()
concord.benchmark.test_collections()
```

## Performance Tips 💡

### Memory Management
- Use `concord.memory.fast_alloc()` for frequently allocated/deallocated objects
- Prefer `length_squared()` over `length()` when comparing distances
- Pre-allocate collections with expected capacity
- Use single precision vectors (`new_f`) when double precision isn't needed

### Vector Math
- Use batch operations for multiple vectors
- Cache frequently used vector calculations
- Use `normalize_fast()` when slight precision loss is acceptable
- Prefer `distance_to_squared()` for distance comparisons

### Hash Tables
- Initialize with appropriate size to avoid resizing
- Use string interning for frequently used keys
- Consider bloom filters for set membership tests with large datasets

### String Operations
- Use string builder for concatenating many strings
- Intern frequently used strings to save memory
- Use Boyer-Moore search for pattern matching in large texts

### Garry's Mod Specific
- Use entity pooling for frequently created/destroyed entities
- Batch entity operations when possible
- Use spatial queries instead of iterating all entities
- Cache distance calculations between entities

### General
- Profile your code to identify bottlenecks
- Use the caching system for expensive computations
- Leverage async tasks for non-blocking operations
- Use appropriate data structures for your use case

## Thread Safety ⚠️

Most operations are **not thread-safe** except:
- Lock-free queue operations
- Atomic operations where explicitly mentioned
- Read-only operations on immutable data

## License 📄

This library is provided as-is for educational purposes. Use at your own risk.
