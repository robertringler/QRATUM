//! Benchmarks for the memory subsystem.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use emu_memory::{Mmu, MemoryRegion, RegionPermissions, PAGE_SIZE};
use emu_memory::region::RegionType;

fn bench_memory_read_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_ops");
    
    // Create MMU with 64MB guest memory
    let mmu = Mmu::new(64 * 1024 * 1024);
    
    // Map a 1MB RAM region
    let region = MemoryRegion::new(
        "bench_ram",
        0x1000,
        1024 * 1024,
        RegionType::Ram,
        RegionPermissions::RW,
    );
    mmu.map_region(region).unwrap();
    
    // Benchmark sequential writes
    group.bench_function("write_u32_sequential", |b| {
        b.iter(|| {
            for i in 0..1000 {
                mmu.write_u32(0x1000 + i * 4, black_box(0xDEADBEEF)).unwrap();
            }
        })
    });
    
    // Benchmark sequential reads
    group.bench_function("read_u32_sequential", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(mmu.read_u32(0x1000 + i * 4).unwrap());
            }
        })
    });
    
    // Benchmark random access pattern
    let addrs: Vec<u64> = (0..1000).map(|i| 0x1000 + ((i * 17) % 1000) * 4).collect();
    
    group.bench_function("read_u32_random", |b| {
        b.iter(|| {
            for &addr in &addrs {
                black_box(mmu.read_u32(addr).unwrap());
            }
        })
    });
    
    group.finish();
}

fn bench_page_table(c: &mut Criterion) {
    use emu_memory::page_table::{PageTable, PageEntry, PageFlags};
    
    let mut group = c.benchmark_group("page_table");
    
    let pt = PageTable::new(0x1000_0000, 64 * 1024 * 1024);
    
    // Pre-populate with some entries
    for i in 0..10000 {
        let entry = PageEntry::new(i, PageFlags::PRESENT | PageFlags::READ);
        pt.map(i, entry).unwrap();
    }
    
    group.bench_function("translate_hit", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(pt.translate(black_box(i * PAGE_SIZE as u64 + 0x100)).unwrap());
            }
        })
    });
    
    group.bench_function("translate_miss", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let _ = pt.translate(black_box(0x1_0000_0000 + i * PAGE_SIZE as u64));
            }
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_memory_read_write, bench_page_table);
criterion_main!(benches);
