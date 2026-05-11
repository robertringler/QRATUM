# SYSTEM_PROFILE.md - Platform & Environment Attestation

**Document Type**: Audit-Ready System Profile  
**Generated**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")  
**Git Commit**: d6bafe894be45d50e63a36241e536bfe70d2f6b3  
**Benchmark Session**: $TIMESTAMP  

---

## 1. CPU INFORMATION

### Summary

| Property | Value |
|----------|-------|
| Model | AMD EPYC 7763 64-Core Processor |
| Family | 25 |
| Stepping | 1 |
| Cores/Socket | 2 |
| Threads/Core | 2 |
| Total CPUs | 4 |
| BogoMIPS | 4890.86 |
| Virtualization | AMD-V (Microsoft Hypervisor) |

### SIMD Capabilities

- SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2
- AVX, AVX2
- FMA
- AES-NI (hardware AES)
- SHA extensions

### Cache Hierarchy

| Level | Size | Instances |
|-------|------|-----------|
| L1d | 64 KiB | 2 |
| L1i | 64 KiB | 2 |
| L2 | 1 MiB | 2 |
| L3 | 32 MiB | 1 |

### NUMA Topology

- NUMA Nodes: 1
- Node 0 CPUs: 0-3

### Raw lscpu Output

```
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        48 bits physical, 48 bits virtual
Byte Order:                           Little Endian
CPU(s):                               4
On-line CPU(s) list:                  0-3
Vendor ID:                            AuthenticAMD
Model name:                           AMD EPYC 7763 64-Core Processor
CPU family:                           25
Model:                                1
Thread(s) per core:                   2
Core(s) per socket:                   2
Socket(s):                            1
Stepping:                             1
BogoMIPS:                             4890.86
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves user_shstk clzero xsaveerptr rdpru arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip vaes vpclmulqdq rdpid fsrm
Virtualization:                       AMD-V
Hypervisor vendor:                    Microsoft
Virtualization type:                  full
L1d cache:                            64 KiB (2 instances)
L1i cache:                            64 KiB (2 instances)
L2 cache:                             1 MiB (2 instances)
L3 cache:                             32 MiB (1 instance)
NUMA node(s):                         1
NUMA node0 CPU(s):                    0-3
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Vulnerable: Safe RET, no microcode
Vulnerability Spec store bypass:      Vulnerable
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Retpolines; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected
```

---

## 2. MEMORY INFORMATION

### Summary

| Property | Value |
|----------|-------|
| Total RAM | 16 GiB |
| Available | ~15 GiB |
| Swap | 4 GiB |

### Raw /proc/meminfo Output

```
MemTotal:       16379472 kB
MemFree:         7094052 kB
MemAvailable:   15037412 kB
Buffers:           82212 kB
Cached:          7973192 kB
SwapCached:            0 kB
Active:          1231628 kB
Inactive:        7496952 kB
Active(anon):     728528 kB
Inactive(anon):        0 kB
Active(file):     503100 kB
Inactive(file):  7496952 kB
Unevictable:       46772 kB
Mlocked:           43700 kB
SwapTotal:       4194300 kB
SwapFree:        4194300 kB
Zswap:                 0 kB
Zswapped:              0 kB
Dirty:              2084 kB
Writeback:             0 kB
AnonPages:        702040 kB
Mapped:           378436 kB
Shmem:             46608 kB
KReclaimable:     289236 kB
Slab:             385820 kB
SReclaimable:     289236 kB
SUnreclaim:        96584 kB
KernelStack:        5440 kB
PageTables:        16472 kB
SecPageTables:         0 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:    12384036 kB
Committed_AS:    2997748 kB
VmallocTotal:   34359738367 kB
VmallocUsed:       33888 kB
VmallocChunk:          0 kB
Percpu:             2048 kB
HardwareCorrupted:     0 kB
AnonHugePages:    186368 kB
ShmemHugePages:        0 kB
ShmemPmdMapped:        0 kB
FileHugePages:         0 kB
FilePmdMapped:         0 kB
Unaccepted:            0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB
DirectMap4k:      114624 kB
DirectMap2M:     5128192 kB
DirectMap1G:    13631488 kB
```

---

## 3. GPU INFORMATION

### Status

**NO NVIDIA GPU DETECTED**

This benchmark will execute CPU-only workloads. GPU-accelerated benchmarks (CUDA/cuQuantum) are excluded from scope.

---

## 4. STORAGE INFORMATION

### Summary

| Filesystem | Size | Used | Available | Mount |
|------------|------|------|-----------|-------|
| /dev/root | 72G | 51G | 22G | / |
| /dev/sda1 | 74G | 4.1G | 66G | /mnt |

### Block Devices

```
NAME    MAJ:MIN RM  SIZE RO TYPE MOUNTPOINTS
sda       8:0    0   75G  0 disk 
└─sda1    8:1    0   75G  0 part /mnt
sdb       8:16   0   75G  0 disk 
├─sdb1    8:17   0   74G  0 part /
├─sdb14   8:30   0    4M  0 part 
├─sdb15   8:31   0  106M  0 part /boot/efi
└─sdb16 259:0    0  913M  0 part /boot
```

---

## 5. OPERATING SYSTEM & KERNEL

### Summary

| Property | Value |
|----------|-------|
| OS | Ubuntu 24.04.3 LTS (Noble Numbat) |
| Kernel | 6.11.0-1018-azure |
| Architecture | x86_64 |
| Hypervisor | Microsoft Azure |

### Raw os-release

```
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo
```

---

## 6. SOFTWARE ENVIRONMENT

### Compilers & Interpreters

| Tool | Version |
|------|---------|
| Python | 3.12.3 |
| pip | 24.0 |
| GCC | 13.3.0 |

### Python Dependencies (Benchmark-Relevant)

| Package | Version |
|---------|---------|
| NumPy | 2.4.1 |
| SciPy | 1.17.0 |
| PyYAML | 6.0.1 |
| Matplotlib | 3.10.8 |
| NetworkX | 3.6.1 |

---

## 7. PERFORMANCE-RELEVANT CONFIGURATION

### CPU Governor

- Managed by hypervisor (Azure)
- No direct tuning access

### SIMD Optimization

- AVX2 available and enabled
- SHA extensions available for cryptographic operations

### Memory Configuration

- THP (Transparent Huge Pages): System default
- No custom NUMA binding (single NUMA node)

---

## 8. LIMITATIONS & EXCLUSIONS

1. **No GPU**: CUDA/cuQuantum benchmarks excluded
2. **Virtualized**: Some low-level CPU features may be limited
3. **Shared Resources**: Azure VM may have variable I/O performance
4. **No Quantum Hardware**: All quantum benchmarks are simulation-only

---

**END OF SYSTEM PROFILE**
