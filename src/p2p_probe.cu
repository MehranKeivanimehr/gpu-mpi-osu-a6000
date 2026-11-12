// p2p_probe.cu -- direct CUDA probe of the GPU<->GPU paths, independent of MPI/UCX/NCCL.
//
// Usage: p2p_probe <gpuA> <gpuB>
//   Part 1 (one process, both GPUs visible): cudaDeviceCanAccessPeer, cudaDeviceEnablePeerAccess time,
//          cudaMemcpyPeer A->B at several sizes (timed with events, contents checked), and a kernel on
//          B that reads A's buffer directly over P2P (contents checked).
//   Part 2 (two processes, like MPI ranks): process A allocates + exports a cudaIpcMemHandle; process B
//          times cudaIpcOpenMemHandle and a copy out of the mapped buffer, checks contents.
// Every step prints elapsed time; the caller wraps the run in `timeout`.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); fflush(stdout); exit(3);} } while (0)

static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

__global__ void fill(unsigned *p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (unsigned)(i * 2654435761u) ^ seed;
}
__global__ void check(const unsigned *p, size_t n, unsigned seed, unsigned long long *bad) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    if (p[i] != ((unsigned)(i * 2654435761u) ^ seed)) atomicAdd(bad, 1ull);
}
// B reads A's memory directly through the peer mapping and copies it to local memory
__global__ void p2p_read(const unsigned *remote, unsigned *local, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    local[i] = remote[i];
}

static unsigned long long count_bad(int dev, const unsigned *p, size_t n, unsigned seed) {
  unsigned long long *d_bad, h_bad = 0;
  CK(cudaSetDevice(dev));
  CK(cudaMalloc(&d_bad, sizeof(*d_bad)));
  CK(cudaMemset(d_bad, 0, sizeof(*d_bad)));
  check<<<1024, 256>>>(p, n, seed, d_bad);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&h_bad, d_bad, sizeof(h_bad), cudaMemcpyDeviceToHost));
  CK(cudaFree(d_bad));
  return h_bad;
}

static void part1(int a, int b) {
  int ab = 0, ba = 0;
  CK(cudaDeviceCanAccessPeer(&ab, a, b));
  CK(cudaDeviceCanAccessPeer(&ba, b, a));
  printf("part1 canAccessPeer %d->%d=%d %d->%d=%d\n", a, b, ab, b, a, ba);
  const size_t sizes[] = {4096, 1 << 20, 64 << 20};
  const size_t maxn = (64 << 20) / sizeof(unsigned);
  unsigned *pa, *pb;
  CK(cudaSetDevice(a)); CK(cudaMalloc(&pa, maxn * sizeof(unsigned)));
  CK(cudaSetDevice(b)); CK(cudaMalloc(&pb, maxn * sizeof(unsigned)));
  for (int peer = 0; peer <= 1; peer++) {
    if (peer) {
      double t0 = now_s();
      CK(cudaSetDevice(a)); cudaError_t e1 = cudaDeviceEnablePeerAccess(b, 0);
      CK(cudaSetDevice(b)); cudaError_t e2 = cudaDeviceEnablePeerAccess(a, 0);
      printf("part1 enablePeerAccess both ways: %s / %s  %.3f ms\n", cudaGetErrorString(e1),
             cudaGetErrorString(e2), 1e3 * (now_s() - t0));
      if (e1 != cudaSuccess || e2 != cudaSuccess) break;
    }
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
      size_t n = sizes[si] / sizeof(unsigned);
      unsigned seed = 0x1234u + (unsigned)si + 100u * peer;
      CK(cudaSetDevice(a)); fill<<<1024, 256>>>(pa, n, seed); CK(cudaDeviceSynchronize());
      CK(cudaSetDevice(b)); CK(cudaMemset(pb, 0, n * sizeof(unsigned))); CK(cudaDeviceSynchronize());
      cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      double t0 = now_s();
      CK(cudaEventRecord(e0, 0));
      CK(cudaMemcpyPeerAsync(pb, b, pa, a, n * sizeof(unsigned), 0));
      CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      double wall = now_s() - t0;
      unsigned long long bad = count_bad(b, pb, n, seed);
      printf("part1 memcpyPeer peer_access=%d bytes=%zu event_ms=%.3f wall_ms=%.3f GB/s=%.2f bad_words=%llu\n",
             peer, sizes[si], ms, 1e3 * wall, sizes[si] / (ms * 1e-3) / 1e9, bad);
      fflush(stdout);
      if (peer) {   // direct P2P load by a kernel on B
        CK(cudaMemset(pb, 0, n * sizeof(unsigned))); CK(cudaDeviceSynchronize());
        double t1 = now_s();
        p2p_read<<<1024, 256>>>(pa, pb, n); CK(cudaDeviceSynchronize());
        double w1 = now_s() - t1;
        bad = count_bad(b, pb, n, seed);
        printf("part1 kernelP2Pread bytes=%zu wall_ms=%.3f bad_words=%llu\n", sizes[si], 1e3 * w1, bad);
        fflush(stdout);
      }
    }
  }
  CK(cudaSetDevice(a)); CK(cudaFree(pa)); CK(cudaSetDevice(b)); CK(cudaFree(pb));
}

static int part2(int a, int b) {
  // fork before any CUDA call in the children; the parent already used CUDA in part1, so part2 runs
  // from a fresh process (main re-execs itself with "ipc" argument).
  int fds[2], back[2];
  if (pipe(fds) || pipe(back)) return 1;
  const size_t bytes = 64 << 20, n = bytes / sizeof(unsigned);
  pid_t pid = fork();
  if (pid == 0) {                       // exporter on GPU a
    CK(cudaSetDevice(a));
    unsigned *p; CK(cudaMalloc(&p, bytes));
    fill<<<1024, 256>>>(p, n, 0xBEEFu); CK(cudaDeviceSynchronize());
    cudaIpcMemHandle_t h; CK(cudaIpcGetMemHandle(&h, p));
    if (write(fds[1], &h, sizeof(h)) != sizeof(h)) exit(4);
    char c; if (read(back[0], &c, 1) != 1) exit(4);   // keep allocation alive until importer is done
    exit(0);
  }
  pid_t pid2 = fork();
  if (pid2 == 0) {                      // importer on GPU b
    cudaIpcMemHandle_t h;
    if (read(fds[0], &h, sizeof(h)) != sizeof(h)) exit(4);
    CK(cudaSetDevice(b));
    unsigned *local; CK(cudaMalloc(&local, bytes));
    for (int it = 0; it < 3; it++) {
      void *rp = NULL;
      double t0 = now_s();
      cudaError_t e = cudaIpcOpenMemHandle(&rp, h, cudaIpcMemLazyEnablePeerAccess);
      double t_open = now_s() - t0;
      if (e != cudaSuccess) { printf("part2 ipcOpen it=%d FAILED %s %.3f ms\n", it, cudaGetErrorString(e), 1e3 * t_open); fflush(stdout); break; }
      CK(cudaMemset(local, 0, bytes)); CK(cudaDeviceSynchronize());
      t0 = now_s();
      CK(cudaMemcpy(local, rp, bytes, cudaMemcpyDeviceToDevice));
      double t_copy = now_s() - t0;
      unsigned long long bad = count_bad(b, local, n, 0xBEEFu);
      t0 = now_s();
      CK(cudaIpcCloseMemHandle(rp));
      double t_close = now_s() - t0;
      printf("part2 ipc it=%d open_ms=%.3f copy64MiB_ms=%.3f GB/s=%.2f close_ms=%.3f bad_words=%llu\n", it,
             1e3 * t_open, 1e3 * t_copy, bytes / t_copy / 1e9, 1e3 * t_close, bad);
      fflush(stdout);
    }
    if (write(back[1], "x", 1) != 1) exit(4);
    exit(0);
  }
  int st1, st2; waitpid(pid2, &st2, 0); waitpid(pid, &st1, 0);
  printf("part2 exit codes exporter=%d importer=%d\n", WEXITSTATUS(st1), WEXITSTATUS(st2));
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s gpuA gpuB [ipc]\n", argv[0]); return 2; }
  int a = atoi(argv[1]), b = atoi(argv[2]);
  setvbuf(stdout, NULL, _IOLBF, 0);
  if (argc > 3 && !strcmp(argv[3], "ipc")) return part2(a, b);
  part1(a, b);
  return 0;
}
