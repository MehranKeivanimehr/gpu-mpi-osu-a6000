/* allreduce_sizes.c -- time MPI_Allreduce (MPI_FLOAT, MPI_SUM) at arbitrary sizes, on device or host buffers.
 *
 * Usage: allreduce_sizes -m dev|host -s 24,28,32,... (MiB, run in the order given) [-w warmup] [-n iters]
 *
 * Per size:
 *   - send/recv buffers are the first `bytes` of one max-size allocation per rank (allocated once, like OMB);
 *   - rank r fills element i of sendbuf with v(r,i) = (i % 251) + 1 + 1000*r  (small integers: float sums are exact);
 *   - recvbuf is filled with 0xFF bytes (a NaN pattern) before the warmup and again before the timed loop;
 *   - `warmup` untimed calls, then `iters` timed calls, each preceded by MPI_Barrier; each call is timed with MPI_Wtime;
 *   - the reported per-iteration time is the MAX across ranks (MPI_Reduce MAX over the per-iteration arrays);
 *   - correctness: after the timed loop every rank copies recvbuf to the host and compares a sample of elements
 *     (every `stride`-th element plus the first and last 4096) with the host-computed sum
 *     n*((i%251)+1) + 1000*n(n-1)/2. n_bad is summed over ranks.
 * Output (rank 0): one "RESULT" line and one "ITERS" line (all per-iteration max times, us) per size.
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    MPI_Abort(MPI_COMM_WORLD, 3); } } while (0)

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *mem = "dev";
    char sizes_arg[4096] = "32,64";
    int warmup = 5, iters = 20;
    long stride = 997;  /* sample every 997th element (prime, so it walks across pages and chunk boundaries) */
    int opt;
    while ((opt = getopt(argc, argv, "m:s:w:n:k:")) != -1) {
        if (opt == 'm') mem = optarg;
        else if (opt == 's') { strncpy(sizes_arg, optarg, sizeof(sizes_arg) - 1); }
        else if (opt == 'w') warmup = atoi(optarg);
        else if (opt == 'n') iters = atoi(optarg);
        else if (opt == 'k') stride = atol(optarg);
    }
    int on_dev = (strcmp(mem, "dev") == 0);

    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    /* parse sizes (MiB) */
    long sizes[64]; int nsizes = 0; long max_mib = 0;
    char tmp[4096]; strncpy(tmp, sizes_arg, sizeof(tmp) - 1); tmp[sizeof(tmp) - 1] = 0;
    for (char *t = strtok(tmp, ","); t && nsizes < 64; t = strtok(NULL, ",")) {
        sizes[nsizes] = atol(t);
        if (sizes[nsizes] > max_mib) max_mib = sizes[nsizes];
        nsizes++;
    }
    size_t max_bytes = (size_t)max_mib << 20;
    size_t max_count = max_bytes / sizeof(float);

    /* identify the GPU (also for host-buffer runs: MVAPICH initialises CUDA either way) */
    char busid[32] = "none";
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0) {
        CK(cudaSetDevice(0));
        CK(cudaDeviceGetPCIBusId(busid, sizeof(busid), 0));
    }
    char lib[MPI_MAX_LIBRARY_VERSION_STRING]; int liblen;
    MPI_Get_library_version(lib, &liblen);
    char *nl = strchr(lib, '\n'); if (nl) *nl = 0;
    char host[256]; gethostname(host, sizeof(host));
    for (int r = 0; r < nranks; r++) {
        if (r == rank)
            printf("RANKINFO rank=%d nranks=%d host=%s cuda_visible_devices=%s gpu_busid=%s mem=%s\n", rank, nranks,
                   host, getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "unset", busid, mem);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (rank == 0) {
        printf("CONFIG mpi_library=\"%s\" warmup=%d iters=%d stride=%ld sizes_mib=%s datatype=MPI_FLOAT op=MPI_SUM\n",
               lib, warmup, iters, stride, sizes_arg);
        fflush(stdout);
    }

    /* buffers */
    float *h_init = (float *)malloc(max_bytes);
    float *h_check = (float *)malloc(max_bytes);
    if (!h_init || !h_check) { fprintf(stderr, "host malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    for (size_t i = 0; i < max_count; i++) h_init[i] = (float)((i % 251) + 1 + 1000 * rank);
    float *sbuf, *rbuf;
    if (on_dev) {
        CK(cudaMalloc((void **)&sbuf, max_bytes));
        CK(cudaMalloc((void **)&rbuf, max_bytes));
        CK(cudaMemcpy(sbuf, h_init, max_bytes, cudaMemcpyHostToDevice));
    } else {
        sbuf = (float *)malloc(max_bytes);
        rbuf = (float *)malloc(max_bytes);
        if (!sbuf || !rbuf) { fprintf(stderr, "host malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
        memcpy(sbuf, h_init, max_bytes);
    }
    double *t = (double *)malloc(sizeof(double) * iters);
    double *tmax = (double *)malloc(sizeof(double) * iters);
    double *srt = (double *)malloc(sizeof(double) * iters);
    const float offs = (float)(1000.0 * nranks * (nranks - 1) / 2.0);

    for (int s = 0; s < nsizes; s++) {
        size_t bytes = (size_t)sizes[s] << 20;
        int count = (int)(bytes / sizeof(float));
        /* poison recvbuf */
        if (on_dev) CK(cudaMemset(rbuf, 0xFF, bytes)); else memset(rbuf, 0xFF, bytes);
        for (int w = 0; w < warmup; w++)
            MPI_Allreduce(sbuf, rbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        if (on_dev) CK(cudaMemset(rbuf, 0xFF, bytes)); else memset(rbuf, 0xFF, bytes);
        if (on_dev) CK(cudaDeviceSynchronize());
        for (int it = 0; it < iters; it++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            MPI_Allreduce(sbuf, rbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            t[it] = MPI_Wtime() - t0;
        }
        MPI_Reduce(t, tmax, iters, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        /* correctness on a sample */
        const float *res;
        if (on_dev) { CK(cudaMemcpy(h_check, rbuf, bytes, cudaMemcpyDeviceToHost)); res = h_check; }
        else res = rbuf;
        long nbad = 0, nchk = 0;
        for (size_t i = 0; i < (size_t)count; ) {
            float expect = (float)nranks * (float)((i % 251) + 1) + offs;
            if (res[i] != expect) {
                if (nbad < 3)
                    fprintf(stderr, "BAD rank=%d size_mib=%ld i=%zu got=%g expect=%g\n", rank, sizes[s], i,
                            (double)res[i], (double)expect);
                nbad++;
            }
            nchk++;
            if (i < 4096 || i + 4096 >= (size_t)count) i++; else i += stride;
        }
        long nbad_all = 0, nchk_all = 0;
        MPI_Reduce(&nbad, &nbad_all, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&nchk, &nchk_all, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            memcpy(srt, tmax, sizeof(double) * iters);
            qsort(srt, iters, sizeof(double), cmp_d);
            double med = (iters % 2) ? srt[iters / 2] : 0.5 * (srt[iters / 2 - 1] + srt[iters / 2]);
            double sum = 0; for (int it = 0; it < iters; it++) sum += tmax[it];
            double busbw = (double)bytes * 2.0 * (nranks - 1) / nranks / med / 1e9;
            printf("RESULT size_mib=%ld bytes=%zu mem=%s iters=%d median_us=%.2f min_us=%.2f max_us=%.2f mean_us=%.2f "
                   "busbw_GBps_at_median=%.3f n_checked=%ld n_bad=%ld\n",
                   sizes[s], bytes, mem, iters, med * 1e6, srt[0] * 1e6, srt[iters - 1] * 1e6, sum / iters * 1e6,
                   busbw, nchk_all, nbad_all);
            printf("ITERS size_mib=%ld t_us=", sizes[s]);
            for (int it = 0; it < iters; it++) printf("%s%.2f", it ? "," : "", tmax[it] * 1e6);
            printf("\n");
            fflush(stdout);
        }
    }
    if (on_dev) { CK(cudaFree(sbuf)); CK(cudaFree(rbuf)); } else { free(sbuf); free(rbuf); }
    free(h_init); free(h_check); free(t); free(tmax); free(srt);
    MPI_Finalize();
    return 0;
}
