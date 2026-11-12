/* Runtime check that the MPI library reports CUDA support (MPICH/MVAPICH MPIX API). */
#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
    int rank, sup = -1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPIX_GPU_query_support(MPIX_GPU_SUPPORT_CUDA, &sup);
    if (rank == 0) printf("MPIX_GPU_query_support(CUDA)=%d  MPIX_Query_cuda_support()=%d\n", sup, MPIX_Query_cuda_support());
    MPI_Finalize();
    return 0;
}
