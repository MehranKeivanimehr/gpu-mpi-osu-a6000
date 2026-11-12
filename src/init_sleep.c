#include <mpi.h>
#include <unistd.h>
int main(int c, char **v) { MPI_Init(&c, &v); sleep(6); MPI_Finalize(); return 0; }
