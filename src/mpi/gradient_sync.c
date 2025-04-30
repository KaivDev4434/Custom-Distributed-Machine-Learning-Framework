#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gradient_sync.h"

// Initialize MPI
void init_mpi(int* rank, int* size) {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
}

// Synchronize gradients across all processes
void sync_gradients(float* gradients, int size) {
    float *temp_buffer = (float *)malloc(size * sizeof(float));
    if (temp_buffer == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for gradient synchronization\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // Copy original gradients to preserve them
    memcpy(temp_buffer, gradients, size * sizeof(float));

    // Use MPI_Allreduce with MPI_SUM for efficient gradient synchronization
    int ret = MPI_Allreduce(temp_buffer, gradients, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    if (ret != MPI_SUCCESS) {
        fprintf(stderr, "Error: MPI_Allreduce failed during gradient synchronization\n");
        free(temp_buffer);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // Get world size for averaging
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Average the gradients
    #pragma omp parallel for if(size > 1000)  // Use OpenMP for large arrays
    for (int i = 0; i < size; i++) {
        gradients[i] /= world_size;
    }

    free(temp_buffer);
}

// Finalize MPI
void finalize_mpi() {
    MPI_Finalize();
}

// Get the number of processes
int get_world_size() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return world_size;
}

// Get the current process rank
int get_world_rank() {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    return world_rank;
} 