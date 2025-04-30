#ifndef GRADIENT_SYNC_H
#define GRADIENT_SYNC_H

#ifdef __cplusplus
extern "C" {
#endif

// Synchronize gradients across all processes
void sync_gradients(float *gradients, int size);

// Initialize MPI
void init_mpi(int* rank, int* size);

// Finalize MPI
void finalize_mpi();

// Get the number of processes
int get_world_size();

// Get the current process rank
int get_world_rank();

#ifdef __cplusplus
}
#endif

#endif // GRADIENT_SYNC_H 