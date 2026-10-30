#!/bin/bash
# Maps MPI rank variables to the ones DeepSpeed reads, then starts Python.
export RANK=${OMPI_COMM_WORLD_RANK:-$PMI_RANK}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-$PMI_LOCAL_RANK}
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-$PMI_SIZE}
exec python3 "$@"
