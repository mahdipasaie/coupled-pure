#!/bin/bash
#SBATCH --account=def-oforion
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=14           # number of MPI processes
#SBATCH --mem-per-cpu=10000M    # memory; default unit is megabytes
#SBATCH --time=7-00:00        # time (DD-HH:MM)

# Load the required modules
module load StdEnv/2020
module load gcc/9.3.0
module load hdf5-mpi/1.10.6
module load boost/1.72.0
module load eigen
module load python/3.10.2
module load scipy-stack/2023b
module load mpi4py/3.0.3
module load petsc/3.17.1
module load slepc/3.17.2
module load scotch/6.0.9
module load fftw-mpi/3.3.8
module load ipp/2020.1.217
module load swig
module load flexiblas

# Activate FEniCS virtual environment and run your Python script
source /home/pasha77/fenics/bin/activate
source /home/pasha77/fenics/share/dolfin/dolfin.conf

srun python coupled_cluster.py






