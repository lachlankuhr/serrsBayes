#!/bin/bash -l
#PBS -N mars-2020-cpu
#PBS -l walltime=4:00:0
#PBS -l ncpus=16
#PBS -l mem=8gb
#PBS -l cpuarch=avx2
#PBS -j oe
#PBS -V
$PBS_NODEFILE
module load r/3.5.1-foss-2018a
cd $PBS_O_WORKDIR
Rscript --no-save ../vignettes/fitVoigtPeaksSMC_2.R
