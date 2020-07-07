#!/usr/bin/env bash
#SBATCH -J capstone_bacteria
#SBATCH -o output-%j.txt
#SBATCH -e error-%j.txt
#SBATCH -p standard-mem-s --mem=250G
#SBATCH --mail-user jpartee@smu.edu
#SBATCH --mail-type=all

python kmer_compression_trials_with_col_comp.py $1 $2

