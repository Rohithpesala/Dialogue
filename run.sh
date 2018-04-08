#!/bin/bash
#
#SBATCH --job-name=dialog
#SBATCH --output=mem1.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=3-08:00:00     # Runtime in D-HH:MM
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rpesala@cs.umass.edu


python main.py
sleep 1
exit
