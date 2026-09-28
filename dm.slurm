#!/bin/bash
#SBATCH --job-name=dm
#SBATCH --partition=small-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=%x_%j.log
#SBATCH --time=02:30:00
# Load required modules

conda activate mgbenv 

python3 interp2box.py -o edenhofer_dustmaps/cartesian -b '(1024,1024,1024)::((-1250, 1250), (-1250,1250), (-1250,1250))' -- ./edenhofer_dustmaps/edenhofer_2023/mean_and_std_healpix.fits
python3 interp2lbd.py -o edenhofer_dustmaps/spherical -b "(1572,786,2381)::((180,-180),(-90,90),(69,1250))" -- ./edenhofer_dustmaps/edenhofer_2023/mean_and_std_healpix.fits
#python3 interp2lbd.py -o edenhofer_dustmaps/spherical -b '(360,180,1182)::((0,360),(-90,90),(69,1250))' -- ./edenhofer_dustmaps/edenhofer_2023/mean_and_std_healpix.fits

