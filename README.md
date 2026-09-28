# Helpful Code to Define and Run a Hyperion Model for Local ISM


## Dust Map Interpolation

python3 interp2box.py -o edenhofer_dustmaps/cartesian -b '(1024,1024,1024)::((-1250, 1250), (-1250,1250), (-1250,1250))' -- ./edenhofer_dustmaps/edenhofer_2023/mean_and_std_healpix.fits
python3 interp2lbd.py -o edenhofer_dustmaps/spherical -b "(1572,786,2381)::((180,-180),(-90,90),(69,1250))" -- ./edenhofer_dustmaps/edenhofer_2023/mean_and_std_healpix.fits


## Run Hyperion Model

yes y | hyperion local_bubble_astrodust_1e7photons_dustmap.rtin local_bubble_astrodust_1e7photons_dustmap.rtout
