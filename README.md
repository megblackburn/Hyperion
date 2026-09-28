# Define and Run a Hyperion Model for Local ISM


## Install Hyperion

```
conda install -c conda-forge hyperion
```

## Dust Map Interpolation

Interpolate the dust map data provided by Edenhofer et al. (2023), available on [Zenodo](10.5281/zenodo.8187942) in cartesian:

```bash
python3 interp2box.py -o edenhofer_dustmaps/cartesian -b '(1024,1024,1024)::((-1250, 1250), (-1250,1250), (-1250,1250))' -- ./path/to/mean_and_std_healpix.fits
```
or spherical:

```bash
python3 interp2lbd.py -o edenhofer_dustmaps/spherical -b "(1572,786,2381)::((180,-180),(-90,90),(69,1250))" -- ./path/to/mean_and_std_healpix.fits
```

## Setup Hyperion Model

[mw.py](mw.py) enables setup for the O star sources utilised in McCallum et al. (2025) available [here](https://zenodo.org/records/15041318), the Astrodust+PAH model of  Hensley & Draine (2022) presented [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3B6E6S), and the Edenhofer et al. (2023) dust map in a spherical grid.

```
python3 mw.py
```

## Run Hyperion Model

```bash
yes y | hyperion {input_model_name}.rtin {output_model_name}.rtout
```
