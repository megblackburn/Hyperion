from dustmaps.config import config
import dustmaps.edenhofer2023
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import healpy as hp
from dustmaps.edenhofer2023 import Edenhofer2023Query
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits
from hyperion.model import Model
import matplotlib.pyplot as plt
from hyperion.model import ModelOutput
import os
from hyperion.util.integrate import integrate_loglog

from wmbasic_loader import wmbasicspectra

## constants:

speed_of_light = 299792458 # m/s
planck = 6.62607015e-34

if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
   
   



def create_hyperion_model(remake_dust_map, define_model, use_single_channel, use_spherex_channels, add_specific_sources, source_names, specific_source_string, spherex_max_wav, spherex_min_wav, n_wavelength_bins,  N_photons, N_iterations, low_res = True):
    if remake_dust_map == True:
        print('Creating new dust map from interpolation ...')

        if low_res == True:
            extinction_map = fits.open('./edenhofer_dustmaps/spherical_lowres/_180_-180_-90_90_69_1250_lbd.fits')
        else:
            extinction_map = fits.open('./edenhofer_dustmaps/spherical/_180_-180_-90_90_69_1250_lbd.fits')

        A = extinction_map[1].data 
        print('max and min A: ', np.nanmax(A), np.nanmin(A))

        A = np.nan_to_num(A, nan=0.0)

        ni_factor = 2700 # new val?

        n_h = A * ni_factor

        mu_h = 1.4 * 1.67e-24  # grams maybe 1.4 *
        dust_to_gas_ratio = 0.000605 # mccallum 2025 val

        gas_density = n_h * mu_h
        dust_density = gas_density * dust_to_gas_ratio # A * mu_h, n_h * mu_h * dust_to_gas_ratio
        print('max dust density: ', np.nanmax(dust_density), np.nanmin(dust_density))
        print('max gas density: ', np.nanmax(gas_density), np.nanmin(gas_density))
        
        if low_res == True:
            np.save('./edenhofer_dustmaps/spherical_lowres/dust_dens_spherical_lowres.npy', dust_density)
            np.save('./edenhofer_dustmaps/spherical_lowres/gas_dens_spherical_lowres.npy', gas_density)
        else:
            np.save('./edenhofer_dustmaps/spherical/dust_dens_spherical_highres.npy', dust_density)
            np.save('./edenhofer_dustmaps/spherical/gas_dens_spherical_highres.npy', gas_density)
    else:
        print('Loading existing dust map from file ...')
        if low_res == True: 
            dust_density = np.load('./edenhofer_dustmaps/spherical_lowres/dust_dens_spherical_lowres.npy')
            gas_density = np.load('./edenhofer_dustmaps/spherical_lowres/gas_dens_spherical_lowres.npy')
        else:   
            dust_density = np.load('./edenhofer_dustmaps/spherical/dust_dens_spherical_highres.npy')
            gas_density = np.load('./edenhofer_dustmaps/spherical/gas_dens_spherical_highres.npy')


    print('Dust model has been loaded')

    sourcepath = './stellarsources/processed_sources.fits'

    # useful constants

    pc = 3.0856e18 # in cm

    min_distance = 69 * pc # pc
    max_distance = 1250 * pc # pc

    nphi = 1572
    ntheta = 786
    nr = 2381 #1250-69+1

    low_res = True
    if low_res == True:
        nphi = 786
        ntheta = 393
        nr = 1190 #1250-69+1
        
    if define_model == True:
        with fits.open(sourcepath) as file:
            data = file[1].data
            header = file[1].header
            
            stellar_names = data['Name']
            longitude = data['l (deg)']
            latitude = data['b (deg)']
            stellar_distance = data['d (pc)']
            distance_error = data['sigma_d (pc)']
            QH0 = data['QH0/(1e46 s^{-1})']
            stellar_type = data['Type']
            Teff = data['Teff (K)']
            Lbol = data['L_bol (erg s^{-1})']
            g = data['g (cm s^{-2})']


        star_coords = SkyCoord(l=longitude * u.deg, b=latitude * u.deg, distance = stellar_distance * u.pc, frame='galactic')
        #print('Teff: ', Teff)
        xs = star_coords.cartesian.x.value 
        ys = star_coords.cartesian.y.value 
        zs = star_coords.cartesian.z.value 
        
        print('stellar sources have been loaded')
        print('Defining model ...')
        m = Model()
        padding = 0.5
        kmh_dust = os.path.abspath('dust_model/kmh/kmh_lite.hdf5')
        astrodust = os.path.abspath('dust_model/astrodust/astrodust_hyperion.hdf5')
    
        rs = np.linspace(0.0, max_distance + padding, nr + 1)
        phis = np.linspace(0, 2 * np.pi, nphi + 1)
        thetas = np.linspace(0, np.pi, ntheta + 1)

        m.set_spherical_polar_grid(rs, thetas, phis)
        
        t_phi, t_theta, t_r = m.grid.shape
        
        #------------- FILL DUST DENSITY ARRAY ----------------# 
        # Needs to be full from 0 pc -  I think
        
        print('Adding dust density grid to model ...')
        
        dust_density_ordered = np.transpose(dust_density, (2, 1, 0))

        dust_density_filled = np.zeros((t_phi, t_theta, t_r), dtype=np.float32)

        cell_width_cm = (max_distance + padding) / t_r
        idx_69pc = int((69.0 * pc) / cell_width_cm)
        
        remaining_slots = t_r - idx_69pc
        dust_density_filled[:, :, idx_69pc:] = dust_density_ordered[:, :, :remaining_slots]

        dust_density_filled[:, :, :idx_69pc] = 0 #1e-30 

        m.add_density_grid(dust_density_filled, astrodust)
        
        print('Density grid added successfully, adding stellar sources ...')
        
        xs *= pc
        ys *= pc
        zs *= pc
        
        if add_specific_sources == True:
            for source_name in source_names:
                matching_indices = np.where(stellar_names == source_name)[0]
                if len(matching_indices) > 0:
                    idx = matching_indices[0] 
                    
                    print(f"Adding single source: {source_name} at coordinates: X={xs[idx]/pc:.1f} pc, Y={ys[idx]/pc:.1f} pc, Z={zs[idx]/pc:.1f} pc")
                    
                    source = m.add_point_source()
                    source.position = (xs[idx], ys[idx], zs[idx])
                    
                    source.luminosity = float(Lbol[idx]) 
                    
                    temp_val = Teff[idx]
                    if temp_val == '>60000':
                        temp_val = 60000
                    source.temperature = float(temp_val)
        else:
            spectrum_cache = {}

            for i in range(len(stellar_names)):
                if Teff[i] == '>60000':
                    Teff[i] = 60000
                    
                r = np.sqrt(xs[i]**2 + ys[i]**2 + zs[i]**2)
                
                if min_distance <= r <= max_distance:
                    grav = float(g[i])
                    T = float(Teff[i])
                    
                    source = m.add_point_source()
                    source.position = (xs[i], ys[i], zs[i])
                    source.luminosity = float(Lbol[i])
                    
  
                    if T >= 30000:
                        cache_key = (T, grav)
                        if cache_key not in spectrum_cache:
                            spectrum = wmbasicspectra(temperature=T, surface_gravity=grav)
                            nu_grid = spectrum.file_nu
                            fnu_grid = spectrum.file_fnu
                            
                            nu_grid, unique_inds = np.unique(nu_grid, return_index=True)
                            fnu_grid = fnu_grid[unique_inds]
                            
                            spectrum_cache[cache_key] = (nu_grid, fnu_grid)
                        
                        source.spectrum = spectrum_cache[cache_key]
                        print(f" -> Attached {stellar_names[i]} (Atmosphere Template) at r={r/pc:.1f} pc")
                    else:
                        source.temperature = T
                        print(f" -> Attached {stellar_names[i]} (Analytical Blackbody) at r={r/pc:.1f} pc")
                else:
                    print(f" -> Skipped {stellar_names[i]}: Sits outside limits, r = {r/pc:.1f} pc")
                    
            print('Sources addeded, moving on to images and SEDs ...')
            if use_single_channel == True:
               
                print('Adding single image and SED for wavelength range: ', spherex_min_wav, spherex_max_wav)
                image = m.add_peeled_images(sed=False, image=True)
                image.set_inside_observer((0.,0.,0.))
                image.set_wavelength_range(n_wavelength_bins, spherex_min_wav, spherex_max_wav)
                image.set_image_limits(180., -180., -90, 90)
                image.set_image_size(1572, 786)
                sed = m.add_peeled_images(sed=True, image=False)
                sed.set_viewing_angles([60.0],[80.0])
                sed.set_wavelength_range(n_wavelength_bins, spherex_min_wav, spherex_max_wav)
                sed.set_track_origin('basic')

            elif use_spherex_channels == True:
                spherex_bands = [
                    {"name": "Band 1", "min": 0.75, "max": 1.09},
                    {"name": "Band 2", "min": 1.10, "max": 1.62},
                    {"name": "Band 3", "min": 1.63, "max": 2.41},
                    {"name": "Band 4", "min": 2.42, "max": 3.82},
                    {"name": "Band 5", "min": 3.83, "max": 4.41},
                    {"name": "Band 6", "min": 4.42, "max": 5.00}
                ]
                print('Adding SED and images for all spherex channels ...')
                for band in spherex_bands:
                    image = m.add_peeled_images(sed=False, image=True)
                    
                    image.set_inside_observer((0., 0., 0.))
                    image.set_image_limits(180., -180., -90., 90.)
                    image.set_image_size(1572, 786)
                    
                    image.set_wavelength_range(17, band["min"], band["max"])
                    
                    print(f"Configured {band['name']} ({band['min']} - {band['max']} µm) with 17 channels.")
            
                print('Adding SED for spherex channels ...')
                spherex_min_wav = 0.75 # microns
                spherex_max_wav = 5.0
                nbins = 102
                
                sed = m.add_peeled_images(sed=True, image=False)
                sed.set_viewing_angles([60.0],[80.0])
                sed.set_wavelength_range(nbins, spherex_min_wav, spherex_max_wav)
                sed.set_track_origin('basic')
           
           
            print('SEDs and images configured, setting photon counts and iterations ...')
           # Ensure specific energy, density and temperature are outputs
            m.conf.output.output_specific_energy = 'last'
            m.conf.output.output_density = 'last'
       
            m.set_n_initial_iterations(N_iterations)
            m.set_raytracing(True)
            m.set_n_photons(initial=N_photons, imaging=N_photons,
                        raytracing_sources=N_photons, raytracing_dust=N_photons)
        # m.set_n_photons(initial=1e5, imaging=1e5,
                    #    raytracing_sources=1e5, raytracing_dust=1e5)
                    
            N_photons_str = f"{N_photons:g}"
            
            if add_specific_sources == True:
                input_filename = 'local_bubble_astrodust_'+N_photons_str+'_photons_dustmap_allbands_'+specific_source_string+'_sources.rtin'
            else:
                input_filename = 'local_bubble_astrodust_'+N_photons_str+'_photons_dustmap_allbands.rtin'
            m.write(input_filename)
            print(f"Compilation finished for ", input_filename)



sns = ['* zet Pup', '* gam02 Vel']
sss = 'gum_nebula'
create_hyperion_model(remake_dust_map=False, define_model=True, use_single_channel=True, use_spherex_channels=False, add_specific_sources=True, source_names=sns, specific_source_string = sss, spherex_max_wav=5.0, spherex_min_wav=0.75, n_wavelength_bins=102,  N_photons=1e5, N_iterations=5)