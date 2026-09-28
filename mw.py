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
   
   

     
    
get_map=False
if get_map:

    extinction_map = fits.open('./edenhofer_dustmaps/spherical/_0_360_-90_90_69_1250_lbd.fits')

    A = extinction_map[1].data 
    print('max and min A: ', np.nanmax(A), np.nanmin(A))

    A = np.nan_to_num(A, nan=0.0)

    ni_factor = 2700 # new val?

    n_h = A * ni_factor

    mu_h = 1.0 * 1.67e-24  # grams maybe 1.4 *
    dust_to_gas_ratio = 0.000605 # mccallum 2025 val

    gas_density = n_h * mu_h
    dust_density = gas_density * dust_to_gas_ratio # A * mu_h, n_h * mu_h * dust_to_gas_ratio
    print('max dust density: ', np.nanmax(dust_density), np.nanmin(dust_density))
    print('max gas density: ', np.nanmax(gas_density), np.nanmin(gas_density))

    np.save('./edenhofer_dustmaps/spherical/dust_dens_spherical.npy', dust_density)
    np.save('./edenhofer_dustmaps/spherical/gas_dens_spherical.npy', gas_density)
else:
    dust_density = np.load('./edenhofer_dustmaps/spherical/dust_dens_spherical.npy')
    gas_density = np.load('./edenhofer_dustmaps/spherical/gas_dens_spherical.npy')


sourcepath = './stellarsources/processed_sources.fits'

# useful constants

pc = 3.0856e18 # in cm

min_distance = 69 * pc # pc
max_distance = 1250 * pc # pc

nphi = 360
ntheta = 180
nr = 1250-69+1


#print('xs: ', xs)
##### MODEL #############

want_model = True
if want_model == True:
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
    m = Model()
    padding = 0.5
    kmh_dust = os.path.abspath('dust_model/kmh/kmh_lite.hdf5')
    astrodust = os.path.abspath('dust_model/astrodust/astrodust_hyperion.hdf5')
  
    rs = np.linspace(0.0, max_distance + padding, nr + 1)
    phis = np.linspace(0, 2 * np.pi, nphi + 1)
    thetas = np.linspace(0, np.pi, ntheta + 1)

    m.set_spherical_polar_grid(rs, thetas, phis)
    
    t_phi, t_theta, t_r = m.grid.shape
    
    dust_density_ordered = np.transpose(dust_density, (2, 1, 0))

    dust_density_filled = np.zeros((t_phi, t_theta, t_r), dtype=np.float32)

    cell_width_cm = (max_distance + padding) / t_r
    idx_69pc = int((69.0 * pc) / cell_width_cm)
    
    remaining_slots = t_r - idx_69pc
    dust_density_filled[:, :, idx_69pc:] = dust_density_ordered[:, :, :remaining_slots]

    dust_density_filled[:, :, :idx_69pc] = 0 #1e-30 
    
   # dust = np.ones((nphi, ntheta, nr)) * 1e-20

    m.add_density_grid(dust_density_filled, astrodust)
    
    xs *= pc
    ys *= pc
    zs *= pc
   # m.add_derived_quantity('gas_density', gas_density.T)
    single_source=False
    if single_source == True:
        source = m.add_point_source()
        source.position = (0.1 * pc, 0.1 * pc, 0.1 * pc)
        source.luminosity = 2.e28
        source.temperature = 30000
    else:
        for i in range(len(stellar_names)):
            if Teff[i] == '>60000':
                Teff[i] = 60000
                
            r = np.sqrt(xs[i]**2 + ys[i]**2 + zs[i]**2)
            
            if min_distance <= r <= max_distance:
                
                grav = float(g[i])
                T = float(Teff[i])
                print('temp: ', T)
                spectrum = wmbasicspectra(temperature=T, surface_gravity=grav)
                
                nu_grid = spectrum.file_nu
                fnu_grid = spectrum.file_fnu
                
                
                nu_grid, unique_inds = np.unique(nu_grid, return_index=True)
                fnu_grid = fnu_grid[unique_inds]
                
            
                
                source = m.add_point_source()
                source.position = (xs[i] , ys[i] , zs[i] )
                source.luminosity = float(Lbol[i]) # convert to erg photon^-1
              #  source.temperature = float(T)
                source.spectrum = (nu_grid, fnu_grid)
                
                print(f" -> Attached {stellar_names[i]} at coordinates: X={xs[i]/pc:.1f} pc, Y={ys[i]/pc:.1f} pc, Z={zs[i]/pc:.1f} pc")
                print('lum: ', Lbol[i])
            else:
                print(f" -> Skipped {stellar_names[i]}: Sits outside the physical grid limits ({stellar_distance[i]:.1f} pc), r = ", r)
    
   # m.set_forced_first_interaction(False)
   # m.set_n_photons(initial=1e6, imaging=1e6)
  #  m.set_wavelength_range(nbins=100, xmin=0.1, xmax=1000.0)
        spherex_min_wav = 0.75 # microns
        spherex_max_wav = 5.0
        
        
      
        image = m.add_peeled_images(sed=False, image=True)
        image.set_inside_observer((0.,0.,0.))
        image.set_wavelength_range(102, spherex_min_wav, spherex_max_wav)
        image.set_image_limits(180., -180., -90, 90)
        image.set_image_size(1572, 786)
        
        #image.set_viewing_angles([60.], [80.])
        #image.set_image_size(400, 400)
        #image.set_image_limits(-1250 * pc, 1250 * pc, -1250 * pc, 1250 * pc
        
        
            # Set up SED for 10 viewing angles
        sed = m.add_peeled_images(sed=True, image=False)
        sed.set_viewing_angles([60.0],[80.0])
        sed.set_wavelength_range(102, spherex_min_wav, spherex_max_wav)
        sed.set_track_origin('basic')
        
        m.conf.output.output_specific_energy = 'last'
        m.conf.output.output_density = 'last'
      #  image.set_viewing_angles(np.linspace(0.0, 180.0, 10), 
          #  np.linspace(0.0, 360.0, 20)) # View from the Galactic Equator
       # image.set_viewing_bins(10,10)
       # image.set_image_size(512,512)
       # image.set_image_limits(xmin=-1250*pc, xmax=1250*pc, ymin=-1250*pc, ymax=1250*pc)
       # image.set_wavelength_range(20, 1, 1000) # Track specific bands (e.g., optical to IR)
    
        m.set_n_initial_iterations(5)
        m.set_raytracing(True)
        m.set_n_photons(initial=1e7, imaging=1e7,
                    raytracing_sources=1e7, raytracing_dust=1e7)
       # m.set_n_photons(initial=1e5, imaging=1e5,
                #    raytracing_sources=1e5, raytracing_dust=1e5)
        
        input_filename = 'local_bubble_astrodust_1e7photons_dustmap.rtin'
        m.write(input_filename)
        print(f"Compilation finished for ", input_filename)




########## PLOTS ##############
plot_model_output =  False
if plot_model_output == True:
   # m = ModelOutput('./local_bubble_all_sources_results_astrodust_all_1e5photons_im2.rtout')
    m = ModelOutput('./local_bubble_all_sources_results_astrodust_all_1e7photons_im2_newdust_spectra.rtout')
    save_string = 'local_bubble_all_sources_astrodust_1e7photons_im2_newdust_spectra_'
    
    inside_observer=True
    if inside_observer == False:
        image = m.get_image(inclination=0, distance=1250 * pc, units='ergs/cm^2/s/Hz') # cant use if inside observer
    else:
        #### inclination 0
        image = m.get_image(units='ergs/cm^2/s/Hz', inclination=0)
        
    fint = np.zeros(image.val.shape[:-1])
    for j,i in np.ndindex(fint.shape):
        fint[j,i] = integrate_loglog(image.nu, image.val[j,i,:])
    
    l = np.radians(np.linspace(180, -180, fint.shape[1]+1))
    b = np.radians(np.linspace(-90,90, fint.shape[0]+1))
    dl = l[1:] - l[:-1]
    db = np.sin(b[1:]) - np.sin(b[:-1])
    DL,DB = np.meshgrid(dl,db)
    area = np.abs(DL * DB)
    
    intensity = fint/area
    
    fourpijnu = round(np.sum(intensity * area), 4)
    fig = plt.figure(figsize=(11,6))
    ax = fig.add_subplot(1,1,1, projection='aitoff')
    mesh = ax.pcolormesh(l,b,intensity, cmap='gist_heat', norm=plt.cm.colors.LogNorm())
    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    fig.colorbar(mesh, cax=cax)
    ax.grid()
    fig.savefig('./plots/'+save_string+'_intensity.pdf')

    waves = [0.1,1,10,100]
    c = 2.989e14
    nus = c/np.array(waves)
    
    for wl, nu in zip(waves, nus):
        nu_ind = np.argmin(np.abs(image.wav-wl))
        fslice = image.val[:,:,nu_ind]
        intensity=fslice/area
        
        fig = plt.figure(figsize=(11,6))
        ax = fig.add_subplot(1,1,1,projection='aitoff')
        mesh = ax.pcolormesh(l,b,intensity,cmap='gist_heat', norm=plt.cm.colors.LogNorm())
        cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
        fig.colorbar(mesh, cax=cax, label=r'Intensity at $\lambda$='+str(wl))
        ax.grid()
        fig.savefig('./plots/'+save_string+'_wavelength_'+str(wl)+'.pdf')
        

    image_data = m.get_image(group=0, distance=None)
    simulated_flux_matrix = image_data.val[0,:,:,0]
    
    plt.figure(figsize=(8,7))
    plt.imshow(simulated_flux_matrix, origin='lower', cmap='hot', norm=plt.cm.colors.LogNorm(), extent = [-1250*pc, 1250*pc, -1250*pc, 1250*pc])
    
    plt.colorbar(label='Surface Brightness (erg/cm$^2$/s/Hz)')
    plt.title('Hyperion Optical & IR')
    plt.savefig('./plots/hyperion_output_spherical_surface_brightness_spectra.png', bbox_inches='tight')
    g = m.get_quantities()
    
    density = g['density'][0].array
    specific_energy = g['specific_energy'][0].array
    temperature = g['temperature'][0].array
    
 #   sources = m.v1_or_v2_sources if hasattr(m, 'v1_or_v2_sources') else m.v2_sources #m.sources
    
    xs = []
    ys = []
    zs = []
    ##for s in sources:
      #  if hasattr(s, 'position'):
      #      xs.append(s.position[0])
       #     ys.append(s.position[1])
        #    zs.append(s.position[2])
            

    
    r_walls, theta_walls, phi_walls = g.r_wall, g.t_wall, g.p_wall
    r_cen = 0.5 * (r_walls[:-1] + r_walls[1:])
    theta_cen = 0.5 * (theta_walls[:-1] + theta_walls[1:])
    phi_cen = 0.5 * (phi_walls[:-1] + phi_walls[1:])

    nside = 64
    npix = hp.nside2npix(nside)
    pix_inds = np.arange(npix)
    thetahp, phihp = hp.pix2ang(nside, pix_inds)
    r_ind = len(r_cen)/2
    
    hp_dens = np.zeros(npix)
    hp_temp = np.zeros(npix)
    hp_em = np.zeros(npix)
    
    hp_stars = np.zeros(npix)
    
    for pix in range(npix):
        th_ind = np.argmin(np.abs(theta_cen-thetahp[pix]))
        ph_ind = np.argmin(np.abs(phi_cen - phihp[pix]))
        r_ind = int(r_ind)
        th_ind = int(th_ind)
        ph_ind = int(ph_ind)
        hp_dens[pix] += density[ph_ind,th_ind, r_ind]
        hp_temp[pix] += temperature[ph_ind, th_ind, r_ind]
        hp_em[pix] += specific_energy[ph_ind, th_ind, r_ind]
        #hp_stars[pix] += 1.0
                
    fig = plt.figure(figsize=(12,15))
    ax1 = fig.add_subplot(3,1,1)
    plt.axes(ax1)
    hp.mollview(np.log10(hp_dens), hold=True, cmap='inferno',unit="log(Density) (g cm^{-3})")
    
    ax2 = fig.add_subplot(3,1,2)
    plt.axes(ax2)
    hp.mollview((hp_temp), hold=True, cmap='hot',unit='Temp (Kelvin)')
  #  plt.savefig('local_bubble_spherical_moll_projections.png', bbox_inches='tight')
    
    ax3 = fig.add_subplot(3,1,3)
    plt.axes(ax3)
    hp.mollview(np.log10(hp_em), hold=True, cmap='hot', unit='Specific Energy (?)')
    plt.savefig('./plots/'+save_string+'moll_projections.png', bbox_inches='tight')
    
    # seperate figures
    lon_moll = np.linspace(-np.pi, np.pi, nphi)     # 360 elements
    lat_moll = np.linspace(-np.pi/2, np.pi/2, ntheta) # 180 elements
    
    LON_mesh, LAT_mesh = np.meshgrid(lon_moll, lat_moll, indexing='ij') # Shape: (360, 180)

    phi_cen = phi_cen-np.pi
    theta_cen = np.pi/2 - theta_cen
    LON_mesh, LAT_mesh = np.meshgrid(phi_cen, theta_cen)
    
    fig_moll = plt.figure(figsize=(11, 6))
    ax_moll = fig_moll.add_subplot(111, projection='mollweide')

    integrated_sky_column = np.sum(density, axis=2)
    mesh_moll = ax_moll.pcolormesh(
        LON_mesh, LAT_mesh, integrated_sky_column.T,  # <--- FIXED
        norm=plt.cm.colors.LogNorm(), 
        cmap='hot', 
        shading='nearest'
    )

   # ax_moll.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(mesh_moll, orientation='horizontal', pad=0.05, label='Integrated Density')
    fig_moll.savefig('./plots/'+save_string+'hyperion_dust_density_mollweide_all.pdf', bbox_inches='tight')
    plt.close()
    
    
    
    fig_moll = plt.figure(figsize=(11, 6))
    ax_moll = fig_moll.add_subplot(111, projection='mollweide')
    integrated_sky_column = np.mean(temperature, axis=2)
    mesh_moll = ax_moll.pcolormesh(
        LON_mesh, LAT_mesh, integrated_sky_column.T,  # <--- FIXED
        norm=plt.cm.colors.LogNorm(), 
        cmap='hot', 
        shading='nearest'
    )

    
   # ax_moll.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(mesh_moll, orientation='horizontal', pad=0.05, label='Mean Temperature')
    fig_moll.savefig('./plots/'+save_string+'hyperion_temperature_mollweide_all.pdf', bbox_inches='tight')
    plt.close()
    
    fig_moll = plt.figure(figsize=(11, 6))
    ax_moll = fig_moll.add_subplot(111, projection='mollweide')
    integrated_sky_column = np.sum(specific_energy, axis=2)
    mesh_moll = ax_moll.pcolormesh(
        LON_mesh, LAT_mesh, integrated_sky_column.T,  # <--- FIXED
        norm=plt.cm.colors.LogNorm(), 
        cmap='hot', 
        shading='nearest'
    )

   # ax_moll.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(mesh_moll, orientation='horizontal', pad=0.05, label='Integrated Specific Energy')
    fig_moll.savefig('./plots/'+save_string+'hyperion_specific_energy_mollweide_all.pdf', bbox_inches='tight')
    plt.close()
    
    

    # Get the wall positions for r and theta
    rw, tw = g.r_wall / pc, g.t_wall

    # Make a 2-d grid of the wall positions (used by pcolormesh)
    R, T = np.meshgrid(rw, tw)

    # Make a plot in (r, theta) space
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    c = ax.pcolormesh(R, T, g['temperature'][0].array[0, :, :])
   # ax.set_xscale('log')
    ax.set_xlim(rw[1], rw[-1])
    ax.set_ylim(tw[0], tw[-1])
    ax.set_xlabel('r (au)')
    ax.set_ylabel(r'$\theta$')
   #ax.set_yticks([np.pi, np.pi * 0.75, np.pi * 0.5, np.pi * 0.25, 0.])
   # ax.set_yticklabels([r'$\pi$', r'$3\pi/4$', r'$\pi/2$', r'$\pi/4$', r'$0$'])
    cb = fig.colorbar(c)
    cb.set_label('Temperature (K)')
    fig.savefig('./plots/'+save_string+'temperature_spherical_rt.png', bbox_inches='tight')
    
    



################### PLOTS OF DUST MAP #############################
want_plots = False
if want_plots:
        
    nphi = 360
    ntheta = 180
    nr = 1250-69

    pc = 3.0856e18 # in cm 

    dmin = 69
    dmax = 1250

    rs = np.linspace(dmin, dmax, nr+1) 
    phis = np.linspace(0, 2*np.pi, nphi+1)
    thetas = np.linspace(0, np.pi, ntheta+1)

    equator_idx = 90  # N_theta // 2
    
    # GAS DENS!!!!!!!!!!!!!!!
    full_plane_slice = gas_density[:, equator_idx, :] # Keeps all 1176 radial bins

    phi_rad = np.linspace(0, 2 * np.pi, 360)

    PHI, R = np.meshgrid(phi_rad, rs, indexing='ij')

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='polar')

    mesh = ax.pcolormesh(
        PHI, R, full_plane_slice.T, 
        norm=plt.cm.colors.LogNorm(), #vmin=1e-26, vmax=1e-22), 
        cmap='inferno', 
        shading='auto'
    )

    ax.set_theta_zero_location('S')  # Centers 0° (Galactic Center) at the bottom
    ax.set_theta_direction(1)        # Counter-clockwise rotation
    ax.set_rlim(0, 1250)             # Extended radial limit to show the full 1.25 kpc space

    ax.set_rticks([200, 400, 600, 800, 1000, 1200])

    ax.plot(0, 0, marker='*', color='cyan', markersize=14, markeredgecolor='black', label='Sun')

    plt.colorbar(mesh, pad=0.1, label=r'Gas Density $\rho$ (g/cm$^3$)')
    plt.title('Full-Scale Polar Map of the Galactic Plane ($b = 0^\circ$)\nTotal 1.25 kpc Spatial Domain', pad=25)
    plt.legend(loc='upper right')
    fig.savefig('./plots/gas_density_polar.png')

    

    integrated_sky_column = np.sum(gas_density, axis=0)

    # Build the coordinate grids matching standard Mollweide bounds
    lon_moll = np.linspace(-np.pi, np.pi, nphi)     # 360 elements
    lat_moll = np.linspace(-np.pi/2, np.pi/2, ntheta) # 180 elements
    LON_mesh, LAT_mesh = np.meshgrid(lon_moll, lat_moll, indexing='ij') # Shape: (360, 180)

    fig_moll = plt.figure(figsize=(11, 6))
    ax_moll = fig_moll.add_subplot(111, projection='mollweide')

    # FIX 2: Add .T to integrated_sky_column to transpose it from (180, 360) to (360, 180)
    mesh_moll = ax_moll.pcolormesh(
        LON_mesh, LAT_mesh, integrated_sky_column.T,  # <--- FIXED
        norm=plt.cm.colors.LogNorm(), 
        cmap='inferno', 
        shading='auto'
    )

    ax_moll.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(mesh_moll, orientation='horizontal', pad=0.05, label='Integrated Gas Column Parameter')
    plt.title('All-Sky Mollweide Projection of Total Gas Distribution', pad=20)
    fig_moll.savefig('./plots/gas_density_mollweide.png', bbox_inches='tight')
    plt.close()
    
    # DUST DENS!!!!!!!!!!!!!!!!!!!
    
    full_plane_slice = dust_density[:, equator_idx, :] # Keeps all 1176 radial bins

    phi_rad = np.linspace(0, 2 * np.pi, 360)

    PHI, R = np.meshgrid(phi_rad, rs, indexing='ij')

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='polar')

    mesh = ax.pcolormesh(
        PHI, R, full_plane_slice.T, 
        norm=plt.cm.colors.LogNorm(), #vmin=1e-26, vmax=1e-22), 
        cmap='hot', 
        shading='auto'
    )

    ax.set_theta_zero_location('S')  # Centers 0° (Galactic Center) at the bottom
    ax.set_theta_direction(1)        # Counter-clockwise rotation
    ax.set_rlim(0, 1250)             # Extended radial limit to show the full 1.25 kpc space

    ax.set_rticks([200, 400, 600, 800, 1000, 1200])

    ax.plot(0, 0, marker='*', color='cyan', markersize=14, markeredgecolor='black', label='Sun')

    plt.colorbar(mesh, pad=0.1, label=r'Gas Density $\rho$ (g/cm$^3$)')
    plt.title('Full-Scale Polar Map of the Galactic Plane ($b = 0^\circ$)\nTotal 1.25 kpc Spatial Domain', pad=25)
    plt.legend(loc='upper right')
    fig.savefig('./plots/dust_density_polar.png')
    
    integrated_sky_column = np.sum(dust_density, axis=0)

    # Build the coordinate grids matching standard Mollweide bounds
    lon_moll = np.linspace(-np.pi, np.pi, nphi)     # 360 elements
    lat_moll = np.linspace(-np.pi/2, np.pi/2, ntheta) # 180 elements
    LON_mesh, LAT_mesh = np.meshgrid(lon_moll, lat_moll, indexing='ij') # Shape: (360, 180)

    fig_moll = plt.figure(figsize=(11, 6))
    ax_moll = fig_moll.add_subplot(111, projection='mollweide')

    # FIX 2: Add .T to integrated_sky_column to transpose it from (180, 360) to (360, 180)
    mesh_moll = ax_moll.pcolormesh(
        LON_mesh, LAT_mesh, integrated_sky_column.T,  # <--- FIXED
        norm=plt.cm.colors.LogNorm(), 
        cmap='hot', 
        shading='auto'
    )

    ax_moll.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(mesh_moll, orientation='horizontal', pad=0.05, label='Integrated Gas Column Parameter')
    plt.title('All-Sky Mollweide Projection of Total Gas Distribution', pad=20)
    fig_moll.savefig('./plots/dust_density_mollweide.png', bbox_inches='tight')
    plt.close()

    print("Both polar and all-sky Mollweide maps updated and saved!")
    
    