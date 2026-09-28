import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


spherical = False

if spherical == True:
    fits_file = './edenhofer_dustmaps/spherical/_180_-180_-90_90_69_1250_lbd.fits'

    with fits.open(fits_file) as hdul:
        hdul.info() 
        dust_cube = hdul[1].data 

    print("Actual loaded dust map shape:", dust_cube.shape)


    mid_lat_idx = dust_cube.shape[1] // 2
    galactic_plane_slice = dust_cube[:, mid_lat_idx, :]
    
    galactic_plane_slice = np.nansum(dust_cube, axis=1)

    print('max and min of interpolated dust map:', np.nanmax(galactic_plane_slice), np.nanmin(galactic_plane_slice))



    num_radial_bins = galactic_plane_slice.shape[0]     
    num_longitude_bins = galactic_plane_slice.shape[1] 

    dr = (1250.0 - 69.0) / num_radial_bins

    galactic_plane_slice = galactic_plane_slice * 2.8 / dr
    print('max and min of interpolated dust map:', np.nanmax(galactic_plane_slice), np.nanmin(galactic_plane_slice))
    vmax_99_9 = np.nanpercentile(galactic_plane_slice, 99.9)
    print(f"Raw Map scaled to A_V/pc. 99.9% Quantile saturation set to: {vmax_99_9:.5f} mag/pc")


    distances_pc = np.linspace(69, 1250, num_radial_bins)
    longitudes_rad = np.linspace(np.pi, -np.pi, num_longitude_bins)


    Theta, R = np.meshgrid(longitudes_rad, distances_pc)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    mesh = ax.pcolormesh(Theta, R, galactic_plane_slice, cmap='grey_r', shading='nearest',vmin=0, vmax = vmax_99_9)#, vmin=0, vmax=1.6)

    ax.set_title("Edenhofer et al 2024 Dust Map")
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.1, shrink=0.7)
    cbar.set_label(' $A_V / \mathrm{pc}$ [mag/pc]', fontsize=12)


    output_plot_path = './edenhofer_dustmaps/spherical/galactic_plane_fixed.png'
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"Corrected plot successfully saved to: {output_plot_path}")


#---------------------------- CARTESIAN --------------------------------#
fits_file = './edenhofer_dustmaps/cartesian/_-1250_1250_-1250_1250_-1250_1250_xyz.fits'

with fits.open(fits_file) as hdul:
    hdul.info() 
    dust_cube = hdul[1].data  # Expected shape format: (Z, Y, X) or (X, Y, Z) depending on FITS setup

print("Actual loaded Cartesian dust map shape:", dust_cube.shape)


galactic_plane_projection_xy = np.nansum(dust_cube, axis=0)
galactic_plane_projection_xz = np.nansum(dust_cube, axis=1)
galactic_plane_projection_yz = np.nansum(dust_cube, axis=2)

num_y_bins = galactic_plane_projection_xz.shape[0]     
num_x_bins = galactic_plane_projection_xz.shape[1] 


dx = 2500.0 / num_x_bins
dy = 2500.0 / num_y_bins
dz = 2500.0 / dust_cube.shape[0]

galactic_plane_projection_xz *= 2.8 #/ dz 
galactic_plane_projection_yz *= 2.8 #/ dz
galactic_plane_projection_xy *= 2.8 #/ dz

print('Max and min of Cartesian projection map:', np.nanmax(galactic_plane_projection_xy), np.nanmin(galactic_plane_projection_xy))

vmax_99_9 = np.nanpercentile(galactic_plane_projection_xy, 99.9)
print(f"99.9% Quantile saturation set to: {vmax_99_9:.5f}")

x_pc = np.linspace(-1250, 1250, num_x_bins)
y_pc = np.linspace(-1250, 1250, num_y_bins)

X, Y = np.meshgrid(x_pc, y_pc)

fig, (a1, a2, a3) = plt.subplots(3,1, sharex = True, gridspec_kw={'height_ratios': [25, 8, 8]},  figsize=(10, 15))

mesh = a1.pcolormesh(X, Y, galactic_plane_projection_xy, cmap='grey_r', shading='nearest', vmin=0, vmax=vmax_99_9)

a1.set_xlabel("X [pc]", fontsize=12)
a1.set_ylabel("Y [pc]", fontsize=12)
a1.set_aspect('equal') # Keep the spatial scale 1:1

cbar = plt.colorbar(mesh, ax=a1, orientation='vertical', pad=0.05, shrink=0.7)
cbar.set_label('$A(V)$', fontsize=12)

mesh2 = a2.pcolormesh(X, Y, galactic_plane_projection_xz, cmap='grey_r', shading='nearest', vmin=0, vmax=vmax_99_9*3)

a2.set_xlabel("X [pc]", fontsize=12)
a2.set_ylabel("Z [pc]", fontsize=12)
a2.set_ylim(-400,400)
a2.set_aspect('equal') # Keep the spatial scale 1:1

cbar = plt.colorbar(mesh2, ax=a2, orientation='vertical', pad=0.05, shrink=0.7)
cbar.set_label('$A(V)$', fontsize=12)

mesh3 = a3.pcolormesh(X, Y, galactic_plane_projection_yz, cmap='grey_r', shading='nearest', vmin=0, vmax=vmax_99_9*3)

a3.set_xlabel("Y [pc]", fontsize=12)
a3.set_ylabel("Z [pc]", fontsize=12)
a3.set_ylim(-400, 400)
a3.set_aspect('equal') # Keep the spatial scale 1:1

cbar = plt.colorbar(mesh3, ax=a3, orientation='vertical', pad=0.05, shrink=0.7)
cbar.set_label('$A(V)$', fontsize=12)

output_plot_path = './edenhofer_dustmaps/cartesian/galactic_plane_cartesian_z_proj.png'
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"Projection plot successfully saved to: {output_plot_path}")
