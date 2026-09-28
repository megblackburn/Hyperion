import numpy as np
import matplotlib.pyplot as plt
from hyperion.model import ModelOutput
import healpy as hp
from scipy.interpolate import RegularGridInterpolator

# 1. Load the model output file
mo = ModelOutput('local_bubble_astrodust_1e7photons_dustmap.rtout')

plot_dust = False
if plot_dust == True:
    # 2. Extract physical quantities
    quantities = mo.get_quantities()

    # 3. Access the dictionary array using the correct 'density' key string
    # quantities.quantities['density'] returns a list of density grids (one per dust type)
    # Index 0 grabs your first dust component ('astrodust')
    dust_density = quantities.quantities['density'][0]  

    # 4. Extract the grid wall edges for geometry matching
    r_walls = quantities.r_wall
    phi_walls = quantities.p_wall

    # 5. Extract a midplane slice (theta index close to the middle)
    theta_idx = dust_density.shape[1] // 2 
    density_slice = dust_density[:, theta_idx, :]

    # 6. Set up a meshgrid using the walls for proper spatial alignment
    R, P = np.meshgrid(r_walls, phi_walls)
    pc_conversion = 3.08567758e18  # 1 pc in cm

    X = R * np.cos(P) / pc_conversion
    Y = R * np.sin(P) / pc_conversion

    # 7. Plot the input dust map
    plt.figure(figsize=(7, 6))
    mesh = plt.pcolormesh(X, Y, np.log10(density_slice + 1e-35), cmap='viridis', shading='flat')
    plt.colorbar(mesh, label='Log10 Dust Density ($g/cm^3$)')
    plt.xlabel('X (pc)')
    plt.ylabel('Y (pc)')
    plt.title('Hyperion Input Dust Map (Midplane Slice)')
    plt.axis('equal')


    plt.savefig('dust_density_midplane_slice.png', dpi=300)


# Load simulation results
# 1. Grab your peeled image sequence using the group keyword argument
# group=0 corresponds to the first image block added in your setup script
image_set = mo.get_image(group=0)

# 3. FIX: Use '.wav' attribute instead of '.wavelength'
wav_idx = 50  
selected_wav = image_set.wav[wav_idx]

# 4. Extract the Stokes I intensity map
# Dimensions are (n_viewing_angles, n_y, n_x, n_wavelengths).
# For an inside observer, n_viewing_angles is exactly 1 (index 0).
intensity = image_set.val[0, :, :, wav_idx]

# 5. Plot the all-sky SPHEREx projection map
plt.figure(figsize=(12, 6))
plt.imshow(np.log10(intensity + 1e-40), 
           extent=[180, -180, -90, 90],  # Matches your custom image limits
           cmap='inferno', 
           origin='lower')

plt.colorbar(label=r'Log10 Intensity ($ergs/s/cm^2/sr/Hz$)')
plt.xlabel(r'Galactic Longitude ($^\circ$)')
plt.ylabel(r'Galactic Latitude ($^\circ$)')
# Note the r'...' prefix below to prevent syntax warnings for '\m'
plt.title(r'Synthetic SPHEREx Map at $\lambda$ = ' + f'{selected_wav:.2f}' + r' $\mu m$')
plt.grid(color='white', linestyle='--', alpha=0.3)

plt.savefig('spherex_allsky_map.png', dpi=300)


# ==========================================================
# FIX FOR LINE 77: Extracting the SED (Second Image Block)
# ==========================================================
# Use group=1 since it is the second peeled configuration block added
sed_set = mo.get_sed(group=1) 

# For an inside observer, sed_set.val has dimensions (n_viewing_angles, n_apertures, n_wavelengths).
# Since there is 1 viewing angle and 1 aperture, index both dimensions at 0.
sed_flux = sed_set.val[0, 0, :] 

# ==========================================================
# FIX FOR LINE 85: Plotting the SED Spectrum Profile
# ==========================================================
plt.figure(figsize=(8, 5))

# Use .wav to grab the wavelength array list
plt.loglog(sed_set.wav, sed_flux, color='black', lw=1.5)

# Use the r'...' prefix to eliminate the 'invalid escape sequence' warning
plt.xlabel(r'Wavelength ($\mu m$)')
plt.ylabel(r'Flux Density ($ergs/s/cm^2/Hz$)')
plt.title('Synthetic Peeled SED Spectrum Profile')
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.savefig('spherex_sed.png', dpi=300)


image_set = mo.get_image(group=0)

# 2. Select the wavelength index you want to view 
# Let's map it roughly to your reference values (0.76 um, 3.3 um, 4.05 um)
# Find closest indexes in your 102 channels:
target_wav = 3.3  # microns
wav_idx = np.argmin(np.abs(image_set.wav - target_wav))
selected_wav = image_set.wav[wav_idx]

# 3. Extract the 2D rectangular image array (Plate Carrée format)
# Hyperion shape: (n_y, n_x). Yours is (786, 1572)
intensity = image_set.val[0, :, :, wav_idx]

# Hyperion output intensity units are ergs/s/cm^2/sr/Hz. 
# Convert to MJy/sr to match the plot if preferred:
# intensity_mjy_sr = intensity * 1e23 * 1e-6 

# 4. Convert Plate Carrée grid into a HEALPix map array
nside = 256  # Defines the output resolution grid density on the sphere
theta_grid = np.linspace(np.pi, 0, intensity.shape[0])  # Latitude rows (Colatitude 0 to pi)
phi_grid = np.linspace(-np.pi, np.pi, intensity.shape[1])  # Longitude columns

# Get the spherical coordinates for all HEALPix pixels
n_hp_pixels = hp.nside2npix(nside)
hp_theta, hp_phi = hp.pix2ang(nside, np.arange(n_hp_pixels))

# Wrap coordinates back to align map orientation
hp_phi = np.where(hp_phi > np.pi, hp_phi - 2 * np.pi, hp_phi)

# Interpolate onto the new HEALPix grid surface
interpolator = RegularGridInterpolator((theta_grid, phi_grid), intensity, 
                                       bounds_error=False, fill_value=0)
healpix_map = interpolator(np.column_stack((hp_theta, hp_phi)))

# 5. Plot using healpy's mollweide viewer
plt.figure(figsize=(8, 5))
hp.mollview(
    np.log10(healpix_map + 1e-40),
    title=f"Synthetic Map at {selected_wav:.2f} $\mu$m",
    cmap="inferno",
    norm="user",
    min=-1.3, max=1.2,  # Set the min/max log bounds to match your reference scale
    unit=r"$\log_{10}(\mathrm{Flux})$",
    hold=True
)
hp.graticule(color="white", alpha=0.3)


plt.savefig('spherex_mollweide_map.png', dpi=300)

# 1. Load the model simulation output
image_set = mo.get_image(group=0)

# Target wavelengths requested
target_wavs = [0.76, 3.3, 4.05]

# 2. Extract array sizes directly from the image values cube
# Shape: (n_viewing_angles, n_y, n_x, n_wavelengths)
_, ny, nx, _ = image_set.val.shape

# 3. Create coordinate meshes matching your set_image_limits(180, -180, -90, 90)
# Matplotlib's mollweide projection requires coordinates in RADIANS
lon_rad = np.linspace(np.pi, -np.pi, nx)   # Galactic Longitude: 180 to -180 deg
lat_rad = np.linspace(-np.pi/2, np.pi/2, ny) # Galactic Latitude: -90 to 90 deg
LON, LAT = np.meshgrid(lon_rad, lat_rad)

# 4. Initialize a clean 1-row, 3-column matplotlib figure layout
fig = plt.figure(figsize=(18, 6))

for i, wav in enumerate(target_wavs):
    # Find the closest matching wavelength index inside Hyperion arrays
    wav_idx = np.argmin(np.abs(image_set.wav - wav))
    actual_wav = image_set.wav[wav_idx]
    
    # Extract the intensity map (already scaled to MJy/sr)
    intensity_mjy_sr = image_set.val[0, :, :, wav_idx]
    
    # Add a subplot with the native mollweide projection activated
    ax = fig.add_subplot(1, 3, i + 1, projection='mollweide')
    
    # Plot using pcolormesh with your explicit log scale limits
    mesh = ax.pcolormesh(
        LON, LAT, 
        np.log10(intensity_mjy_sr + 1e-40), 
        cmap='inferno', 
       #vmin=-1.3, vmax=1.2,
        shading='auto'
    )
    
    # Format gridlines and titles
    ax.grid(color='white', linestyle='--', alpha=0.3)
    ax.set_title(f"{actual_wav:.2f} $\mu$m View", fontsize=12, pad=10)
    
    # Add an individual horizontal colorbar below each column panel
    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar.set_label(r'$\log_{10}(\mathrm{MJy / sr})$', fontsize=10)

plt.tight_layout()

plt.savefig('spherex_mollweide_multi_panel.png', dpi=300)


target_wavs = [0.76, 3.3, 4.05]

# 2. Extract dimensions directly from the raw array shape
# Shape format: (n_viewing_angles, n_y, n_x, n_wavelengths)
_, ny, nx, _ = image_set.val.shape

# 3. Calculate the solid angle per pixel (Omega_pix) in steradians
delta_lon = 360.0 * (np.pi / 180.0)  # 2*pi radians
delta_lat = 180.0 * (np.pi / 180.0)  # pi radians
omega_pix = (delta_lon / nx) * (delta_lat / ny)

# 4. Initialize the 1-row, 3-column flat rectangular grid layout
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, wav in enumerate(target_wavs):
    ax = axes[i]
    
    # Locate the closest matching wavelength channel index
    wav_idx = np.argmin(np.abs(image_set.wav - wav))
    actual_wav = image_set.wav[wav_idx]
    
    # Grab the raw cgs flux array (ergs/s/cm^2/Hz per pixel)
    flux_per_pixel = image_set.val[0, :, :, wav_idx]
    
    # STEP 1: Convert ergs/s/cm^2/Hz to Jy (1 cgs flux = 1e23 Jy)
    flux_jy_per_pixel = flux_per_pixel * 1e23
    
    # STEP 2: Divide by pixel solid angle to convert Jy/pixel to Jy/sr
    surface_brightness_jy_sr = flux_jy_per_pixel / omega_pix
    
    # STEP 3: Convert Jy/sr to MJy/sr (1 Jy = 1e-6 MJy)
    intensity_mjy_sr = surface_brightness_jy_sr * 1e-6
    
    # Plot using raw imshow mapping
    im = ax.imshow(
        np.log10(intensity_mjy_sr + 1e-40),
        cmap='inferno',
        vmin = 1.2* np.min(np.log10(intensity_mjy_sr+1e-40)),  # Dynamic lower limit based on max intensity
        vmax = 0.8* np.max(np.log10(intensity_mjy_sr+1e-40)),  # Dynamic upper limit based on max intensity
      #  vmin=-1.3, vmax=1.2,   # Locked onto your target comparison limits
        extent=[180, -180, -90, 90],
        origin='lower',
        aspect='equal'
    )
    
    # Grid lines and formatting layout
    ax.grid(color='white', linestyle='--', alpha=0.3)
    ax.set_title(f"{actual_wav:.2f} $\mu$m View", fontsize=12, pad=10)
    ax.set_xlabel(r'Galactic Longitude ($^\circ$)')
    
    if i == 0:
        ax.set_ylabel(r'Galactic Latitude ($^\circ$)')

# Position a single shared horizontal colorbar across the bottom
cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.04]) # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'$\log_{10}(\mathrm{MJy / sr})$', fontsize=11)

# Adjust layout to make room for the bottom colorbar
plt.tight_layout() # Leave space at the bottom for the colorbar
plt.savefig('spherex_flat_multi_panel.png', dpi=300)