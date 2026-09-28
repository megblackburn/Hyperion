from astropy.io import fits
from astropy.table import Table
import numpy as np
import h5py
from hyperion.util.constants import pc
import matplotlib.pyplot as plt
from hyperion.dust import SphericalDust


if not hasattr(np, 'string_'):
    np.string_ = np.bytes_


sourcepath = './dust_model/astrodust/astrodust+PAH_MW_RV3.1.fits'

sourcepath2 = './dust_model/kmh/kmh_lite.hdf5'

with fits.open(sourcepath) as file:
    file.info()
    data = file[1].data
    header = file[1].header
    
    wavelengths = file['WAVELENGTHS'].data
    u = file['LOG10 U'].data
    extinction_mat = file['EXTINCTION'].data
    scattering_mat = file['SCATTERING'].data
    emission_mat = file['EMISSION'].data
    
#print('extinctions: ', extinction_mat)
emission = np.mean(emission_mat, axis=2)
emission = emission.T


#------------- PLOT DUST PROPERTIES ----------------#


fig, ex = plt.subplots(1,1)
ex.plot(wavelengths, extinction_mat[:,1], color='darkred', linewidth=1, linestyle='--', label=r'${\rm Astrodust}$')
ex.plot(wavelengths, extinction_mat[:,2], color='darkblue', linewidth=1, linestyle='--', label=r'${\rm PAH}$')
ex.plot(wavelengths, extinction_mat[:,3], color='black', linewidth=1, label=r'Total', zorder=0)
ex.set_xscale('log')
ex.set_yscale('log')
ex.set_xlabel(r'$\lambda\ [\mu{\rm m}]$', fontsize=12)
ex.set_ylabel(r'$\tau_\lambda/N_{\rm H}\ [{\rm cm}^{2}\ {\rm H}^{-1}]$',
                 fontsize=12)
ex.grid(True, which="both", linestyle="--", alpha=0.5)
ex.legend(fontsize=11)
output_image = "./plots/astrodust_total_extinction.png"
fig.savefig(output_image, dpi=300, bbox_inches='tight')

fig, sca = plt.subplots(1,1)
sca.plot(1/wavelengths, scattering_mat[:,1]/extinction_mat[:,1], color='darkred', linewidth=1, linestyle='--', label=r'${\rm Astrodust}$')
sca.plot(1/wavelengths, scattering_mat[:,2]/extinction_mat[:,2], color='darkblue', linewidth=1, linestyle='--', label=r'${\rm PAH}$')
sca.plot(1/wavelengths, scattering_mat[:,3]/extinction_mat[:,3], color='black', linewidth=1, label=r'Total', zorder=0)
sca.set_xscale('linear')
sca.set_yscale('linear')
sca.set_xlabel(r'$\lambda^{-1}\ [\mu{\rm m}^{-1}]$',fontsize=12)
sca.set_ylabel(r'${\rm Albedo}\ \omega$',fontsize=12)
sca.grid(True, which="both", linestyle="--", alpha=0.5)
sca.legend(fontsize=11)
output_image = "./plots/astrodust_total_albedo.png"
fig.savefig(output_image, dpi=300, bbox_inches='tight')

uinds = np.argmin(np.abs(u-0.2))
fig, em = plt.subplots(1,1)
em.plot(wavelengths, emission_mat[uinds,:,0], color='darkred', linewidth=1, linestyle='--', label=r'${\rm Astrodust}$')
em.plot(wavelengths, emission_mat[uinds,:,1], color='darkblue', linewidth=1, linestyle='--', label=r'${\rm PAH}$')
em.plot(wavelengths, emission_mat[uinds,:,2], color='black', linewidth=1, label=r'Total', zorder=0)
em.set_xscale('log')
em.set_yscale('log')
em.set_xlabel(r'$\lambda\ [\mu{\rm m}]$', fontsize=12)
em.set_ylabel(r'$\lambda I_\lambda/N_{\rm H}\ [{\rm erg}\ {\rm s}^{-1}\ {\rm sr}^{-1}\ {\rm H}^{-1}]$',
                 fontsize=9)
em.grid(True, which="both", linestyle="--", alpha=0.5)
em.legend(fontsize=11)
output_image = "./plots/astrodust_total_emission.png"
fig.savefig(output_image, dpi=300, bbox_inches='tight')



#---------------------------- ADD HYPERION DUST MODEL ----------------------------#
wavelengths /= 1e6
c = 2.989e8 # m/s
frequencies = c/wavelengths
print('len wavelengths = ', len(wavelengths), 'len(frequencies) = ', len(frequencies))

# hyperion needs acsending order for nu so reorder everything, also needs everything to be 1D matrix
sorted_inds = np.argsort(frequencies)
frequencies = frequencies[sorted_inds]
print(np.shape(extinction_mat))
ext = extinction_mat[:,3]
print(np.shape(ext))
ext = ext[sorted_inds]
sca = scattering_mat[:,3]
sca=sca[sorted_inds]
albedo = scattering_mat[:,3]/extinction_mat[:,3]
albedo = albedo[sorted_inds]

print(np.shape(emission_mat[:,:,2]))
print('len nu: ', len(frequencies), ', len var: ', len(u))
emission = emission_mat[:,:,2]
emission = emission.T
#emission=emission[sorted_inds,:]
print(np.shape(emission))

dust = SphericalDust()


dust.optical_properties.nu = frequencies  
dust.optical_properties.chi = ext         
dust.optical_properties.albedo = albedo 

length_f = len(frequencies)
dust.optical_properties.mu = np.linspace(-1.0, 1.0, length_f)  # Length 1000

dust.optical_properties.P1 = np.ones((length_f, length_f))
dust.optical_properties.P2 = np.zeros((length_f, length_f))
dust.optical_properties.P3 = np.zeros((length_f, length_f))
dust.optical_properties.P4 = np.zeros((length_f, length_f))


dust.emissivities.var = 10**u  #  convert from log10 u
dust.emissivities.nu = frequencies          
dust.emissivities.jnu = emission             
dust.emissivities.var_name = 'specific_energy'

dust.optical_properties.extrapolate_nu(1.e9, 1.e17)

#dust.compute_mean_opacities()
#dust.set_lte_emissivities(n_temp=1000, temp_min=0.1, temp_max=100000.0)

output_path = './dust_model/astrodust/astrodust_hyperion.hdf5'
dust.write(output_path)


print('dust model added')

#----------------------------- PLOT HYPERION DUST MODEL TO CHECK ----------------------------#

dust = SphericalDust(output_path)

nu = dust.optical_properties.nu
chi = dust.optical_properties.chi  #  total extinction in cm²/g

wavelengths = 2.998e14 / nu

plt.figure(figsize=(8, 6))
plt.loglog(wavelengths, chi, color='darkblue', linewidth=2, label=r'Total Extinction ($\chi$)')
plt.xlabel(r'Wavelength $\lambda$ ($\mu$m)', fontsize=12)
plt.ylabel(r'Mass Extinction Coefficient $\chi$ (cm$^2$ g$^{-1}$)', fontsize=12)
plt.title('Hyperion Dust Model: Total Extinction Profile', fontsize=14, pad=15)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=11)
output_image = "./plots/hyperion_total_extinction.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')





'''
def inspect_hdf5_columns(name, obj):
    """Callback function to print the paths and shapes of datasets."""
    if isinstance(obj, h5py.Dataset):
        # Calculate size or data types
        shape = obj.shape
        dtype = obj.dtype
        print(f"Column Path:  {name}")
        print(f"  -> Shape:   {shape}")
        print(f"  -> Type:    {dtype}")
        print("-" * 50)
    elif isinstance(obj, h5py.Group):
        # Optional: uncomment the line below if you want to see organizational folders
        # print(f"Group: {name}/")
        pass



with h5py.File(sourcepath2, 'r') as f:
    f.visititems(inspect_hdf5_columns)
    
    
    
    


length_f = len(frequencies)
length_u = len(u)

optical_property_types = [('nu', '<f8'), ('albedo', '<f8'), ('chi','<f8'), ('P1', '<f8', (length_f,)), ('P2', '<f8', (length_f)), ('P3', '<f8', (length_f,)), ('P4', '<f8', (length_f)) ]

optical_properties = np.zeros(length_f, dtype=optical_property_types)
optical_properties['nu'] = frequencies
optical_properties['albedo'] = albedo
optical_properties['chi'] = ext
optical_properties['P1'] = np.ones((length_f, length_f)) # should be isotropic scattering???

emissivities_types = [('nu', '<f8'), ('jnu', '<f8', (91,))] # kmh has 100??
emissivities = np.zeros(length_f,dtype=emissivities_types)
emissivities['nu'] = frequencies
emissivities['jnu'] = emission

emissivity_variable_type = [('specific_energy', '<f8')]
emissivity_variable = np.zeros(length_u, dtype=emissivity_variable_type)
emissivity_variable['specific_energy'] = u

mean_opacities_types = [('temperature', '<f8'), ('specific_energy', '<f8'), ('chi_planck', '<f8'), ('kappa_planck', '<f8'),
                        ('chi_inv_planck', '<f8'), ('kappa_inv_planck', '<f8'), ('chi_rosseland', '<f8'), ('kappa_rosseland', '<f8')]
mean_opacities = np.zeros(length_f, dtype=mean_opacities_types)
mean_opacities['temperature'] = np.logspace(0, 4.0, 1000) # 0 - 10^4 K - dont know if reasonable - what should this be???

scattering_angles_types = [('mu', '<f8')]
scattering_angles = np.zeros(length_f, dtype=scattering_angles_types)
scattering_angles['mu'] = np.linspace(-1.0, 1.0, 1000) # placeholder??? what sbhould this be

output_path = './dust_model/astrodust/astrodust.hdf5'

with h5py.File(output_path, 'w') as f:
    f.attrs['type'] = np.int32(1)
    f.attrs['version'] = np.int32(1) 
    f.create_dataset('optical_properties', data = optical_properties)
    em = f.create_dataset('emissivities', data = emissivities)
    em.attrs['emissvar'] = np.string_('E')
    f.create_dataset('emissivity_variable', data = emissivity_variable)
    f.create_dataset('scattering_angles', data = scattering_angles)
    f.create_dataset('mean_opacities', data = mean_opacities)



dust = SphericalDust(output_path)

dust.set_lte_emissivities(n_temp =1000, temp_min=0.1, temp_max=2000) # dk if this is what we want either - check
dust.write('./dust_model/astrodust/astrodust_hyperion.hdf5')


'''