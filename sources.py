from astropy.io import fits
from astropy.table import Table
import numpy as np
import h5py
from hyperion.util.constants import pc
import matplotlib.pyplot as plt

# source path
sourcepath = './ionising_sources/ionizing_sources.fits'

# useful constants

pc = 3.0856e18 # in cm

# useful function??
def spherical_to_cartesian(l,b,d):
    x = d * np.cos(b) * np.cos(l)
    y = d * np.cos(b) * np.sin(l)
    z = d * np.sin(b)
    return x,y,z

with fits.open(sourcepath) as file:
    file.info()
    data = file[1].data
    header = file[1].header
    
    names = data['Name']
    longitude = data['l (deg)']
    latitude = data['b (deg)']
    distance = data['d (pc)']
    distance_error = data['sigma_d (pc)']
    QH0 = data['QH0/(1e46 s^{-1})']
    stellar_type = data['Type']
    Teff = data['Teff (K)']

theta = 90 + latitude 
phi = longitude

theta = np.radians(theta)
phi = np.radians(phi)
#distance /= pc


x, y, z = spherical_to_cartesian(phi, theta, distance)
xsol,ysol,zsol = spherical_to_cartesian(0,0,0)

sources =  np.column_stack((names, stellar_type))
#print(np.shape(sources))

def inbetweenclass(logL1, logL2):
    logL = (logL1 + logL2)/2
    return logL

def inbetweentype(logL1, logL2):
    logL = (logL1 + logL2)/2
    return logL
def t9p2(logL1, logL2):
    logL = logL1 + 0.4 * (logL2-logL1)
    return logL

classes = [3,4,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5]

LI = [5.99,5.93,5.87,5.84,5.81,5.78,5.75,5.72,5.68,5.65,5.61,5.57]
LIII = [5.96,5.85,5.73,5.67,5.61,5.54,5.48,5.42,5.35,5.28,5.21,5.15]
LV = [5.84,5.67,5.49,5.41,5.32,5.23,5.14,5.05,4.96,4.86,4.77,4.68]

gI = [3.73,3.65,3.57,3.52,3.48,3.44,3.40,3.36,3.32,3.28,3.23,3.19]
gIII = [3.77,3.73,3.69,3.67,3.65,3.63,3.61,3.59,3.57,3.55,3.53,3.51]
gV = [3.92,3.92,3.92,3.92,3.92,3.92,3.92,3.92,3.92,3.92,3.92,3.92]

#LII = inbetweenclass(LI, LIII)
#LIV = inbetweenclass(LI, LV)



BASE_GRID = {
    'O3':   {'V': (LV[0], 49.58), 'III': (LIII[0], 49.70), 'I': (LI[0], 49.73)},
    'O4':   {'V': (LV[1], 49.40), 'III': (LIII[1], 49.53), 'I': (LI[1], 49.61)},
    'O5':   {'V': (LV[2], 49.20), 'III': (LIII[2], 49.36), 'I': (LI[2], 49.49)},
    'O5.5':   {'V': (LV[3], 49.20), 'III': (LIII[3], 49.36), 'I': (LI[3], 49.49)},
    'O6':   {'V': (LV[4], 48.97), 'III': (LIII[4], 49.16), 'I': (LI[4], 49.34)},
    'O6.5':   {'V': (LV[5], 49.20), 'III': (LIII[5], 49.36), 'I': (LI[5], 49.49)},
    'O7':   {'V': (LV[6], 48.72), 'III': (LIII[6], 48.94), 'I': (LI[6], 49.20)},
    'O7.5':   {'V': (LV[7], 49.20), 'III': (LIII[7], 49.36), 'I': (LI[7], 49.49)},
    'O8':   {'V': (LV[8], 48.44), 'III': (LIII[8], 48.72), 'I': (LI[8], 49.07)},
    'O8.5':   {'V': (LV[9], 49.20), 'III': (LIII[9], 49.36), 'I': (LI[9], 49.49)},
    'O9':   {'V': (LV[10], 48.11), 'III': (LIII[10], 48.46), 'I': (LI[10], 48.91)},
    'O9.5': {'V': (LV[11], 47.84), 'III': (LIII[11], 48.24), 'I': (LI[11], 48.80)},
}

BASE_GRID = {
    'O3':   {'V': (LV[0], 49.58, gV[0]),   'III': (LIII[0], 49.70, gIII[0]),   'I': (LI[0], 49.73, gI[0])},
    'O4':   {'V': (LV[1], 49.40, gV[1]),   'III': (LIII[1], 49.53, gIII[1]),   'I': (LI[1], 49.61, gI[1])},
    'O5':   {'V': (LV[2], 49.20, gV[2]),   'III': (LIII[2], 49.36, gIII[2]),   'I': (LI[2], 49.49, gI[2])},
    'O5.5': {'V': (LV[3], 49.20, gV[3]),   'III': (LIII[3], 49.36, gIII[3]),   'I': (LI[3], 49.49, gI[3])},
    'O6':   {'V': (LV[4], 48.97, gV[4]),   'III': (LIII[4], 49.16, gIII[4]),   'I': (LI[4], 49.34, gI[4])},
    'O6.5': {'V': (LV[5], 49.20, gV[5]),   'III': (LIII[5], 49.36, gIII[5]),   'I': (LI[5], 49.49, gI[5])},
    'O7':   {'V': (LV[6], 48.72, gV[6]),   'III': (LIII[6], 48.94, gIII[6]),   'I': (LI[6], 49.20, gI[6])},
    'O7.5': {'V': (LV[7], 49.20, gV[7]),   'III': (LIII[7], 49.36, gIII[7]),   'I': (LI[7], 49.49, gI[7])},
    'O8':   {'V': (LV[8], 48.44, gV[8]),   'III': (LIII[8], 48.72, gIII[8]),   'I': (LI[8], 49.07, gI[8])},
    'O8.5': {'V': (LV[9], 49.20, gV[9]),   'III': (LIII[9], 49.36, gIII[9]),   'I': (LI[9], 49.49, gI[9])},
    'O9':   {'V': (LV[10], 48.11, gV[10]), 'III': (LIII[10], 48.46, gIII[10]), 'I': (LI[10], 48.91, gI[10])},
    'O9.5': {'V': (LV[11], 47.84, gV[11]), 'III': (LIII[11], 48.24, gIII[11]), 'I': (LI[11], 48.80, gI[11])},
}

def parse_and_get_baseline(input_string):
    stype = str(input_string).strip().replace(':', '').replace(' ', '')
    
    if '-' in stype:
        stype = stype.split('-')[0]
        
    if 'O' in stype:
        stype = 'O' + stype.split('O')[-1]
    else:
        return BASE_GRID['O7']['V']
        
    spec_part = None
    lum_part = None
    
    for idx, char in enumerate(stype):
        if char in ['V', 'I']:
            spec_part = stype[:idx]
            lum_part = stype[idx:]
            break

    if lum_part is None:
        spec_part = stype
        lum_part = 'V'
        
    for suffix in ['nan', 'na', 'n', 'f', '((f))', '(f)', 'p', 'e']:
        lum_part = lum_part.replace(suffix, '')
        spec_part = spec_part.replace(suffix, '')
        
    if lum_part == '':
        lum_part = 'V'
        
    if lum_part in ['I', 'Ia', 'Ib', 'Iab']:
        lum_part = 'I'
        
    def get_class_vals(s_type, l_class):
        if l_class in ['V', 'III', 'I']:
            return BASE_GRID[s_type][l_class]
        elif l_class == 'IV':
            return tuple((np.array(BASE_GRID[s_type]['V']) + np.array(BASE_GRID[s_type]['III'])) / 2.0)
        elif l_class == 'II':
            return tuple((np.array(BASE_GRID[s_type]['III']) + np.array(BASE_GRID[s_type]['I'])) / 2.0)
        else:
            return BASE_GRID[s_type]['V']

    if spec_part in BASE_GRID:
        return get_class_vals(spec_part, lum_part)
        
    try:
        subclass_num = float(spec_part[1:])
        if subclass_num % 1.0 == 0.5:
            floor_type = f"O{int(subclass_num)}"
            ceil_type = f"O{int(subclass_num) + 1}"
            if floor_type in BASE_GRID and ceil_type in BASE_GRID:
                v_floor = np.array(get_class_vals(floor_type, lum_part))
                v_ceil = np.array(get_class_vals(ceil_type, lum_part))
                return tuple((v_floor + v_ceil) / 2.0)
    except ValueError:
        pass
        
    if spec_part == 'O9.2':
        return tuple(np.array(get_class_vals('O9', lum_part)) + 0.4 * (np.array(get_class_vals('O9.5', lum_part)) - np.array(get_class_vals('O9', lum_part))))
    elif spec_part == 'O9.7':
        return get_class_vals('O9.5', lum_part)
    
    return BASE_GRID['O7']['V']

def calculate_total_luminosity(Q_actual, Teff, stellar_type_string):

    if Q_actual < 100: 
        Q_actual = 10**Q_actual
        

    logL_base, logQ_base, logg_base = parse_and_get_baseline(stellar_type_string)
    
    L_base_solar = 10**logL_base
    #Q_base = 10**logQ_base
    
    Lsol = 3.828e26 # watts
    Lsol_erg = 3.828e33 # erg/s
    L_actual_solar = L_base_solar * Lsol  #(Q_actual / Q_base)
    L_actual_ergs = L_base_solar * Lsol_erg
    
    g_cgs = 10**logg_base
    
    return L_actual_solar, L_actual_ergs, g_cgs
calculated_l_sol = []
calculated_l_erg = []
calculated_g_cgs = []

#print('stellar_type: ', stellar_type)
for i in range(len(stellar_type)):
    try:

        current_stype = str(stellar_type[i]).strip()
        current_qh0   = float(QH0[i])
        if Teff[i] == '>60000':
            Teff[i] = 60000
        current_teff  = float(Teff[i])
        
        l_sol, l_erg, g_cgs = calculate_total_luminosity(current_qh0, current_teff, current_stype)
        
    except Exception as e:
        print(f"--> Warning: Row {i} failed parsing target '{stellar_type[i]}'. Error: {e}")
        l_sol, l_erg, g_cgs = 10**4.81, (10**4.81)*3.828e33, 10**3.92 
        
    calculated_l_sol.append(l_sol)
    calculated_l_erg.append(l_erg)
    calculated_g_cgs.append(g_cgs)

calculated_l_sol = np.array(calculated_l_sol)
calculated_l_erg = np.array(calculated_l_erg)
calculated_g_cgs = np.array(calculated_g_cgs)

columns = [
    fits.Column(name='Name', format='23A', array=names),
    fits.Column(name='l (deg)', format='D', array=longitude),
    fits.Column(name='b (deg)', format='D', array=latitude),
    fits.Column(name='d (pc)', format='D', array=distance),
    fits.Column(name='sigma_d (pc)', format='D', array=distance_error),
    fits.Column(name='QH0/(1e46 s^{-1})', format='D', array=QH0),
    fits.Column(name='Type', format='8A', array=stellar_type),
    fits.Column(name='Teff (K)', format='8A', array=Teff),
    
    fits.Column(name='L_bol (erg s^{-1})', format='D', array=calculated_l_erg),
    fits.Column(name='L_bol (Lsun)', format='D', array=calculated_l_sol),
    fits.Column(name='g (cm s^{-2})', format='D', array=calculated_g_cgs)
]

new_table_hdu = fits.BinTableHDU.from_columns(columns, header=header)

primary_hdu = fits.PrimaryHDU()
new_hdul = fits.HDUList([primary_hdu, new_table_hdu])

output_fits_path = '/sharedscratch/mgb27/Hyperion/stellarsources/processed_sources.fits'
new_hdul.writeto(output_fits_path, overwrite=True)
print('written the file')

np.savetxt('./stellarsources/names.txt', sources, fmt='%s')
'''
dat = Table.read(sourcepath, hdu=1)

print(dat.colnames)
print(dat)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x,y,z, c='darkred')
ax.scatter(xsol,ysol,zsol, c='goldenrod')
fig.savefig('./plots/sources.png')

   # print('header = ', header)
'''    
    
