import argparse
import ast
import os
import sys
from collections import namedtuple

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("healpix_path", type=str, help="path to a reconstruction")
parser.add_argument(
    "-o",
    "--output-directory",
    dest="output_directory",
    action="store",
    type=str,
    default=None,
    help="output directory in which to place the result (default: result)"
)
parser.add_argument(
    "-b",
    "--box",
    dest="box",
    action="store",
    type=str,
    default=("full", ),
    help="python expression or alias for the box which to export"
)
args = None
in_jupyter = hasattr(sys, "ps1") and not sys.__stdin__.isatty()
if in_jupyter:
    args = input("args").split(" ")
args = parser.parse_args(args)

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits

healpix_path = args.healpix_path
if args.output_directory is not None:
    odir, filename_prefix = args.output_directory, ""
else:
    odir, filename_prefix = os.path.split(healpix_path.removesuffix(".fits"))
box = args.box
box_shape, box_extent = box.split("::",
                                  maxsplit=1) if "::" in box else (None, box)
if box_shape is not None:
    try:
        # NOTE, `ast.literal_eval` might crash or hang on invalid input but
        # does **not** allow for arbitrary code execution
        box_shape = ast.literal_eval(box_shape)
    except SyntaxError:
        raise ValueError("AST parsing failed")
    if not isinstance(box_shape, (tuple, list)):  # Poor man's input validation
        raise TypeError(f"invalid shape {box_shape!r}")
if box_extent.lower().replace(" ", "") in ("leike2020", "leike"):
    box_extent = ((-369.5, 369.5), (-369.5, 369.5), (-269.5, 269.5))
    box_shape = (740, 740, 540) if box_shape is None else box_shape
elif box_extent.lower().replace(" ", "") in ("full", ):
    box_extent = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
    box_shape = (1001, 1001, 1001) if box_shape is None else box_shape
else:
    try:
        # NOTE, `ast.literal_eval` might crash or hang on invalid input but
        # does **not** allow for arbitrary code execution
        box_extent = ast.literal_eval(box_extent)
    except SyntaxError:
        box_aliases = ('leike2020', 'full')
        ve = f"`box` must be a box or a string in {box_aliases}"
        raise ValueError(ve)
    if not isinstance(box_extent, (tuple, list)):  # Poor man's input validation
        raise TypeError(f"invalid box {box_extent!r}")
del args

DustSphere = namedtuple(
    "DustSphere", (
        "data", "nside", "nest", "radii", "coo_bounds", "radii0", "data0",
        "units", "data_uncertainty", "data0_uncertainty"
    )
)


def get_sphere(filepath):
    radii0 = None
    rec_dust0 = None
    rec_uncertainty = None
    rec0_uncertainty = None
    with fits.open(filepath, "readonly") as hdul:
        for hdu in hdul:
            nm = hdu.name.lower()
            if isinstance(hdu, fits.PrimaryHDU) and hdu.data is None:
                continue
            elif nm in ("primary", "mean", "samples", "healpix times distance"):
                dust_density = hdu.data
                nside = hdu.header["NSIDE"]
                nest = hdu.header["ORDERING"].lower().startswith("nest")
                units = hdu.header.get("CUNIT")
                if dust_density.shape[-1] != 12 * nside**2:
                    ve = "invalid shape of dust density {!r}"
                    raise ValueError(ve.format(dust_density.shape))
            elif isinstance(hdu, fits.BinTableHDU):
                if hdu.data.names == ['radial pixel centers']:
                    radii = hdu.data["radial pixel centers"]
                elif hdu.data.names == ['radial pixel boundaries']:
                    coo_bounds = hdu.data["radial pixel boundaries"]
                else:
                    ve = "unrecognized entry in BinTableHDU {!r}"
                    raise ValueError(ve.format(hdu.data.names))
            elif isinstance(hdu, fits.ImageHDU) and (
                nm.startswith("mean of integrated inner") or
                nm.startswith("integrated inner")
            ):
                prfx = "inner density integrated within"
                ctp = hdu.header.get("CTYPE")
                ctp = hdu.header.get("CTYPE1") if ctp is None else ctp
                if ctp.lower().startswith(prfx):
                    radii0 = ctp.lower().removeprefix(prfx)
                    if not radii0.endswith("pc"):
                        ve = "unrecognized units {!r}".format(radii0)
                        raise ValueError(ve)
                    radii0 = radii0.removesuffix("pc")
                    radii0 = float(radii0.strip())
                    rec_dust0 = hdu.data
                    if nest != hdu.header["ORDERING"].lower(
                    ).startswith("nest"):
                        raise ValueError("ordering mismatch")
                    if rec_dust0.shape[-1] != 12 * nside**2:
                        ve = "incompatible shape of dust density {!r}"
                        raise ValueError(ve.format(dust_density.shape))
            elif isinstance(hdu, fits.ImageHDU
                           ) and nm.startswith("std. of integrated inner"):
                prfx = "std. of inner density integrated within"
                ctp = hdu.header.get("CTYPE")
                ctp = hdu.header.get("CTYPE1") if ctp is None else ctp
                if hdu.header["CTYPE"].lower().startswith(prfx):
                    radii0_unc = ctp.lower().removeprefix(prfx)
                    if not radii0_unc.endswith("pc"):
                        ve = "unrecognized units {!r}".format(radii0)
                        raise ValueError(ve)
                    radii0_unc = radii0_unc.removesuffix("pc")
                    radii0_unc = float(radii0_unc.strip())
                    rec0_uncertainty = hdu.data
                    if radii0_unc != radii0:
                        raise ValueError("radii mismatch")
                    if nest != hdu.header["ORDERING"].lower(
                    ).startswith("nest"):
                        raise ValueError("ordering mismatch")
                    if rec0_uncertainty.shape[-1] != 12 * nside**2:
                        ve = "incompatible shape of dust density uncertainty {!r}"
                        raise ValueError(ve.format(rec0_uncertainty.shape))
            elif isinstance(hdu, fits.ImageHDU) and nm.lower() == "std.":
                rec_uncertainty = hdu.data
                if nest != hdu.header["ORDERING"].lower().startswith("nest"):
                    raise ValueError("ordering mismatch")
                if rec_uncertainty.shape[-1] != 12 * nside**2:
                    ve = "incompatible shape of dust density uncertainty {!r}"
                    raise ValueError(ve.format(rec_uncertainty.shape))
            else:
                raise ValueError("unrecognized HDU\n{!r}".format(hdu.header))

    return DustSphere(
        dust_density,
        nside=nside,
        nest=nest,
        radii=radii,
        coo_bounds=coo_bounds,
        radii0=radii0,
        data0=rec_dust0,
        units=units,
        data_uncertainty=rec_uncertainty,
        data0_uncertainty=rec0_uncertainty,
    )


def cart2sph(x, y, z):
    """Converts Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x, y, z : jnp.ndarray of float or float
        Cartesian x-, y- and z-position respectively.

    Returns
    -------
    output: tuple of jnp.ndarray of float or float
        Position in radius (r), azimuthal angle (l) and polar angle (b).
    """
    l = np.arctan2(y, x)
    r1 = x**2 + y**2
    b = np.arctan2(z, np.sqrt(r1))
    r = np.sqrt(r1 + z**2)
    return r, l, b


def get_interp_val(m, theta, phi, nest=False, lonlat=False):
    # Taken from the unreleased version of healpy.pixelfunc.
    # Thanks Jayson Vavrek (@jvavrek) for generlizing the function to multiple
    # healpix maps!
    from healpy.pixelfunc import lonlat2thetaphi, npix2nside
    from healpy import _healpy_pixel_lib as pixlib

    m = np.atleast_2d(m)
    _, npix = m.shape
    nside = npix2nside(npix)
    if lonlat:
        theta, phi = lonlat2thetaphi(theta, phi)
    if nest:
        r = pixlib._get_interpol_nest(nside, theta, phi)
    else:
        r = pixlib._get_interpol_ring(nside, theta, phi)
    p = np.array(r[0:4])
    w = np.array(r[4:8])
    del r
    return np.squeeze(np.sum(m[:, p] * w, axis=1))[()]


def interp_hp2rg(
    new_pos,
    r,
    hp_map,
    nest: bool = True,
    fill_value: float = np.nan,
    *,
    verbose=False
) -> NDArray:
    """Interpolates a HEALPIx times radius sphere to a regular grid.

    Parameters
    ----------
    new_pos : array-like
        Array-like of Cartesian (x, y, z) coordinates. The first axis should be
        the dimension axis, i.e. 3.
    r : array-like
        Radii of the HEALPix spheres. Alternatively, can also be a chart in
        which case, the radii are automatically determined from the chart
    hp_map : array-like
        HEALPix times radius sphere. Broadcasts along the first axes.
    fill_value : float
        Value to insert at `new_pos` postions not within the HEALPix times
        radius sphere.
    """
    r = np.asarray(r)
    if r.ndim != 1:
        raise ValueError("`r` must be one dimensional")
    new_pos = np.asarray(new_pos)
    if new_pos.shape[0] != 3:
        raise ValueError("first axis of `new_pos` must be coordinate axis")

    rlb = np.stack(cart2sph(*new_pos)).reshape(3, -1)
    rg_map = np.full(
        hp_map.shape[:-2] + new_pos.shape[1:], fill_value=fill_value
    )
    for i_l in range(r.size - 1):
        if verbose:
            print(f"{i_l + 1:02d}/{r.size - 1:02d}")
        sel = (r[i_l] <= rlb[0]) & (rlb[0] < r[i_l + 1])
        r_slc, (l_slc, b_slc) = rlb[0, sel], np.degrees(rlb[1:, sel])
        v_l = get_interp_val(
            hp_map[..., i_l, :], l_slc, b_slc, nest=nest, lonlat=True
        )
        v_r = get_interp_val(
            hp_map[..., i_l + 1, :], l_slc, b_slc, nest=nest, lonlat=True
        )
        # DIY radial interpolation with broadcasting
        w_r = (r_slc - r[i_l]) / (r[i_l + 1] - r[i_l])
        assert np.all(w_r >= 0.) and np.all(w_r <= 1.)
        rg_map[...,
               sel.reshape(new_pos.shape[1:])] = (1. - w_r) * v_l + w_r * v_r
    return rg_map


# %%
rec = get_sphere(healpix_path)
rec_header = fits.getheader(healpix_path)

max_extent = int(np.ceil(rec.coo_bounds[-1]))
b = []
for ext_l, ext_r in box_extent:
    ext_l = int(np.sign(ext_l)) * max_extent if np.isinf(ext_l) else ext_l
    ext_r = int(np.sign(ext_r)) * max_extent if np.isinf(ext_r) else ext_r
    b.append((ext_l, ext_r))
box_extent = tuple(b)

os.makedirs(odir, exist_ok=True)

PRESERVED_HEADER_ATTRS = ("AUTHOR", "DATE", "VERSION", "REF", "CONTACT")


def copy_header_attrs(hdu_new, hdu_old):
    hdu_old_header = hdu_old if isinstance(
        hdu_old, fits.Header
    ) else hdu_old.header
    for k in PRESERVED_HEADER_ATTRS:
        if k in hdu_old_header:
            hdu_new.header.set(k, hdu_old_header[k])


# %%
new_pos = np.stack(
    np.meshgrid(
        *(
            np.linspace(el, er, s, endpoint=True)
            for (el, er), s in zip(box_extent, box_shape)
        ),
        indexing="ij"
    )
)

msg = (
    "NOTE: depending on the numbe of output voxels,"
    " the interpolation might consume a lot of memory!"
)
print(msg, file=sys.stderr)
hdul = [fits.PrimaryHDU()]

print("interpolating mean/samples values to RG...", file=sys.stderr)
rg = np.exp(
    interp_hp2rg(new_pos, rec.radii, np.log(rec.data), rec.nest, verbose=True)
)
nm = "Samples" if rg.ndim == 4 else "Mean"
# Order axis to be in ZYX order akin to the Green2019 et al. map
hdul += [fits.ImageHDU(np.transpose(rg).astype(np.float32), name=nm)]
if rec.data_uncertainty is not None:
    print("interpolating std. values to RG...", file=sys.stderr)
    rg_unc = np.sqrt(
        interp_hp2rg(
            new_pos, rec.radii, rec.data_uncertainty**2, rec.nest, verbose=True
        )
    )
    nm = "Std."
    hdul += [fits.ImageHDU(np.transpose(rg_unc).astype(np.float32), name=nm)]

copy_header_attrs(hdul[0], rec_header)
for h in hdul[1:]:
    i = 1
    if rg.ndim == 4:
        h.header.set(f"CTYPE{i}", f"samples")
        h.header.set(f"CUNIT{i}", f"1")
        h.header.set(f"CRVAL{i}", 0.0)
        h.header.set(f"CDELT{i}", 1.0)  # does not make sense for samples
        h.header.set(f"CRPIX{i}", 1.0)
        i += 1
    for i, a in enumerate("XYZ", start=i):
        h.header.set(f"CTYPE{i}", f"{a}")  # Axis names
        h.header.set(f"CUNIT{i}", f"pc")  # Axis units
        # Subtract 1 from the shape when cumputing the volume because we include
        # the endpoint in the box
        off = -1 - (rg.ndim == 4)
        dvol = (box_extent[i + off][1] -
                box_extent[i + off][0]) / (box_shape[i + off] - 1)
        h.header.set(f"CDELT{i}", dvol)  # Pixel extent in units of `CUNIT`
        # Offset in units `CUNIT`
        h.header.set(f"CRVAL{i}", box_extent[i + off][0])
        # Reference point for projections and rotations. Indicates whether
        # `CRVAL`refers to the lower leftmost (0.5) or center (1.0) of the first
        # voxel.
        h.header.set(f"CRPIX{i}", 1.0)
    h.header.set(f"CUNIT", rec.units)  # Dust density units
    copy_header_attrs(h, rec_header)
hdul = fits.HDUList(hdul)
bes = str(box_extent).replace(' ', '')
bes = bes.replace('(', '').replace(')', '').replace(',', '_')
fn = f"{filename_prefix}_{bes}_xyz.fits"
hdul.writeto(os.path.join(odir, fn), overwrite=True)