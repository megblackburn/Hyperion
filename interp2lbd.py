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
    default="(1572,786,2381)::((180,-180),(-90,90),(69,1250))",
    help=
    "python expression for (the shape and) the box in degrees which to export"
)
args = None
in_jupyter = hasattr(sys, "ps1") and not sys.__stdin__.isatty()
if in_jupyter:
    args = input("args").split(" ")
args = parser.parse_args(args)

import numpy as np
from astropy.io import fits

healpix_path = args.healpix_path
if args.output_directory is not None:
    odir, filename_prefix = args.output_directory, ""
else:
    odir, filename_prefix = os.path.split(healpix_path.removesuffix(".fits"))
# shape = (#lon-, #lats-, #radial-) bins
# box = extent in degrees; lon and lat do not include the endpoint, distance does
box = args.box
box_shape, box_extent = box.split("::",
                                  maxsplit=1) if "::" in box else (None, box)
if box_shape is None:
    raise ValueError("`box` must specify the shape of the box")
try:
    # NOTE, `ast.literal_eval` might crash or hang on invalid input but
    # does **not** allow for arbitrary code execution
    box_shape = ast.literal_eval(box_shape)
except SyntaxError:
    raise ValueError("AST parsing failed")
if not isinstance(box_shape, (tuple, list)):  # Poor man's input validation
    raise TypeError(f"invalid shape {box_shape!r}")
try:
    # NOTE, `ast.literal_eval` might crash or hang on invalid input but
    # does **not** allow for arbitrary code execution
    box_extent = ast.literal_eval(box_extent)
except SyntaxError:
    raise ValueError("`box` must specify the extent of a box")
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


# %%
rec = get_sphere(healpix_path)
rec_header = fits.getheader(healpix_path)
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
lon = np.linspace(*box_extent[0], num=box_shape[0], endpoint=False)
lat = np.linspace(*box_extent[1], num=box_shape[1], endpoint=False)
dist = np.linspace(*box_extent[2], num=box_shape[2], endpoint=True)
lon = np.broadcast_to(lon[:, np.newaxis], box_shape[:2])
lat = np.broadcast_to(lat[np.newaxis, :], box_shape[:2])


def interp(data, radii, nside, nest, lon, lat, dist):
    from healpy.pixelfunc import get_interp_weights

    idx, wgt = get_interp_weights(nside, lon, lat, nest=nest, lonlat=True)
    lbd = np.moveaxis(data, -2, -1)
    lbd = (lbd[..., idx, :] * wgt[..., np.newaxis]).sum(axis=-4)

    # Manual radial interpolation
    idx_r = np.searchsorted(radii, dist)
    idx_l = idx_r - 1
    mask = (idx_l < 0) | (idx_r >= radii.size)
    idx_l, idx_r = idx_l.clip(0, radii.size - 1), idx_r.clip(0, radii.size - 1)
    wgt_l, wgt_r = np.abs(radii[idx_l] - dist), np.abs(radii[idx_r] - dist)
    wgt_l, wgt_r = wgt_r / (wgt_l + wgt_r), wgt_l / (wgt_l + wgt_r)
    lbd = (wgt_l * lbd[..., idx_l] + wgt_r * lbd[..., idx_l])
    return np.where(mask, np.nan, lbd)


msg = (
    "NOTE: depending on the number of output voxels,"
    " the interpolation might consume a lot of memory!"
)
print(msg, file=sys.stderr)
hdul = [fits.PrimaryHDU()]

# Interpolate the log-density for higher accuracy
lbd = np.exp(
    interp(np.log(rec.data), rec.radii, rec.nside, rec.nest, lon, lat, dist)
)
nm = "Samples" if lbd.ndim == 4 else "Mean"
# Order axis to be in ZYX order akin to the Green2019 et al. map
hdul += [fits.ImageHDU(np.transpose(lbd).astype(np.float32), name=nm)]
if rec.data_uncertainty is not None:
    lbd_unc = np.sqrt(
        interp(
            rec.data_uncertainty**2, rec.radii, rec.nside, rec.nest, lon, lat,
            dist
        )
    )
    nm = "Std."
    hdul += [fits.ImageHDU(np.transpose(lbd_unc).astype(np.float32), name=nm)]

# %%
copy_header_attrs(hdul[0], rec_header)
for h in hdul[1:]:
    i = 1
    if lbd.ndim == 4:
        h.header.set(f"CTYPE{i}", "samples")
        h.header.set(f"CUNIT{i}", "1")
        h.header.set(f"CRVAL{i}", 0.0)
        h.header.set(f"CDELT{i}", None)  # does not make sense for samples
        h.header.set(f"CRPIX{i}", 0)
        h.header.set(f"CUNIT", rec.units)  # Dust density units
        i += 1
    for i, a in enumerate(("GLON", "GLAT"), start=i):
        h.header.set(f"CTYPE{i}", a)  # Axis names
        h.header.set(f"CUNIT{i}", "deg")  # Axis units
        # Axis offset in units `CUNIT`
        h.header.set(
            f"CRVAL{i}", 0.5 * (box_extent[i - 1][0] + box_extent[i - 1][1])
        )
        # Axis offset in units #pixels
        # Reference point for projections and rotations. Indicates whether `CRVAL`
        # refers to the lower leftmost (0.5) or center (1.0) of the first voxel.
        h.header.set(f"CRPIX{i}", 1.0 + box_shape[i - 1] * 0.5)
        e = (box_extent[i - 1][1] - box_extent[i - 1][0]) / box_shape[i - 1]
        h.header.set(f"CDELT{i}", e)  # Pixel extent in units of `CUNIT`
    i += 1
    h.header.set(f"CTYPE{i}", "distance")
    h.header.set(f"CUNIT{i}", "pc")
    h.header.set(f"CRVAL{i}", box_extent[-1][0])
    # Subtract 1 from the shape when cumputing the volume because we include
    # the endpoint for the distance axis
    h.header.set(
        f"CDELT{i}",
        (box_extent[-1][1] - box_extent[-1][0]) / (box_shape[-1] - 1)
    )
    h.header.set(f"CRPIX{i}", 1.0)  # distance
    copy_header_attrs(h, rec_header)
hdul = fits.HDUList(hdul)
bes = str(box_extent).replace(' ', '')
bes = bes.replace('(', '').replace(')', '').replace(',', '_')
fn = f"{filename_prefix}_{bes}_lbd.fits"
hdul.writeto(os.path.join(odir, fn), overwrite=True)