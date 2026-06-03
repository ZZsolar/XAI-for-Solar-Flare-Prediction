"""
This is the code for calculating SHARP keys with the codes provided by Monica Bobra (github @Monica Bobra).
This script contains all the functions and the functions only for calculating the SHARP keys.
"""


import scipy.ndimage
import numpy as np
import math
from skimage.measure import block_reduce

radsindeg = np.pi/180.
munaught  = 0.0000012566370614



def compute_abs_flux_masked(bz, bz_err, rsun_ref, rsun_obs, cdelt1_arcsec, mask_map):
    """function: compute_abs_flux

    This function computes the total unsigned flux in units of G/cm^2.
    It also returns the number of pixels used in this calculation in the keyword CMASK.

    To compute the unsigned flux, we simply calculate
       flux = surface integral [(vector Bz) dot (normal vector)],
            = surface integral [(magnitude Bz)*(magnitude normal)*(cos theta)].

    However, since the field is radial, we will assume cos theta = 1.
    Therefore, the pixels only need to be corrected for the projection.

    To convert G to G*cm^2, simply multiply by the number of square centimeters per pixel:
       (Gauss/pix^2)(CDELT1)^2(RSUN_REF/RSUN_OBS)^2(100.cm/m)^2
       =Gauss*cm^2
    """
    valid_mask = (~np.isnan(bz)) & (mask_map != 0)
    bz_valid = bz[valid_mask]
    bz_err_valid = bz_err[valid_mask]
    sum_flux = np.sum(np.abs(bz_valid))
    err_flux = np.sum(bz_err_valid ** 2)
    count_mask = bz_valid.size
    scale_factor = (cdelt1_arcsec ** 2) * (rsun_ref / rsun_obs) ** 2 * 100.0 ** 2
    mean_vf = sum_flux * scale_factor
    mean_vf_err = np.sqrt(err_flux) * abs(scale_factor)

    return [mean_vf, mean_vf_err, count_mask]


def compute_bh(bx, by, bz, bx_err, by_err, nx, ny):
    """function: compute_bh

    This function calculates B_h, the horizontal field, in units of Gauss.
    (The magnetic field has native units of Gauss since the filling factor = 1).
    """
    valid_mask = ~np.isnan(bx) & ~np.isnan(by)
    bh = np.zeros((ny, nx))
    bh_err = np.zeros((ny, nx))
    bh[valid_mask] = np.sqrt(bx[valid_mask]**2 + by[valid_mask]**2)
    bx_valid = bx[valid_mask]
    by_valid = by[valid_mask]
    bx_err_valid = bx_err[valid_mask]
    by_err_valid = by_err[valid_mask]

    bh_err[valid_mask] = np.sqrt(
        (bx_valid**2 * bx_err_valid**2 + by_valid**2 * by_err_valid**2) / bh[valid_mask]**2
    )

    # 对无效数据设置为 NaN
    bh[~valid_mask] = np.nan
    bh_err[~valid_mask] = np.nan

    return [bh, bh_err]


def compute_gamma_masked(bz, bh, bz_err, bh_err, mask_map):
    """function: compute_gamma

    This function computes the inclination of the horizontal field (relative to the radial field).
    Error analysis calculations are done in radians (since derivatives are only true in units of radians),
    and multiplied by (180./PI) at the end for consistency in units.
    """
    valid_mask = (
        (~np.isnan(bz)) &
        (~np.isnan(bz_err)) &
        (~np.isnan(bh)) &
        (~np.isnan(bh_err)) &
        (bz != 0) &
        (bh >= 100) &
        (mask_map != 0)
    )
    bh_valid = bh[valid_mask]
    bz_valid = bz[valid_mask]
    bh_err_valid = bh_err[valid_mask]
    bz_err_valid = bz_err[valid_mask]

    # Calculate the inclination in degrees
    sum_gamma = np.sum(np.abs(np.arctan(bh_valid / np.abs(bz_valid))) * (180. / np.pi))

    # Calculate error using the formula provided
    err = np.sum(
        (1 / (1 + (bh_valid ** 2 / bz_valid ** 2))) ** 2 * 
        ((bh_err_valid ** 2 / bz_valid ** 2) + 
         (bh_valid ** 2 * bz_err_valid ** 2 / bz_valid ** 4))
    )

    count_mask = np.sum(valid_mask)
    mean_gamma = sum_gamma / count_mask if count_mask > 0 else np.nan
    mean_gamma_err = (np.sqrt(err) / count_mask) * (180. / np.pi) if count_mask > 0 else np.nan

    return [mean_gamma, mean_gamma_err]


def compute_bt(bx, by, bz, bx_err, by_err, bz_err, nx, ny):
    """function: compute_bt

    This function calculates B_t, the total field, in units of Gauss.
    (The magnetic field has native units of Gauss since the filling factor = 1).
    """
    valid_mask = ~np.isnan(bx) & ~np.isnan(by) & ~np.isnan(bz)
    bt = np.zeros((ny, nx))
    bt_err = np.zeros((ny, nx))

    bt[valid_mask] = np.sqrt(bx[valid_mask]**2 + by[valid_mask]**2 + bz[valid_mask]**2)
    bx_valid = bx[valid_mask]
    by_valid = by[valid_mask]
    bz_valid = bz[valid_mask]
    bx_err_valid = bx_err[valid_mask]
    by_err_valid = by_err[valid_mask]
    bz_err_valid = bz_err[valid_mask]

    bt_err[valid_mask] = np.sqrt(
        (bx_valid**2 * bx_err_valid**2) +
        (by_valid**2 * by_err_valid**2) +
        (bz_valid**2 * bz_err_valid**2)
    ) / bt[valid_mask]

    bt[~valid_mask] = np.nan
    bt_err[~valid_mask] = np.nan

    return [bt, bt_err]


def computeBtderivative_masked(bt, bt_err, nx, ny, mask_map):
    """function: computeBtderivative

    This function computes the derivative of the total field, or sqrt[(dB_total/dx)^2 + (dB_total/dy)^2].
    The native units in the series hmi.sharp_720s and hmi.sharp_cea_720s are in Gauss/pixel.

    Here are the steps to convert from Gauss/pixel to Gauss/Mm:
    The units of the magnetic field, or dB_total, are in Gauss.
    The units of length, i.e. dx or dy, are in pixels.
    The units of dB_total/dx or dB_total/dy = (Gauss/pix)(pix/arcsec)(arsec/meter)(meter/Mm), or
                                            = (Gauss/pix)(1/cdelt1_arcsec)(RSUN_OBS/RSUN_REF)(1000000)
                                            = Gauss/Mm
    In other words, multiply MEANGBT by a factor of (1/cdelt1_arcsec)*(RSUN_OBS/RSUN_REF)*(1000000).

    Note that cdelt1_arcsec is defined in the get_data function above.

    """
    derx_bt = np.zeros((ny, nx))
    dery_bt = np.zeros((ny, nx))
    err_term1 = np.zeros((ny, nx))
    err_term2 = np.zeros((ny, nx))
    
    derx_bt[:, 1:-1] = (bt[:, 2:] - bt[:, :-2]) * 0.5
    dery_bt[1:-1, :] = (bt[2:, :] - bt[:-2, :]) * 0.5
    derx_bt[:, 0] = ((-3 * bt[:, 0]) + (4 * bt[:, 1]) - bt[:, 2]) * 0.5
    derx_bt[:, -1] = ((3 * bt[:, -1]) - (4 * bt[:, -2]) + bt[:, -3]) * 0.5
    dery_bt[0, :] = ((-3 * bt[0, :]) + (4 * bt[1, :]) - bt[2, :]) * 0.5
    dery_bt[-1, :] = ((3 * bt[-1, :]) - (4 * bt[-2, :]) + bt[-3, :]) * 0.5
    
    err_term1[:, 1:-1] = ((bt[:, 2:] - bt[:, :-2])**2 * 
                          (bt_err[:, 2:]**2 + bt_err[:, :-2]**2))
    err_term2[1:-1, :] = ((bt[2:, :] - bt[:-2, :])**2 * 
                          (bt_err[2:, :]**2 + bt_err[:-2, :]**2))
    valid_mask = (
        (derx_bt[1:-1,1:-1] + dery_bt[1:-1, 1:-1] != 0) & \
        (~np.isnan(bt[1:-1, 1:-1])) & \
        (~np.isnan(bt[2:, 1:-1])) & \
        (~np.isnan(bt[0:-2, 1:-1])) & \
        (~np.isnan(bt[1:-1, 2:])) & \
        (~np.isnan(bt[1:-1, 0:-2])) & \
        (~np.isnan(bt_err[1:-1, 1:-1])) & \
        (~np.isnan(derx_bt[1:-1, 1:-1])) & \
        (~np.isnan(dery_bt[1:-1, 1:-1])) & \
        (mask_map[1:-1, 1:-1] != 0)
        )
        
    # Compute sum and error
    if np.any(valid_mask):
        sum_derivatives = np.sum(np.sqrt(derx_bt[1:-1, 1:-1][valid_mask]**2 + dery_bt[1:-1, 1:-1][valid_mask]**2))
        denom = 16.0 * (derx_bt[1:-1, 1:-1][valid_mask]**2 + dery_bt[1:-1, 1:-1][valid_mask]**2)
        err = np.sum(err_term1[1:-1, 1:-1][valid_mask] / denom) + np.sum(err_term2[1:-1, 1:-1][valid_mask] / denom)
        count_mask = np.count_nonzero(valid_mask)
        
    mean_derivative_bt = sum_derivatives / count_mask if count_mask > 0 else 0.0
    mean_derivative_bt_err = (np.sqrt(err) / count_mask) if count_mask > 0 else 0.0

    return [mean_derivative_bt, mean_derivative_bt_err]


def computeBhderivative_masked(bh, bh_err, nx, ny, mask_map):
    """function: computeBhderivative

    This function computes the derivative of the horizontal field, or sqrt[(dB_h/dx)^2 + (dB_h/dy)^2].
    The native units in the series hmi.sharp_720s and hmi.sharp_cea_720s are in Gauss/pixel.

    Here are the steps to convert from Gauss/pixel to Gauss/Mm:
    The units of the magnetic field, or dB_h, are in Gauss.
    The units of length, i.e. dx or dy, are in pixels.
    The units of dB_h/dx or dB_h/dy = (Gauss/pix)(pix/arcsec)(arsec/meter)(meter/Mm), or
                                    = (Gauss/pix)(1/cdelt1_arcsec)(RSUN_OBS/RSUN_REF)(1000000)
                                    = Gauss/Mm
    In other words, multiply MEANGBH by a factor of (1/cdelt1_arcsec)*(RSUN_OBS/RSUN_REF)*(1000000).

    Note that cdelt1_arcsec is defined in the get_data function above.

    """
    derx_bh = np.zeros((ny, nx))
    dery_bh = np.zeros((ny, nx))
    err_term1 = np.zeros((ny, nx))
    err_term2 = np.zeros((ny, nx))

    derx_bh[:, 1:-1] = (bh[:, 2:] - bh[:, :-2]) * 0.5
    dery_bh[1:-1, :] = (bh[2:, :] - bh[:-2, :]) * 0.5
    derx_bh[:, 0] = (-3 * bh[:, 0] + 4 * bh[:, 1] - bh[:, 2]) * 0.5
    derx_bh[:, -1] = (3 * bh[:, -1] - 4 * bh[:, -2] + bh[:, -3]) * 0.5
    dery_bh[0, :] = (-3 * bh[0, :] + 4 * bh[1, :] - bh[2, :]) * 0.5
    dery_bh[-1, :] = (3 * bh[-1, :] - 4 * bh[-2, :] + bh[-3, :]) * 0.5

    err_term1[:, 1:-1] = (((bh[:, 2:] - bh[:, :-2]) ** 2) * 
                          (bh_err[:, 2:] ** 2 + bh_err[:, :-2] ** 2))
    err_term2[1:-1, :] = (((bh[2:, :] - bh[:-2, :]) ** 2) * 
                          (bh_err[2:, :] ** 2 + bh_err[:-2, :] ** 2))
    valid_mask = (
        (derx_bh[1:-1,1:-1] + dery_bh[1:-1, 1:-1] != 0) & \
        (~np.isnan(bh[1:-1, 1:-1])) & \
        (~np.isnan(bh[2:, 1:-1])) & \
        (~np.isnan(bh[0:-2, 1:-1])) & \
        (~np.isnan(bh[1:-1, 2:])) & \
        (~np.isnan(bh[1:-1, 0:-2])) & \
        (~np.isnan(bh_err[1:-1, 1:-1])) & \
        (~np.isnan(derx_bh[1:-1, 1:-1])) & \
        (~np.isnan(dery_bh[1:-1, 1:-1])) & \
        (mask_map[1:-1, 1:-1] != 0)
        )
    if np.any(valid_mask):
        sum_derivatives = np.sum(np.sqrt(derx_bh[1:-1, 1:-1][valid_mask]**2 + dery_bh[1:-1, 1:-1][valid_mask]**2))
        denom = 16.0 * (derx_bh[1:-1, 1:-1][valid_mask]**2 + dery_bh[1:-1, 1:-1][valid_mask]**2)
        err = np.sum(err_term1[1:-1, 1:-1][valid_mask] / denom) + np.sum(err_term2[1:-1, 1:-1][valid_mask] / denom)
        count_mask = np.count_nonzero(valid_mask)
        
    mean_derivative_bt = sum_derivatives / count_mask if count_mask > 0 else 0.0
    mean_derivative_bt_err = (np.sqrt(err) / count_mask) if count_mask > 0 else 0.0

    return [mean_derivative_bt, mean_derivative_bt_err]


def computeBzderivative_masked(bz, bz_err, nx, ny, mask_map):
    """function: computeBzderivative

    This function computes the derivative of the vertical field, or sqrt[(dB_z/dx)^2 + (dB_z/dy)^2].
    The native units in the series hmi.sharp_720s and hmi.sharp_cea_720s are in Gauss/pixel.

    Here are the steps to convert from Gauss/pixel to Gauss/Mm:
    The units of the magnetic field, or dB_z, are in Gauss.
    The units of length, i.e. dx or dy, are in pixels.
    The units of dB_z/dx or dB_z/dy = (Gauss/pix)(pix/arcsec)(arsec/meter)(meter/Mm), or
                                    = (Gauss/pix)(1/cdelt1_arcsec)(RSUN_OBS/RSUN_REF)(1000000)
                                    = Gauss/Mm
    In other words, multiply MEANGBZ by a factor of (1/cdelt1_arcsec)*(RSUN_OBS/RSUN_REF)*(1000000).

    Note that cdelt1_arcsec is defined in the get_data function above.

    """
    derx_bz = np.zeros((ny, nx))
    dery_bz = np.zeros((ny, nx))
    err_term1 = np.zeros((ny, nx))
    err_term2 = np.zeros((ny, nx))

    derx_bz[:, 1:-1] = (bz[:, 2:] - bz[:, 0:-2]) * 0.5
    dery_bz[1:-1, :] = (bz[2:, :] - bz[0:-2, :]) * 0.5
    derx_bz[:, 0] = (-3 * bz[:, 0] + 4 * bz[:, 1] - bz[:, 2]) * 0.5
    derx_bz[:, -1] = (3 * bz[:, -1] - 4 * bz[:, -2] + bz[:, -3]) * 0.5
    dery_bz[0, :] = (-3 * bz[0, :] + 4 * bz[1, :] - bz[2, :]) * 0.5
    dery_bz[-1, :] = (3 * bz[-1, :] - 4 * bz[-2, :] + bz[-3, :]) * 0.5
    
    err_term1[:, 1:-1] = (((bz[:, 2:] - bz[:, 0:-2]) ** 2) * 
                          (bz_err[:, 2:] ** 2 + bz_err[:, 0:-2] ** 2))
    err_term2[1:-1, :] = (((bz[2:, :] - bz[0:-2, :]) ** 2) * 
                          (bz_err[2:, :] ** 2 + bz_err[0:-2, :] ** 2))
    valid_mask = (
        (derx_bz[1:-1,1:-1] + dery_bz[1:-1, 1:-1] != 0) & \
        (~np.isnan(bz[1:-1, 1:-1])) & \
        (~np.isnan(bz[2:, 1:-1])) & \
        (~np.isnan(bz[0:-2, 1:-1])) & \
        (~np.isnan(bz[1:-1, 2:])) & \
        (~np.isnan(bz[1:-1, 0:-2])) & \
        (~np.isnan(bz_err[1:-1, 1:-1])) & \
        (~np.isnan(derx_bz[1:-1, 1:-1])) & \
        (~np.isnan(dery_bz[1:-1, 1:-1])) & \
        (mask_map[1:-1, 1:-1] == 1)
        )
    if np.any(valid_mask):
        sum_derivatives = np.sum(np.sqrt(derx_bz[1:-1, 1:-1][valid_mask]**2 + dery_bz[1:-1, 1:-1][valid_mask]**2))
        denom = 16.0 * (derx_bz[1:-1, 1:-1][valid_mask]**2 + dery_bz[1:-1, 1:-1][valid_mask]**2)
        err = np.sum(err_term1[1:-1, 1:-1][valid_mask] / denom) + np.sum(err_term2[1:-1, 1:-1][valid_mask] / denom)
        count_mask = np.count_nonzero(valid_mask)
        
    mean_derivative_bt = sum_derivatives / count_mask if count_mask > 0 else 0.0
    mean_derivative_bt_err = (np.sqrt(err) / count_mask) if count_mask > 0 else 0.0

    return [mean_derivative_bt, mean_derivative_bt_err]



def computeJz_masked(bx, by, bx_err, by_err, nx, ny):
    """function: computeJz

    This function computes the z-component of the current.

    In discretized space like data pixels, the current (or curl of B) is calculated as the integration
    of the field Bx and By along the circumference of the data pixel divided by the area of the pixel.

    One form of differencing the curl is expressed as:
    (dx * (Bx(i,j-1)+Bx(i,j)) / 2
    +dy * (By(i+1,j)+By(i,j)) / 2
    -dx * (Bx(i,j+1)+Bx(i,j)) / 2
    -dy * (By(i-1,j)+By(i,j)) / 2) / (dx * dy)

    To change units from Gauss/pixel to mA/m^2 (the units for Jz in Leka and Barnes, 2003),
    one must perform the following unit conversions:
    (Gauss)(1/arcsec)(arcsec/meter)(Newton/Gauss*Ampere*meter)(Ampere^2/Newton)(milliAmpere/Ampere), or
    (Gauss)(1/CDELT1)(RSUN_OBS/RSUN_REF)(1 T / 10^4 Gauss)(1 / 4*PI*10^-7)( 10^3 milliAmpere/Ampere), or
    (Gauss)(1/CDELT1)(RSUN_OBS/RSUN_REF)(0.00010)(1/MUNAUGHT)(1000.),
    where a Tesla is represented as a Newton/Ampere*meter.

    The units of total unsigned vertical current (us_i) are simply in A. In this case, we would have the following:
    (Gauss/pix)(1/CDELT1)(RSUN_OBS/RSUN_REF)(0.00010)(1/MUNAUGHT)(CDELT1)(CDELT1)(RSUN_REF/RSUN_OBS)(RSUN_REF/RSUN_OBS)
    = (Gauss/pix)(0.00010)(1/MUNAUGHT)(CDELT1)(RSUN_REF/RSUN_OBS)
    """
    derx = np.zeros((ny, nx))
    dery = np.zeros((ny, nx))
    err_term1 = np.zeros((ny, nx))
    err_term2 = np.zeros((ny, nx))
    jz = np.zeros((ny, nx))
    jz_err = np.zeros((ny, nx))
    derx[:, 1:nx-1] = 0.5 * (by[:, 2:nx] - by[:, :nx-2])
    dery[1:ny-1, :] = 0.5 * (bx[2:ny, :] - bx[:ny-2, :])
    err_term1[:, 1:nx-1] = (by_err[:, 2:nx] ** 2 + by_err[:, :nx-2] ** 2)
    err_term2[1:ny-1, :] = (bx_err[2:ny, :] ** 2 + bx_err[:ny-2, :] ** 2)
    derx[:, 0] = (-3 * by[:, 0] + 4 * by[:, 1] - by[:, 2]) * 0.5
    derx[:, -1] = (3 * by[:, -1] - 4 * by[:, -2] + by[:, -3]) * 0.5
    dery[0, :] = (-3 * bx[0, :] + 4 * bx[1, :] - bx[2, :]) * 0.5
    dery[-1, :] = (3 * bx[-1, :] - 4 * bx[-2, :] + bx[-3, :]) * 0.5
    jz[1:ny-1, 1:nx-1] = derx[1:ny-1, 1:nx-1] - dery[1:ny-1, 1:nx-1]
    jz_err[1:ny-1, 1:nx-1] = 0.5 * np.sqrt(err_term1[1:ny-1, 1:nx-1] + err_term2[1:ny-1, 1:nx-1])

    return [jz, jz_err, derx, dery]



def computeJzmoments_masked(jz, jz_err, derx, dery, rsun_ref, rsun_obs, cdelt1_arcsec, munaught, mask_map):
    """Function: computeJzmoments

    This function computes moments of the vertical current.
    The mean vertical current density is in units of mA/m^2.
    The total unsigned vertical current is in units of Amperes.
    """
    valid_mask = ~np.isnan(jz) & ~np.isnan(derx) & ~np.isnan(dery) & (mask_map != 0)
    jz_valid = jz[valid_mask]
    jz_err_valid = jz_err[valid_mask]
    curl = np.sum(jz_valid) * (1 / cdelt1_arcsec) * (rsun_obs / rsun_ref) * (0.00010) * (1 / munaught) * (1000.)
    us_i = np.sum(np.abs(jz_valid)) * (cdelt1_arcsec) * (rsun_ref / rsun_obs) * (0.00010) * (1 / munaught)
    err = np.sum(jz_err_valid ** 2)

    count_mask = jz_valid.size

    mean_jz = curl / count_mask if count_mask > 0 else 0
    mean_jz_err = (np.sqrt(err) / count_mask) * ((1 / cdelt1_arcsec) * (rsun_obs / rsun_ref) * (0.00010) * (1 / munaught) * (1000.)) if count_mask > 0 else 0

    us_i_err = np.sqrt(err) * ((cdelt1_arcsec) * (rsun_ref / rsun_obs) * (0.00010) * (1 / munaught))

    return [mean_jz, mean_jz_err, us_i, us_i_err]



def computeAlpha_masked(jz, jz_err, bz, bz_err, rsun_ref, rsun_obs, cdelt1_arcsec, mask_map):
    """function: computeAlpha

    This function computes the twist parameter.

    The twist parameter, alpha, is defined as alpha = Jz/Bz. In this case, the calculation for alpha is weighted by Bz:

    numerator   = sum of all Jz*Bz
    denominator = sum of Bz*Bz
    alpha       = numerator/denominator

    The units of alpha are in 1/Mm
    The units of Jz are in Gauss/pix; the units of Bz are in Gauss.

    Therefore, the units of Jz/Bz = (Gauss/pix)(1/Gauss)(pix/arcsec)(arsec/meter)(meter/Mm), or
    = (Gauss/pix)(1/Gauss)(1/CDELT1)(RSUN_OBS/RSUN_REF)(10^6)
    = 1/Mm
    """

    C = (1 / cdelt1_arcsec) * (rsun_obs / rsun_ref) * 1000000.0

    # Create a mask for valid entries
    valid_mask = ~np.isnan(jz) & ~np.isnan(bz) & (jz != 0) & (bz != 0) & (mask_map != 0)

    # Use valid mask to filter jz and bz
    jz_valid = jz[valid_mask]
    bz_valid = bz[valid_mask]
    
    # Compute A and B using vectorized operations
    A = np.sum(jz_valid * bz_valid)
    B = np.sum(bz_valid ** 2)

    # Compute the total error term using the same valid mask
    jz_err_valid = jz_err[valid_mask]
    bz_err_valid = bz_err[valid_mask]
    
    total = np.sum((bz_valid ** 2 * jz_err_valid ** 2) + 
                   ((jz_valid - 2 * bz_valid * A / B) ** 2) * (bz_err_valid ** 2))

    # Calculate alpha and its error
    alpha_total = (A / B) * C if B != 0 else 0
    mean_alpha_err = (C / B) * np.sqrt(total) if B != 0 else 0

    return [alpha_total, mean_alpha_err]



def computeHelicity_masked(jz, jz_err, bz, bz_err, rsun_ref, rsun_obs, cdelt1_arcsec, mask_map):
    """function: computeHelicity

    This function computes a proxy for the current helicity and various moments.

    The current helicity is defined as Bz*Jz and the units are G^2 / m
    The units of Jz are in G/pix; the units of Bz are in G.
    Therefore, the units of Bz*Jz = (Gauss)*(Gauss/pix) = (Gauss^2/pix)(pix/arcsec)(arcsec/meter)
    = (Gauss^2/pix)(1/CDELT1)(RSUN_OBS/RSUN_REF)
    =  G^2 / m.
    """

    # Calculate constants for unit conversion
    C = (1 / cdelt1_arcsec) * (rsun_obs / rsun_ref)

    # Create a mask for valid entries
    valid_mask = ~np.isnan(jz) & ~np.isnan(bz) & (jz != 0) & (bz != 0) & ~np.isnan(jz_err) & ~np.isnan(bz_err) & (mask_map != 0)

    # Use valid mask to filter jz, jz_err, bz, bz_err
    jz_valid = jz[valid_mask]
    bz_valid = bz[valid_mask]
    jz_err_valid = jz_err[valid_mask]
    bz_err_valid = bz_err[valid_mask]

    # Compute sums
    sum_helicity = np.sum(jz_valid * bz_valid) * C
    total_us_helicity = np.sum(np.abs(jz_valid * bz_valid)) * C
    
    # Compute error term
    err = (jz_err_valid ** 2 * bz_valid ** 2) + (bz_err_valid ** 2 * jz_valid ** 2)
    
    # Calculate count of valid entries
    count_mask = np.sum(valid_mask)

    # Calculate mean helicity and errors
    mean_helicity = sum_helicity / count_mask if count_mask > 0 else 0
    mean_helicity_err = (np.sqrt(np.sum(err)) / count_mask) * C if count_mask > 0 else 0
    total_us_helicity_err = np.sqrt(np.sum(err)) * C
    total_abs_helicity_err = total_us_helicity_err

    return [mean_helicity, mean_helicity_err, total_us_helicity, total_us_helicity_err, abs(sum_helicity), total_abs_helicity_err]



def computeSumAbsPerPolarity_masked(jz, jz_err, bz, rsun_ref, rsun_obs, cdelt1_arcsec, munaught, mask_map):
    """function: computeSumAbsPerPolarity

    This function computes the sum of the absolute value of the current per polarity. It is defined as follows:

    The sum of the absolute value per polarity is defined as the following:
    abs(sum(jz gt 0)) + abs(sum(jz lt 0)) and the units are in Amperes per arcsecond.
    The units of jz are in G/pix. In this case, we would have the following:
    Jz = (Gauss/pix)(1/CDELT1)(0.00010)(1/MUNAUGHT)(RSUN_REF/RSUN_OBS)(RSUN_REF/RSUN_OBS)(RSUN_OBS/RSUN_REF),
       = (Gauss/pix)(1/CDELT1)(0.00010)(1/MUNAUGHT)(RSUN_REF/RSUN_OBS)

    The error in this quantity is the same as the error in the mean vertical current.
    """

    # Constants for unit conversion
    factor = (1 / cdelt1_arcsec) * (0.00010) * (1 / munaught) * (rsun_ref / rsun_obs)

    # Create a mask for valid entries
    valid_mask = ~np.isnan(jz) & ~np.isnan(bz) & (mask_map != 0)

    # Filter valid arrays
    jz_valid = jz[valid_mask]
    bz_valid = bz[valid_mask]
    jz_err_valid = jz_err[valid_mask]

    # Compute sums based on polarity
    sum1 = np.sum(jz_valid[bz_valid > 0]) * factor
    sum2 = np.sum(jz_valid[bz_valid <= 0]) * factor

    # Calculate the total and errors
    totaljz = abs(sum1) + abs(sum2)
    totaljz_err = np.sqrt(np.sum(jz_err_valid ** 2)) * factor

    return [totaljz, totaljz_err]



def computeFreeEnergy_masked(bx_err, by_err, bx, by, bpx, bpy, rsun_ref, rsun_obs, cdelt1_arcsec, mask_map):
    """
    function: computeFreeEnergy

    This function computes the mean photospheric excess magnetic energy and total photospheric excess magnetic energy density.

    The units for magnetic energy density in cgs are ergs per cubic centimeter. The formula B^2/8*PI integrated over all space, dV
    automatically yields erg per cubic centimeter for an input B in Gauss. Note that the 8*PI can come out of the integral; thus,
    the integral is over B^2 dV and the 8*PI is divided at the end.

    Total magnetic energy is the magnetic energy density times dA, or the area, and the units are thus ergs/cm. To convert
    ergs per centimeter cubed to ergs per centimeter, simply multiply by the area per pixel in cm:
    erg/cm^3*(CDELT1^2)*(RSUN_REF/RSUN_OBS ^2)*(100.^2)
    = erg/cm(1/pix^2)
    """
    
    
    # Constants for area and scaling
    area_factor = cdelt1_arcsec ** 2 * (rsun_ref / rsun_obs) ** 2 * 100.0 ** 2
    coeff = 1 / (8.0 * np.pi)

    # Create a mask for valid entries
    valid_mask = ~np.isnan(bx) & ~np.isnan(by) & (mask_map != 0)

    # Filter valid arrays
    bx_valid = bx[valid_mask]
    by_valid = by[valid_mask]
    bpx_valid = bpx[valid_mask]
    bpy_valid = bpy[valid_mask]
    bx_err_valid = bx_err[valid_mask]
    by_err_valid = by_err[valid_mask]

    # Calculate differences
    diff_bx = bx_valid - bpx_valid
    diff_by = by_valid - bpy_valid

    # Compute sums
    sum1 = np.sum(diff_bx ** 2 + diff_by ** 2)
    sum = np.sum((diff_bx ** 2 + diff_by ** 2) * area_factor)

    # Compute error terms
    err = 4.0 * np.sum((diff_bx ** 2 * bx_err_valid ** 2) + (diff_by ** 2 * by_err_valid ** 2))

    count_mask = valid_mask.sum()

    # Calculate mean and total potentials
    meanpot = sum1 / (count_mask * 8.0 * np.pi)
    meanpot_err = np.sqrt(err) / (count_mask * 8.0 * np.pi)
    
    totpot = sum / (8.0 * np.pi)
    totpot_err = np.sqrt(err) * abs(coeff * area_factor)

    return [meanpot, meanpot_err, totpot, totpot_err]



def computeShearAngle_masked(bx_err, by_err, bz_err, bx, by, bz, bpx, bpy, mask_map):
    """Function: computeShearAngle

    This function computes the shear angle, or the angle between the potential field vector and the observed field vector, in degrees.
    """

    # Create a mask for valid entries
    valid_mask = (
                  (~np.isnan(bx)) & (~np.isnan(by)) & (~np.isnan(bz)) & \
                  (~np.isnan(bpx)) & (~np.isnan(bpy)) & \
                  (~np.isnan(bx_err)) & (~np.isnan(by_err)) & ~np.isnan(bz_err) & \
                  (mask_map != 0)
                  )
                  

    # Filter valid arrays
    bx_valid = bx[valid_mask]
    by_valid = by[valid_mask]
    bz_valid = bz[valid_mask]
    bpx_valid = bpx[valid_mask]
    bpy_valid = bpy[valid_mask]
    bx_err_valid = bx_err[valid_mask]
    #by_err_valid = by_err[valid_mask]
    #bz_err_valid = bz_err[valid_mask]

    # Compute dot products and magnitudes
    dotproduct = bpx_valid * bx_valid + bpy_valid * by_valid + bz_valid * bz_valid
    magnitude_potential = np.sqrt(bpx_valid**2 + bpy_valid**2 + bz_valid**2)
    magnitude_vector = np.sqrt(bx_valid**2 + by_valid**2 + bz_valid**2)

    # Compute shear angles
    shear_angle = np.arccos(dotproduct / (magnitude_potential * magnitude_vector)) *(180.0/np.pi)
    
    # Calculate the mean shear angle and count
    meanshear_angle = np.nanmean(shear_angle)
    
    # Count the angles greater than 45 degrees
    count_mask = np.sum(shear_angle > 45)

    # Calculate error terms
    part1 = bx_valid**2 + by_valid**2 + bz_valid**2
    part2 = bpx_valid**2 + bpy_valid**2 + bz_valid**2
    part3 = bx_valid * bpx_valid + by_valid * bpy_valid + bz_valid * bz_valid

    denominator = part1**3 * part2 * (1.0 - (part3**2 / (part1 * part2)))

    # Error calculations (avoiding division by zero)
    term1 = bx_valid * by_valid * bpy_valid - by_valid**2 * bpx_valid + bz_valid * bx_valid * bz_valid - bz_valid**2 * bpx_valid
    err = (term1**2 * bx_err_valid**2) * 3 / denominator

    meanshear_angle_err = np.sqrt(np.nansum(err)) / np.maximum(1, count_mask) * (180. / np.pi)

    # Fractional area of shear greater than 45 degrees
    area_w_shear_gt_45 = (count_mask / len(bx_valid)) * 100.0 if len(bx_valid) > 0 else 0.0

    return [meanshear_angle, meanshear_angle_err, area_w_shear_gt_45]



def computeR_masked(los, los_err, cdelt1_arcsec, mask_map):
    """
    function: computeR

    This function computes R, or the log of the gradient-weighted neutral line length.
    So the output is unitless.

    This function also computes the error in R. The general formula for an error of a
    function is ERR(f(x)) = d/dx f(x) * ERR(x). Thus
                ERR(R) = d/dx (log_10 R) * ERR(x)
                       = [1/ln(10)] * [1/x] * ERR(x)
                       = ERR(x) / ln(10)*x
    """
    
    if mask_map is None:
        mask_map = np.ones(los.shape)

    sigma = 10.0 / 2.3548
    scale = int(round(2.0 / cdelt1_arcsec))

    # =============== [STEP 1] ===============
    # Bin the line-of-sight magnetogram down by a factor of scale
    rim = block_reduce(los, block_size=(scale, scale), func=np.mean)
    los_err_bin = block_reduce(los_err, block_size=(scale, scale), func=np.mean)
    mask_map_bin = block_reduce(mask_map, block_size=(scale, scale), func=np.mean)

    # =============== [STEP 2] ===============
    # Identify positive and negative pixels greater than +/- 150 gauss
    p1p0 = (rim > 150).astype(float)
    p1n0 = (rim < -150).astype(float)

    # =============== [STEP 3] ===============
    # Smooth each of the negative and positive pixel bitmaps by convolving with a boxcar
    nx1 = rim.shape[1]
    ny1 = rim.shape[0]
    boxcar_kernel = np.zeros([ny1, nx1])
    boxcar_kernel[int(round(ny1/2)):int(round(ny1/2))+3, int(round(nx1/2)):int(round(nx1/2))+3] = 0.1111
    p1p = scipy.ndimage.convolve(p1p0, boxcar_kernel, mode='constant', cval=0)
    p1n = scipy.ndimage.convolve(p1n0, boxcar_kernel, mode='constant', cval=0)

    # =============== [STEP 4] ===============
    # Find the pixels for which p1p and p1n are both equal to 1
    p1 = (p1p > 0.0) & (p1n > 0.0)  # Boolean array

    # =============== [STEP 5] ===============
    # Convolve the polarity inversion line map with a Gaussian
    pmap = scipy.ndimage.gaussian_filter(p1.astype(float), sigma, order=0)

    # =============== [STEP 6] ===============
    # The R parameter is calculated
    sum_vals = np.nansum(pmap * np.abs(rim) * mask_map_bin)
    err = np.nansum(pmap * np.abs(los_err_bin) * mask_map_bin)

    if sum_vals < 1.0:
        Rparam = 0.0
        Rparam_err = 0.0
    else:
        Rparam = math.log10(sum_vals)
        Rparam_err = err / (math.log(10) * sum_vals)

    return [Rparam, Rparam_err]


def computeLOSderivative_masked(los, los_err, nx, ny, mask_map, bitmap, method):
    """function: computeLOSderivative

    This function computes the derivative of the line-of-sight field, or sqrt[(dB_los/dx)^2 + (dB_los/dy)^2].
    The native units in the series hmi.sharp_720s and hmi.sharp_cea_720s are in Gauss/pixel.

    Here are the steps to convert from Gauss/pixel to Gauss/Mm:
    The units of the magnetic field, or dB_los, are in Gauss.
    The units of length, i.e. dx or dy, are in pixels.
    The units of dB_los/dx or dB_los/dy = (Gauss/pix)(pix/arcsec)(arsec/meter)(meter/Mm), or
                                        = (Gauss/pix)(1/cdelt1_arcsec)(RSUN_OBS/RSUN_REF)(1000000)
                                        = Gauss/Mm
    In other words, multiply MEANGBL by a factor of (1/cdelt1_arcsec)*(RSUN_OBS/RSUN_REF)*(1000000).

    Note that cdelt1_arcsec is defined in the get_data function above.
    """

    # unitconstant = (1 / cdelt1_arcsec) * (rsun_obs / rsun_ref) * 1e6

    # Initialize derivative and error arrays
    derx_blos = np.zeros((ny, nx))
    dery_blos = np.zeros((ny, nx))
    err_term1 = np.zeros((ny, nx))
    err_term2 = np.zeros((ny, nx))

    # Compute derivatives using central difference for the interior points
    derx_blos[:, 1:-1] = (los[:, 2:] - los[:, :-2]) * 0.5
    dery_blos[1:-1, :] = (los[2:, :] - los[:-2, :]) * 0.5

    # Edge cases for x-derivative
    derx_blos[:, 0] = (-3 * los[:, 0] + 4 * los[:, 1] - los[:, 2]) * 0.5
    derx_blos[:, -1] = (3 * los[:, -1] - 4 * los[:, -2] + los[:, -3]) * 0.5
    dery_blos[0, :] = (-3 * los[0, :] + 4 * los[1, :] - los[2, :]) * 0.5
    dery_blos[-1, :] = (3 * los[-1, :] - 4 * los[-2, :] + los[-3, :]) * 0.5
    
    denominator = derx_blos**2 + dery_blos**2

    # Calculate error terms
    err_term1[:, 1:-1] = ((los[:, 2:] - los[:, :-2])**2) * (los_err[:, 2:]**2 + los_err[:, :-2]**2)
    err_term2[1:-1, :] = ((los[2:, :] - los[:-2, :])**2) * (los_err[2:, :]**2 + los_err[:-2, :]**2)

    # Calculate sum and error    
    base_mask = (
        (~np.isnan(los[1:-1, 1:-1])) & \
        (~np.isnan(los[2:, 1:-1])) & \
        (~np.isnan(los[0:-2, 1:-1])) & \
        (~np.isnan(los[1:-1, 2:])) & \
        (~np.isnan(los[1:-1, 0:-2])) & \
        (~np.isnan(derx_blos[1:-1, 1:-1])) & \
        (~np.isnan(dery_blos[1:-1, 1:-1])) & \
        (~np.isnan(denominator[1:-1, 1:-1])) & \
        (denominator[1:-1, 1:-1] != 0)
    )
    
    if method == 'ori':
        valid_mask = base_mask & (bitmap[1:-1, 1:-1] >= 30)
    elif method == 'pil':
        valid_mask = base_mask & (bitmap[1:-1, 1:-1] >= 30) & (mask_map[1:-1, 1:-1] != 0)
    elif method == 'mfr':
        valid_mask = base_mask & (mask_map[1:-1, 1:-1] != 0)
    else:
        raise ValueError("method set error")

    # Compute sum and error
    if np.any(valid_mask):
        sum_derivatives = np.sum(np.sqrt(derx_blos[1:-1, 1:-1][valid_mask]**2 + dery_blos[1:-1, 1:-1][valid_mask]**2))
        denom = 16.0 * (derx_blos[1:-1, 1:-1][valid_mask]**2 + dery_blos[1:-1, 1:-1][valid_mask]**2)
        err = np.sum(err_term1[1:-1, 1:-1][valid_mask] / denom) + np.sum(err_term2[1:-1, 1:-1][valid_mask] / denom)
        count_mask = np.count_nonzero(valid_mask)
        
    mean_derivative_bt = sum_derivatives / count_mask if count_mask > 0 else 0.0
    mean_derivative_bt_err = (np.sqrt(err) / count_mask) if count_mask > 0 else 0.0

    return [mean_derivative_bt, mean_derivative_bt_err]



def compute_abs_flux_los_masked(los, los_err, rsun_ref, rsun_obs, cdelt1_arcsec, mask_map, bitmap, method):
    """function: compute_abs_flux_los

    This function computes the total unsigned flux, on the line-of-sight field, in units of G/cm^2.
    It also returns the number of pixels used in this calculation in the keyword CMASK.

    To compute the unsigned flux, we simply calculate
       flux = surface integral [(vector Blos) dot (normal vector)],
            = surface integral [(magnitude Blos)*(magnitude normal)*(cos theta)].

    However, since the field is radial, we will assume cos theta = 1.
    Therefore, the pixels only need to be corrected for the projection.

    To convert G to G*cm^2, simply multiply by the number of square centimeters per pixel:
       (Gauss/pix^2)(CDELT1)^2(RSUN_REF/RSUN_OBS)^2(100.cm/m)^2
       =Gauss*cm^2
    """

    # Create a mask for valid pixels
    if method == 'ori':
        valid_mask = ~np.isnan(los) & (bitmap >= 30)
    elif method == 'pil':
        valid_mask = ~np.isnan(los) & (bitmap >= 30) & (mask_map != 0)
    elif method == 'mfr':
        valid_mask = ~np.isnan(los) & (mask_map != 0)
    else:
        raise ValueError("method set error")
    
    # Compute the absolute values and errors for valid pixels
    abs_los = np.abs(los[valid_mask])
    abs_los_err = los_err[valid_mask]

    # Calculate total flux and error
    total_flux = np.sum(abs_los)
    total_err = np.sum(abs_los_err**2)

    count_mask = np.count_nonzero(valid_mask)

    # Convert to G/cm^2
    conversion_factor = cdelt1_arcsec**2 * (rsun_ref / rsun_obs)**2 * 100**2
    mean_vf = total_flux * conversion_factor
    mean_vf_err = np.sqrt(total_err) * conversion_factor

    return [mean_vf, mean_vf_err, count_mask]



def greenpot(bz, nx, ny):
    """
    function: greenpot

    This function extrapolates the potential magnetic field using Green's functions.
    The underlying assuption of a potential field is that it is Maxwell-stress free.
    The monopole depth is 0.01 pixels.
    """
    #print('Calculating the potential field. This takes a minute.')

    dz = 0.001
    bztmp = np.nan_to_num(bz)  # Replace NaNs with zeros
    pfpot = np.zeros((ny, nx))
    
    # Create a grid of indices
    x_indices = np.arange(nx)
    y_indices = np.arange(ny)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices, indexing='ij')

    # Calculate the distance matrix
    rdist = 1.0 / np.sqrt(x_grid**2 + y_grid**2 + dz**2)

    # Define the window size
    window_size = min(nx, ny)
    # rwindow = min(np.sqrt(1.0 * window_size**2 + 0.01), 10.0)
    rwindow = 10.0
    iwindow = int(rwindow)

    # Loop through each point to calculate the potential field
    for iny in range(ny):
        for inx in range(nx):
            if np.isnan(bz[iny, inx]):
                pfpot[iny, inx] = 0.0
            else:
                # Define the window limits
                j2s = max(0, iny - iwindow)
                j2e = min(ny, iny + iwindow)
                i2s = max(0, inx - iwindow)
                i2e = min(nx, inx + iwindow)

                # Calculate the potential field
                val1 = bztmp[j2s:j2e, i2s:i2e]
                distances = rdist[np.abs(y_indices[j2s:j2e][:, None] - iny), np.abs(x_indices[i2s:i2e] - inx)]
                pfpot[iny, inx] = np.sum(val1 * distances * dz)

    # Calculate magnetic field components
    bxp = np.zeros_like(pfpot)
    byp = np.zeros_like(pfpot)
    bxp[1:-1, 1:-1] = -(pfpot[1:-1, 2:] - pfpot[1:-1, :-2]) * 0.5
    byp[1:-1, 1:-1] = -(pfpot[2:, 1:-1] - pfpot[:-2, 1:-1]) * 0.5

    return [bxp, byp]


# ===========================================