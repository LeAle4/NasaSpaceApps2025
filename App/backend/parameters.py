"""Project-wide constants describing feature/visualization column lists.

DATA_HEADERS lists the expected feature column names used by the ML model.
These must match the columns present in any CSV loaded for prediction. The
list is intentionally explicit so callers can pass it to pandas.read_csv(usecols=...).

DATA_VISUALIZATION is a smaller set of columns commonly used by the
visualization utilities in the analysis package.
"""

DATA_HEADERS = ['ra', 'dec', 'koi_gmag', 'koi_count', 'koi_num_transits',
       'koi_max_mult_ev', 'koi_bin_oedp_sig', 'koi_ldm_coeff1',
       'koi_model_snr', 'koi_prad', 'koi_impact', 'koi_duration', 'koi_depth',
       'koi_dor', 'koi_incl', 'koi_teq', 'koi_steff', 'koi_slogg', 'koi_smet',
       'koi_srad', 'koi_smass', 'koi_fwm_stat_sig', 'koi_fwm_srao',
       'koi_fwm_sdeco', 'koi_fwm_prao', 'koi_fwm_pdeco', 'koi_dicco_mra',
       'koi_dicco_mdec', 'koi_dicco_msky', 'koi_tce_plnt_num', 'koi_time0',
       'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
       'koi_insol', 'koi_srho']

DATA_VISUALIZATION = ['kepid','koi_period','koi_time0bk','koi_smass','koi_srad','koi_prad','koi_sma','koi_eccen','koi_incl','koi_longp','koi_steff']
