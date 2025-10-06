Features (used for training & prediction)

The following columns are required for the model:
```ra, dec, koi_gmag, koi_count, koi_num_transits,
koi_max_mult_ev, koi_bin_oedp_sig, koi_ldm_coeff1,
koi_model_snr, koi_prad, koi_impact, koi_duration, koi_depth,
koi_dor, koi_incl, koi_teq, koi_steff, koi_slogg, koi_smet,
koi_srad, koi_smass, koi_fwm_stat_sig, koi_fwm_srao,
koi_fwm_sdeco, koi_fwm_prao, koi_fwm_pdeco, koi_dicco_mra,
koi_dicco_mdec, koi_dicco_msky, koi_tce_plnt_num, koi_time0,
koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec,
koi_insol, koi_srho
```
And the label:
```
koi_disposition
```

You must provide those seperately in .csv in the Model Training tab and then train it and save the model

To use it to predict, load a .csv with columns mentioned in the Model Prediction tab through the Add Batch and load the model then hit Start Prediction

If you desire to visualize the orbit, load the raw data from the KOI dataset from a .csv, choose the planet below and click the button
