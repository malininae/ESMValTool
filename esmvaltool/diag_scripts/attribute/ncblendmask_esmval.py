import sys
import numpy as np
import iris
import glob
import logging
import os

import esmvalcore.preprocessor as preproc

logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
# Following two functions for blending and masking modified from Cowtan 2015.
# Calculate blended temperatures using general methods
# Source of original Cowtan code:
# http://www-users.york.ac.uk//~kdc3/papers/robust2015/methods.html
# Usage:
#  python ncblendmask.py <mode> tas.nc tos.nc sic.nc sftlf.nc obs.nc dec_warming obs_dec_warming ann_warming gmst_comp_warming diag_name obs ensobs ensobs_diag ensobs_dec_warming
#  <mode> is one of xxx, mxx, xax, max, xxf, mxf, xaf, maf
# max means use masking of anomalies, with time-varying sea ice. See Cowtan website for more details.
# tas.nc. tos.nc, sic.nc, sftlf.nc and obs.nc are names of NetCDF files containing tas, tos, siconc, sftlf from the simulation.
# obs.nc is the name of the observations NetCDF file (the median if using an ensemble obs dataset).
# dec_warming is 2010-2019 warming in GSAT from the model.
# obs_dec_warming is 2010-2019 warming in GMST from the obs.
# ann_warming is a timeseries of annual mean GSAT from the model.
# gmst_comp_warming is 2010-2019 warming globally-complete GMST from the model.
# diag_name is an input diagnostic name e.g. gmst05, hemi10, where the last two digits are the averaging period.
# obs indicates which obs dataset is being used had5/had4 etc.
# ensobs is the partial filename of ensemble obs dataset, if ensemble data is used, otherwise empty string.
# ensobs_diag is the diabnostic requested for each of the ensemble members of an ensemble obs dataset.
# ensobs_dec_warming is 2010-2019 warming in GMST for each for each of the ensemble members of obs dataset.
#Outputs
# diag is the requested diagnostic (e.g. gmst05) for the model.
# obs_diag is the requested dignostic (e.g. gmst05) for the obs.

## Nathan Gillett - Adapted from ncblendmask-nc4.py from Cowtan 2015

def ncblendmask_esmval(tas_file,obs_file,diag_name,ensobs='',ensobs_diag=[],ensobs_dec_warming=[], cfg=dict()):
# MAIN PROGRAM

# m = mask
# a = blend anomalies
# f = fix ice
# (use x for none)

  tas_cube = iris.load_cube(tas_file)

  obs_ens_mean_cube = iris.load_cube(obs_file)

  tas_cube = preproc.anomalies(tas_cube, period='monthly', reference=dict(start_year=1961, start_month=1, start_day=1,
                                                end_year=1990, end_month=12,end_day=31))

  area_stat = preproc.area_statistics(tas_cube,operator='mean')

  start_year = cfg['input_data'][tas_file]['start_year']
  end_year = cfg['input_data'][tas_file]['end_year']

  obs_ens_mean_cube = preproc.extract_time(obs_ens_mean_cube, start_year, 1, 1, end_year, 12, 31)
  obs_ens_mean_cube = preproc.extract_shape(obs_ens_mean_cube, os.path.join(cfg['auxiliary_data_dir'], 'canada.shp'),
                                          method='contains', crop=True)

  obs_ens_mean_cube_area = preproc.area_statistics(obs_ens_mean_cube, operator='mean')

  yearly_stat = preproc.annual_statistics(area_stat, 'mean')
  yearly_stat = yearly_stat.data
  obs_ens_yearly_stat = preproc.annual_statistics(obs_ens_mean_cube_area, 'mean')
  obs_ens_yearly_stat = obs_ens_yearly_stat.data

  if int(diag_name[4:6]) > 1:
      yearly_stat = np.mean(yearly_stat.reshape(len(yearly_stat)//int(diag_name[4:6]), -1), axis=1)
      obs_ens_yearly_stat = np.mean(obs_ens_yearly_stat.reshape(len(obs_ens_yearly_stat)//int(diag_name[4:6]), -1), axis=1)

  diag = yearly_stat - np.mean(yearly_stat)

  last_dec_cb = preproc.extract_time(area_stat, 2010, 1, 1, 2019, 12, 31)
  ref_per_cb = preproc.extract_time(area_stat, 1850, 1, 1, 1900, 12, 31)

  dec_warming = np.mean(last_dec_cb.data) - np.mean(ref_per_cb.data)
  ann_warming = yearly_stat - np.mean(ref_per_cb.data)
  gmst_comp_warming = ann_warming

  obs_ens_mean_cube_last_dec = preproc.extract_time(obs_ens_mean_cube_area, 2010, 1, 1, 2019, 12, 31)
  obs_ens_mean_cube_ref_per = preproc.extract_time(obs_ens_mean_cube_area, 1850, 1, 1, 1900, 12, 31)

  obs_dec_warming = np.mean(obs_ens_mean_cube_last_dec.data) - np.mean(obs_ens_mean_cube_ref_per.data)
  obs_diag = obs_ens_yearly_stat - np.mean(obs_ens_yearly_stat)

  # Liza's comment
  if np.max(diag) > 10:
    logger.error('Something went wrong')

  #Repeat obs diagnostics for each member of ensemble observational dataset if ensobs is set.
  #Assume missing data mask is the same as for main obs dataset.
  if ensobs != '':
    all_obs_files = glob.glob(ensobs+'*')
    if obs_file in all_obs_files: all_obs_files.remove(obs_file)

    for all_obs_file in all_obs_files:
        all_obs_cube = iris.load_cube(all_obs_file)
        all_obs_cube = preproc.extract_time(all_obs_cube, start_year, 1, 1, end_year, 12, 31)
        all_obs_cube = preproc.extract_shape(all_obs_cube, os.path.join(cfg['auxiliary_data_dir'], 'canada.shp'),
                                          method='contains', crop=True)
        all_obs_cube_area = preproc.annual_statistics(all_obs_cube, 'mean')
        all_obs_last_dec_cb = preproc.extract_time(all_obs_cube_area, 2010, 1, 1, 2019, 12, 31)
        all_obs_ref_per_cb = preproc.extract_time(all_obs_cube_area, 1850, 1, 1, 1900, 12, 31)
        ensobs_dec_warming.append(np.mean(all_obs_last_dec_cb.data) - np.mean(all_obs_ref_per_cb.data))
        all_obs_yearly_stat = preproc.annual_statistics(all_obs_cube_area, 'mean')
        all_obs_yearly_stat = all_obs_yearly_stat.data
        if int(diag_name[4:6]) > 1:
            all_obs_yearly_stat = np.mean(all_obs_yearly_stat.reshape(len(all_obs_yearly_stat)//int(diag_name[4:6]), -1), axis=1)

        ensobs_diag.append(all_obs_yearly_stat - np.mean(all_obs_yearly_stat))

  return (diag, obs_diag, dec_warming, obs_dec_warming, ann_warming, gmst_comp_warming)
