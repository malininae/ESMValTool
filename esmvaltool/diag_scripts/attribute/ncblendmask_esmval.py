import sys
import numpy as np
import iris
import scipy.stats
import math
import netCDF4
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

# cell areas, used for calculating area weighted averages
def areas( grid ):
  area = grid*[0.0]
  for i in range(grid):
    area[i] = ( ( math.sin(math.radians(180.0*(i+1)/grid-90.0)) -
                  math.sin(math.radians(180.0*(i  )/grid-90.0)) ) /
                math.sin(math.radians(180.0/grid)) )
  return area


def ncblendmask_esmval(tas_file,sftlf_file,obs_file,diag_name,obs='had4',ensobs='',ensobs_diag=[],ensobs_dec_warming=[]):
# MAIN PROGRAM

# m = mask
# a = blend anomalies
# f = fix ice
# (use x for none)

  tas_cube = iris.load_cube(tas_file)

  sftlf_cube = iris.load_cube(sftlf_file)

  obs_cube = iris.load_cube(obs_file)

  tas_cube = preproc.anomalies(tas_cube, period='monthly',
                                 reference=dict(start_year=1961, start_month=1, start_day=1,
                                                end_year=1990, end_month=12,end_day=31))

  # calculate area weights
  w = np.zeros_like(tas_cube.data)
  a = areas(sftlf_cube.shape[0])
  for m in range(w.shape[0]):
      for j in range(w.shape[2]):
        w[m,:,j] = a[:]

  wm=w.copy()
  # calculate diagnostic

  area_stat = preproc.area_statistics(tas_cube,operator='mean')

  if int(diag_name[4:6]) == 1:
    yearly_stat = preproc.annual_statistics(area_stat, 'mean')
    yearly_stat = yearly_stat.data

  diag = yearly_stat - np.mean(yearly_stat)

  last_dec_cb = preproc.extract_time(area_stat, 2010, 1, 1, 2019, 12, 31)
  ref_per_cb = preproc.extract_time(area_stat, 1850, 1, 1, 1900, 12, 31)

  dec_warming = np.mean(last_dec_cb.data) - np.mean(ref_per_cb.data)
  ann_warming = yearly_stat - np.mean(ref_per_cb.data)
  gmst_comp_warming = ann_warming


  obs_dec_warming = calc_dec_warming(obs_tas,wm)

  # Calculate warming in globally-complete blended data.
  obs_diag=calc_diag(obs_tas[0:tos.shape[0],:,:], wm, diag_name)
  # Liza's comment
  if np.max(diag) > 10:
    logger.error('Something went wrong')

  #Repeat obs diagnostics for each member of ensemble observational dataset if ensobs is set.
  #Assume missing data mask is the same as for main obs dataset.
  if ensobs != '':
    #Assume 100 member ensemble observations dataset.
    for ens in range(1,enssize+1):
      nc = netCDF4.Dataset(ensobs+str(ens)+'.nc', "r")
      if obs=='had5':
        obs_tas = nc.variables["tas"][:,:,:]
        #Make it work with HadCRUT5 - repeat last year in obs_tas
        obs_tas=np.concatenate((obs_tas,obs_tas[2016:2028,:,:]))
      else:
        obs_tas = nc.variables["temperature_anomaly"][:,:,:]
      nc.close()
      obs_tas=obs_tas[:,:,regrid_index] #Regrid to match esmvaltool output.
      ensobs_dec_warming.append(calc_dec_warming(obs_tas[0:tas.shape[0],:,:],wm))
      ensobs_diag.append(calc_diag(obs_tas[0:tas.shape[0],:,:],wm,diag_name))

  return (diag, obs_diag, dec_warming, obs_dec_warming, ann_warming, gmst_comp_warming)


def calc_diag(tos,wm,diag_name):
  #Calculate requested diagnostic from gridded SAT/SST.
  av_per=int(diag_name[4:6])*12 #Last two digits of diag_name are averaging period in yrs.
  #compute diagnostic based on masked/blended temperatures.
  if diag_name[0:4]=='gmst':
    nlat=1
  elif diag_name[0:4]=='hemi':
    nlat=2
  else:
    print ('Diagnostic ',diag_name,' not supported')
    exit ()
  nper=math.ceil(tos.shape[0]/av_per) #Round up number of averaging periods.
  diag=np.zeros((nlat,nper))
  # calculate temperatures
  for m in range(nper):
    for l in range(nlat):
      diag[l,m]=np.sum( wm[m*av_per:(m+1)*av_per,l*tos.shape[1]//nlat:(l+1)*tos.shape[1]//nlat,:] *
                           tos[m*av_per:(m+1)*av_per,l*tos.shape[1]//nlat:(l+1)*tos.shape[1]//nlat,:] ) / \
                np.sum( wm[m*av_per:(m+1)*av_per,l*tos.shape[1]//nlat:(l+1)*tos.shape[1]//nlat,:] )
  diag=diag-np.mean(diag,axis=1,keepdims=True)  # Take anomalies over whole period.
  diag=np.reshape(diag,nper*nlat)
  return diag
    # wm * tos / np.sum(wm, axis=2)[:, np.newaxis])


def calc_dec_warming(tas,w):
  gmt_mon=np.zeros(tas.shape[0])
  # calculate 2010-2019 mean relative to 1850-1900, assuming data starts in 1850.
  # If last decade is incomplete, just compute mean from available data.
  for m in range(tas.shape[0]):
      s = np.sum( w[m,:,:] )
      gmt_mon[m] = np.sum( w[m,:,:] * tas[m,:,:] ) / s
  return (np.nanmean(gmt_mon[(2010-1850)*12:(2020-1850)*12])-np.mean(gmt_mon[0:(1901-1850)*12]))


def calc_ann_warming(tas,w):
  #Calculate timeseries of annual mean GSAT.
  nyr=math.ceil(tas.shape[0]/12) #Round up number of years.
  diag=np.zeros(nyr)
  gsat_mon=np.zeros(tas.shape[0])
  # calculate temperatures
  for m in range(tas.shape[0]):
    s = np.sum( w[m,:,:] )
    gsat_mon[m] = np.sum( w[m,:,:] * tas[m,:,:] ) / s
  for m in range(nyr):
    diag[m]=np.mean(gsat_mon[m*12:(m+1)*12]) #Note - will calculate average over incomplete final year.
  diag=diag-np.mean(diag[0:(1901-1850)]) #Take anomalies relative to 1850-1901.
  return (diag)
