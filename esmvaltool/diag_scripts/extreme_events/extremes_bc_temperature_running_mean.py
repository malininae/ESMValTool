'''
Documentation of the temporary extreme event attribution script. 
Written by Elizaveta Malinina (ECCC) elizaveta.malinina-rieger@ec.gc.ca
This script is used to do a rapid extreme event attribution within esmvaltool.
The example input recipes can be found at 
https://github.com/malininae/ESMValTool/blob/extremes_bc/esmvaltool/recipes/malinina23/recipe_ab_extremes_txx_may_2023.yml
for TXx variable in the certain time range and 
https://github.com/malininae/ESMValTool/blob/extremes_bc/esmvaltool/recipes/malinina23/recipe_ab_extremes_tx7x_may_2023.yml
for TXNx (here N=7) in the certain time range. 

NB: currently, the group names are hard coded, please do not change them! 
The input parameters for the cfg dictionary are documented in the mentioned recipes. 

The output will consist of figures for in the standard "plot"
directory as well .csv tables with the GEV fit parameters in the "work" 
directory in esmvaltool_output directory. The information on how to run 
ESMValTool is available at https://docs.esmvaltool.org/en/latest/
The examples of the results can be found in Malinina and Gillett (2024), WACE
https://doi.org/10.1016/j.wace.2024.100642
'''


import csv
import esmvalcore.preprocessor as eprep
import iris
from iris.util import equalise_attributes
from iris.time import PartialDateTime
from datetime import timedelta
import cf_units
import cftime 
import climextremes as cex
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import genextreme as gev
from scipy.stats import kstest, cramervonmises

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, select_metadata, group_metadata, get_diagnostic_filename, save_data, ProvenanceLogger
import esmvaltool.diag_scripts.shared.plot as eplot
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import ProvenanceLogger

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def obtain_obs_info(groups, cfg):
    '''
    This function processes observational data, including GEV fitting

    Input: 
        groups:
           list of dictionaries, comes from ESMValTool input data sorting function
        cfg:
           config dictionary  
    
    Output:
        obs_gev_data:
            dictionary with observational cube, return periods and analysis
            year value and related statistics           
    '''

    # obtaining the observational TXNx group
    obs_abs_info = groups.pop('obs_abs')
    # obtaining the GSAT group (covariate to fit the non-stationary GEV)
    obs_gsat_info = groups.pop('obs_gsat')

    # loading the observational cube (assumption: only 1 per group given)
    raw_abs_obs_cb = iris.load_cube(obs_abs_info[0]['filename'])

    # selecting the year, for which analysis is done
    ana_year_const = iris.Constraint(time = lambda cell: cell.point.year == cfg['analysis_year'])
    ana_year_cube = raw_abs_obs_cb.extract(ana_year_const)

    # selecting the month for the analysis (e.g. 5 /May/)
    ana_month_const = iris.Constraint(time = lambda cell: cell.point.month == cfg['month'])
    abs_ana_month = ana_year_cube.extract(ana_month_const)

    # In case the specific time range is provided selecting the data for that
    # specific time range, rather than for the whole month
    if cfg.get('timerange'):
        clip_pdt_st = PartialDateTime(year=int(cfg['time_start'].split('-')[0]),
                                      month=int(cfg['time_start'].split('-')[1]),
                                      day=int(cfg['time_start'].split('-')[2]))
        clip_pdt_end = PartialDateTime(year=int(cfg['time_end'].split('-')[0]),
                                      month=int(cfg['time_end'].split('-')[1]),
                                      day=int(cfg['time_end'].split('-')[2]))
        clip_constr = iris.Constraint(time= lambda cell: clip_pdt_st<=cell.point <=clip_pdt_end)
        abs_ana_month = abs_ana_month.extract(clip_constr)

    # determine the day of the absolute maximum for the time range/month specified
    max_date_idx = abs_ana_month.coord('time').points[abs_ana_month.data.argmax()]
    max_date = cftime.num2pydate(max_date_idx, abs_ana_month.coord('time').units.origin, abs_ana_month.coord('time').units.calendar)

    # determine the window (+-) 15 days around the day of the observed maximum
    # in the selected time period and extracting it from the climatology cube
    start_date = max_date - timedelta(days=15) ; end_date = max_date + timedelta(days=15)
    pdt_start = PartialDateTime(month=start_date.month, day = start_date.day)
    pdt_end = PartialDateTime(month=end_date.month, day = end_date.day)
    pdt_constraint = iris.Constraint(time= lambda cell: pdt_start<=cell.point <=pdt_end)
    crop_abs_obs_cb = raw_abs_obs_cb.extract(pdt_constraint)

    # determining maxima in each year for the determined above time period window  
    abs_obs_cb = eprep.annual_statistics(crop_abs_obs_cb, operator='max')

    # calculating anomalies. The reference period comes from the recipe
    ano_obs_cb = eprep.anomalies(abs_obs_cb, 'full', 
                                reference = {'start_year': cfg['reference_period'][0], 
                                'start_month': 1, 'start_day':1, 
                                'end_year': cfg['reference_period'][1], 
                                'end_month': 12, 'end_day':31})
    
    # saving TXNx data into an array
    ano_obs_arr = ano_obs_cb.data

    # loading cube and saving it to pandas to use pandas internal function
    gsat_obs_cb = iris.load_cube(obs_gsat_info[0]['filename'])
    gsat_df = pd.DataFrame(gsat_obs_cb.data, columns=['gsat'])

    # smoothing gsat with kernel provided in the recipe
    kernel_size = cfg['smooth_gsat_years']
    gsat_smooth_df = gsat_df.rolling(kernel_size, min_periods=1).mean()
    gsat_smooth_arr = gsat_smooth_df['gsat'].to_numpy()

    # if analysing current year and need GSAT extrapolation
    if cfg.get('add_gsat_year'): 
        gsat_smooth_arr = np.append(gsat_smooth_arr, gsat_smooth_arr[-1])
    
    # determine the value for the analysis year
    ana_year_const = iris.Constraint(time = lambda cell: cell.point.year == cfg['analysis_year'])
    ana_year_value =  float(ano_obs_cb.extract(ana_year_const).data)
    # determine the index of the ana_year_value, if several, assume the latest
    ana_arg = max(np.where(ano_obs_arr == ana_year_value)[0])
    ana_gsat = gsat_smooth_arr[ana_arg]

    # calculate stationary gev 
    obs_stat = cex.fit_gev(ano_obs_arr, returnValue=ana_year_value, getParams=True)
    orig_stat_rp = np.exp(obs_stat['logReturnPeriod'][0]) # it is the only value, we are making it a float

    # calculate non-stationary gev
    obs_non_stat = cex.fit_gev(ano_obs_arr, gsat_smooth_arr, locationFun=1, returnValue=ana_year_value, getParams=True)
    orig_nonstat_rp = np.exp(obs_non_stat['logReturnPeriod'][ana_arg])

    # initializing the list to store bootstrap values and choosing random seed
    bootstrap_rps = list()
    rng = np.random.default_rng(501)

    # bootstrap
    for i in range(1000): 
        # choosing random indices within the length of the observed array
        # using the indices rather than values to use the appropriate covariate
        # see discussion on inclusion of the analsyis value in Malinina&Gillett(2024)
        for_fit_indices = rng.integers(low=0, high=len(ano_obs_arr), size=len(ano_obs_arr))
        for_fit_indices.sort()
        # fitting non-stationary GEV for individual bootstrap sample
        temp_gev = cex.fit_gev(ano_obs_arr[for_fit_indices], gsat_smooth_arr[for_fit_indices],
                               locationFun=1, initial={'location':float(np.around(obs_stat['mle'][0],2)),
                                            'scale':float(np.around(obs_stat['mle'][1],2)), 
                                            'shape':float(np.around(obs_stat['mle'][2],2))}, getParams=True)
        # the fit might be unsuccessful. If successful, save return period 
        # of the analysis values, if not assign nans
        try:
            temp_loc = temp_gev['mle'][0] + temp_gev['mle'][1]*ana_gsat
            temp_rp = np.around(1/gev.sf(ana_year_value, -1*temp_gev['mle'][3],
                                        loc= temp_loc, scale=temp_gev['mle'][2]), 1)
            bootstrap_rps.append(temp_rp)
        except:
            bootstrap_rps.append(np.nan)
    
    bootstrap_rps = np.asarray(bootstrap_rps)

    # determining percentiles, nans ommited (see docs for numpy nan stats why)
    rp_perc = np.nanpercentile(bootstrap_rps, [5,10,50,90,95], method='closest_observation').round(1)

    # saving the observational return periods and fits into a csv file
    era_csv = open(os.path.join(cfg['work_dir'], 'gev_era_data_'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'.csv'), 'w', newline='')
    era_csv_writer = csv.writer(era_csv, delimiter=',')
    era_csv_writer.writerow([str(cfg['analysis_year'])+' ERA5 '+cfg['ax_var_label']+' value '+str(ana_year_value)+ ', ERA5 smoothed GSAT value '+str(ana_gsat)])
    era_csv_writer.writerow(['ERA5 non-stationary GEV params'])
    era_csv_writer.writerow(list(obs_non_stat['mle_names']))
    era_csv_writer.writerow(list(obs_non_stat['mle']))
    era_csv_writer.writerow([str(cfg['analysis_year'])+' ERA5 return period', str(np.around(orig_nonstat_rp,1))])
    era_csv_writer.writerow(['Date of the '+str(cfg['analysis_year'])+' maximum: '+str(max_date)])
    era_csv_writer.writerow(['Bootstrapped uncertanties on ERA5 nonstationary return period'])
    era_csv_writer.writerow(['5_perc', '10_perc', '50_perc', '90_perc', '95_perc'])
    era_csv_writer.writerow(rp_perc)
    era_csv_writer.writerow(['ERA5 stationary GEV params'])
    era_csv_writer.writerow(list(obs_stat['mle_names']))
    era_csv_writer.writerow(list(obs_stat['mle']))
    era_csv_writer.writerow(['ERA5 stationary return period', str(np.around(orig_stat_rp,1))])
    era_csv.close()

    # to pass the data further on saving into a dictionary
    obs_gev_data={'gev_param_names' : obs_non_stat['mle_names'],
                  'gev_param_values': obs_non_stat['mle'],
                  'abs_obs_cb': abs_obs_cb,
                  'ana_year_value': ana_year_value, 
                  'ana_gsat_value': ana_gsat, 
                  'ana_year_RP': orig_nonstat_rp,
                  'ana_year_RP_CI': rp_perc,
                  'ano_obs_cb': ano_obs_cb,
                  'max_date': max_date,
                  'date_constr': pdt_constraint,
                  'smoothed_gsat':gsat_smooth_arr}

    return obs_gev_data


def bootstrap_gev(data_dic, ana_year_value):
    '''
    Calculates GEV parameters and return period uncertainties using bootstrap

    Input:
        data_dic:
            dictionary or CubeList with the single data for models
        ana_year_value: 
            float, analysis year value for which return periods are calculated
            
    Output:
        param_dic: 
            dictionary with the lists of the resulting GEV parameters and 
            return periods from each bootstrap realization
    ''' 

    if type(data_dic) == dict:  
        # determining the max length of the model realisation, the number of bootstrap
        # iterations is this value * 100
        max_cblst_len = np.asarray([len(data_dic[model]['data']) for model in data_dic.keys()]).max()
        iter_pool = max_cblst_len *100
        # determining number of models
        n_real = len(data_dic.keys())
        # determining number of years from first cube, assuming the same number
        # in all of the models
        n_years = np.asarray([data_dic[model]['data'][0].shape for model in data_dic.keys()]).max()
        # defining the size of the pool
        pool_size = int(np.around(n_real*n_years))
    elif type(data_dic) == iris.cube.CubeList: 
        iter_pool = np.max([len(data_dic) *100, 1000])
        # determining number of years from the first cube in the cubelist
        n_years = data_dic[0].shape[0]
        n_real = 3 # this is a number of realisations which shown to be enough to draw conclusions
        # in case the model has less than 3 realisations, pool size has to be
        # the same as numebr of realisation, if not, we just look at 3 
        # realisations to reduce time running and memory consumption
        if len(data_dic)<n_real: 
            pool_size = len(data_dic) * n_years
        else: 
            pool_size = n_real * n_years

    # creating array with the all the TXNx values, done for ease of indexing 
    pool_data = list()
    if type(data_dic) == dict: 
        for model in data_dic.keys():
            for i in range(len(data_dic[model]['data'])):
                pool_data.append(data_dic[model]['data'][i].data)
    elif type(data_dic) == iris.cube.CubeList:
        for cb in data_dic: 
            pool_data.append(cb.data)
    pool_data = np.asarray(pool_data)

    # initializing arrays where GEV parameters and return periods from each
    # bootstrap iterations are stored
    shapes = np.zeros(iter_pool)
    locs = np.zeros(iter_pool)
    scales = np.zeros(iter_pool)
    rps = np.zeros(iter_pool)

    # defining the seed for the random sequence
    rng = np.random.default_rng(501)

    # Bootstrap sequence for GEV fitting (done with replacement)
    for i in range(0, iter_pool):
        # initializing the list for individual bootstrap sample 
        sample_data = list()
        # choosing the random indices from the array with all data
        mod_idxs = rng.choice(pool_data.shape[0], size = pool_size)
        # for each random model index choosing random years
        for mod_idx in mod_idxs: 
            y_idx = rng.choice(n_years)
            # adding the randomly selected data into the sample data array
            sample_data.append(pool_data[mod_idx, y_idx])
        sample_data = np.asarray(sample_data).flatten()
        # fitting stationary GEV distribution and obtaining return period
        # for the event of interest
        gev_params = cex.fit_gev(sample_data, returnValue=ana_year_value, getParams=True)
        # if fit was sucessful, saving the GEV parameters and return periods,
        # if not assign nans
        try:
            shapes[i] = gev_params['mle'][2]; locs[i]=gev_params['mle'][0]
            scales[i] = gev_params['mle'][1]; rps[i] = np.exp(gev_params['logReturnPeriod'][0])
        except: 
            shapes[i] =np.nan ; locs[i] = np.nan ; scales[i] = np.nan ; rps[i] = np.nan 

    # saving the GEV parameters from each bootsrap iteration to a dictionary
    param_dic = {'loc': locs, 'scale': scales, 'shape': shapes, 'return_periods': rps}    

    return param_dic


def make_uncert_figures(data_dic, cfg, border):
    '''
    Plots GEV distributions from single bootstrap samples and calculates
    uncertanty bands for GEV parameters.

    Input: 
        data_dic: 
            dictionary with all the model data and parameters for each
            bootstrap iteration
        cfg:
            config dictionary coming from ESMValTool run
        border:
            list with min and max values encountered in the model data
    
    Output:
        uncert_band:
            dictionary with uncertanties around GEV parameters, pdf and 
            survival function fits. 
    '''

    # defining the colors for each group, temporary solution will move to yml
    colors = {}
    colors['all'] = (196 / 255, 121 / 255, 0)
    colors['nat'] = (0, 79 / 255, 0)
    colors['ssp245'] = (69 / 255, 118 / 255, 191 / 255)

    # listing all model experiments and models in the data dictionary
    exp_list = list(data_dic.keys())  ; exp_list.remove('obs_info') 
    models = list(data_dic[exp_list[0]].keys())

    # creating the values for which GEV is plotted and stored, same for all
    x_gev = np.arange(border[0],border[1], 0.1)

    # location of the text box, will be moved into the relative terms later
    tlocs = {'all': 0.04 , 'nat': 0.01,  'ssp245': 0.08}
    # initializing a dictionary for each experiment (essentially, climate)
    uncert_band = {}
    for exp in exp_list: 
        uncert_band[exp] = {}

    gev_params = ['shape', 'loc', 'scale']

    # plotting figures and saving uncertanties for each model (and multi-model)    
    for model in models: 

        # this is a figure where we will plot single distributions from bootstrap
        fig_single_bootstrap, ax_single_bootstrap = plt.subplots(1)
        fig_single_bootstrap.set_size_inches(12., 8.)

        # this a figure where we plot how the values for GEV params are distributed 
        fig_gev_distr, ax_gev_distr = plt.subplots(len(gev_params))
        fig_gev_distr.set_size_inches(8., 12.)

        for exp in exp_list: 
            # removing the uncertanty from the data dictionary to use here
            gev_dic = data_dic[exp][model].pop('uncert')
            # removing a level in the main dictionary by saving the data there
            data_dic[exp][model] = data_dic[exp][model].pop('data')
            # here we plot distribution of single GEV params
            for n, gev_param in enumerate(gev_params):
                ax_gev_distr[n].hist(gev_dic[gev_param], bins=50, edgecolor='none',
                        facecolor = colors[exp], alpha=0.3, label = exp, density=True)
                ax_gev_distr[n].set_xlabel(gev_param)
                ax_gev_distr[n].set_ylabel('Number density')
                ax_gev_distr[n].set_title('GEV parameter: ' + gev_param)

            # this are arrays where we'll save all pdfs and survival functions
            # to calculate 5/95 perc later 
            all_pdfs = np.zeros((len(x_gev), len(gev_dic['loc'])))
            all_sfs = np.zeros((len(x_gev), len(gev_dic['loc'])))

            # here we plot single distributions
            for i in range(len(gev_dic['loc'])):
                # calculating the pdfs and sf for each bootstrap sample
                # -1 infront of shape, because scipy and climextremes use 
                # opposite definitions of shape 
                gev_pdf = gev.pdf(x_gev, -1*gev_dic['shape'][i],gev_dic['loc'][i], gev_dic['scale'][i])
                gev_sf = gev.sf(x_gev, -1*gev_dic['shape'][i],gev_dic['loc'][i], gev_dic['scale'][i])
                # for each iteration saving the resulting pdfs and sfs
                all_pdfs[:, i] = gev_pdf
                all_sfs[:, i] = gev_sf
                ax_single_bootstrap.plot(x_gev, gev_pdf, color = colors[exp], alpha=0.03)
              
            # saving uncertainties (5/95 perc) for pdf, sf and GEV parameters
            uncert_band[exp][model] = {'x_gev': x_gev, 'pdf_5th_perc' : np.nanpercentile(all_pdfs, 5, axis = 1, method='closest_observation'), 
                                                'pdf_95th_perc': np.nanpercentile(all_pdfs, 95, axis = 1, method='closest_observation'),
                                                'sf_5th_perc' : np.nanpercentile(all_sfs, 5, axis = 1, method='closest_observation'), 
                                                'sf_95th_perc' : np.nanpercentile(all_sfs, 95, axis = 1, method='closest_observation'),
                                                'loc_min': gev_dic['loc'].min(), 'loc_max': gev_dic['loc'].max(),
                                                'loc_5': np.nanpercentile(gev_dic['loc'], 5, method='closest_observation'),
                                                'loc_95': np.nanpercentile(gev_dic['loc'], 95, method='closest_observation'),
                                                'scale_min': gev_dic['scale'].min(), 'scale_max': gev_dic['scale'].max(),
                                                'scale_5': np.nanpercentile(gev_dic['scale'], 5, method='closest_observation'),
                                                'scale_95': np.nanpercentile(gev_dic['scale'], 95, method='closest_observation'),
                                                'shape_min': gev_dic['shape'].min(), 'shape_max': gev_dic['shape'].max(),
                                                'shape_5': np.nanpercentile(gev_dic['shape'], 5, method='closest_observation'),
                                                'shape_95': np.nanpercentile(gev_dic['shape'], 95, method='closest_observation'),
                                                'return_periods_all': gev_dic['return_periods']}
    
            # adding the text on GEV statistics to the plots
            param_str = '                             '+exp
            param_str += '\nshape mean:'+str(np.around(gev_dic['shape'].mean(),3))+', max:'+str(np.around(gev_dic['shape'].max(),3)) + ', min:' + str(np.around(gev_dic['shape'].min(),3)) 
            param_str += '\n loc mean:'+str(np.around(gev_dic['loc'].mean(),3))+', max:'+str(np.around(gev_dic['loc'].max(),3)) + ', min:' + str(np.around(gev_dic['loc'].min(),3)) \
                + '\nscale mean:'+ str(np.around(gev_dic['scale'].mean(),3))+', max:'+str(np.around(gev_dic['scale'].max(),3)) + ', min:' + str(np.around(gev_dic['scale'].min(),3))               
             
            ax_single_bootstrap.text(-0.95*border[0], tlocs[exp], param_str, color = colors[exp])

        # adjusting plot parameters and saving figures
        ax_gev_distr[0].legend(loc=0, fancybox=False, frameon=False)
        fig_gev_distr.suptitle('Distribution of GEV parameters after bootstrap in ' +model,
                    fontsize = 'x-large')
        fig_gev_distr.set_dpi(250)
        plt.tight_layout()
        fig_gev_distr.savefig(os.path.join(cfg['plot_dir'], 'figure_'+cfg['region'].lower()+'_extreme_distr_param_gev_'+ model + diagtools.get_image_format(cfg)))
        plt.close(fig_gev_distr)

        ax_single_bootstrap.legend(loc=0, fancybox=False, frameon=False)
        ax_single_bootstrap.set_xlim(border[0], border[1])
        ax_single_bootstrap.set_ylim(0, ax_single_bootstrap.get_ylim()[1])
        ax_single_bootstrap.set_xlabel(cfg['ax_var_label']+' anomaly, '+ cfg['var_units'])
        ax_single_bootstrap.set_ylabel('Number density')

        fig_single_bootstrap.suptitle('Estimation of GEV fit uncertainty from '+model+' with Bootstrap method',
                    fontsize = 'x-large')
        fig_single_bootstrap.set_dpi(250)

        fig_single_bootstrap.savefig(os.path.join(cfg['plot_dir'], 'figure_'+cfg['region'].lower()+'_extremes_bootstrap_gev_'+ model + diagtools.get_image_format(cfg)))
        plt.close(fig_single_bootstrap)
                            
    return uncert_band


def make_hist_figure(data_dic, cfg, uncert_band, border, apr_param):
    '''
    Plots the main attribution figure (Ex.: Fig 9 Malinina&Gillett (2024)),
    calculates the risk ratios and saves the appropriate statistics in csv.

    Input:
        data_dic:
            dictionary with all the model and obs data 
        cfg:
            config dictionary coming from ESMValTool
        border:
            minimum and maximum values encountered over the data
        apr_param:
            initial parameters for the GEV fit, not used currently
    '''

    # pulling data from the observational dict and the magnitude of the event
    obs_info_arr = data_dic['obs_info']
    event = obs_info_arr['ana_year_value']

    # event probability from observations and event's return period
    era_event_prob = 1/obs_info_arr['ana_year_RP']
    era_event_RP = obs_info_arr['ana_year_RP']

    # opening a csv file with all return periods and intensities
    risk_csv = open(os.path.join(cfg['work_dir'], 'GEV_'+cfg['region']+'_'+cfg['ax_var_label'] +'_risk_data.csv'), 'w', newline='')
    risk_csv_writer = csv.writer(risk_csv, delimiter=',')
    risk_head_row = ['model'] # header for the file

    # opening a csv file with risk ratios uncertainties
    risk_uncert_csv = open(os.path.join(cfg['work_dir'], 'gev_'+cfg['region']+'_'+cfg['ax_var_label'] +'_uncert_risk_data.csv'), 'w', newline='')
    risk_uncert_csv_writer = csv.writer(risk_uncert_csv, delimiter=',')
    risk_uncert_head_row = ['model','all/nat_r_r_5', 'all/nat_r_r_10', 'all/nat_r_r_50','all/nat_r_r_90', 'all/nat_r_r_95',
                            'ssp/nat_r_r_5', 'ssp/nat_r_r_10', 'ssp/nat_r_r_50', 'ssp/nat_r_r_90', 'ssp/nat_r_r_95',
                            'ssp/all_r_r_5', 'ssp/all_r_r_10', 'ssp/all_r_r_50', 'ssp/all_r_r_90', 'ssp/all_r_r_95']
    risk_uncert_csv_writer.writerow(risk_uncert_head_row)

    # listing all experiments (climates)
    exp_list = list(data_dic.keys()) ; exp_list.remove('obs_info')    

    # creating quantile measures for QQ plot, numpy not accepting 0
    quantile_measures = np.arange(0, 1.01, 0.01); quantile_measures[0] = 0.001

    # initializing colors, will be moved to yml
    colors = {'all' : (196 / 255, 121 / 255, 0), 
              'nat' : (0, 79 / 255, 0), 
              'ssp245' : (69 / 255, 118 / 255, 191 / 255)}

    # opening csv file with all GEV parameters values and their uncertainties
    csv_file = open(os.path.join(cfg['work_dir'], 'gev_'+cfg['region']+'_'+cfg['ax_var_label'] +'_parameters.csv'), 'w', newline='')
    gevs_csv_writer = csv.writer(csv_file, delimiter=',')
    head_row = ['model']
    for exp_key in exp_list:
        head_row.extend(['shape_'+exp_key, 'shape_min_'+exp_key, 'shape_5_'+exp_key, 'shape_95_'+exp_key,'shape_max_'+exp_key])
        head_row.extend(['loc_'+exp_key,'loc_min_'+exp_key,'loc_5_'+exp_key,'loc_95_'+exp_key,'loc_max_'+exp_key])
        head_row.extend(['scale_'+exp_key,'scale_min_'+exp_key,'scale_5_'+exp_key,'scale_95_'+exp_key,'scale_max_'+exp_key])
        head_row.extend(['ks_stat_'+exp_key, 'ks_pvalue_'+exp_key, 'cvm_stat_'+exp_key, 'cvm_pvalue_'+exp_key])
        risk_head_row.append(exp_key+'_prob')
        risk_head_row.extend([exp_key+'_return_p', exp_key+'_return_p_5', exp_key+'_return_p_95', exp_key+'_return_p_5',exp_key+'_return_p_95'])
        risk_head_row.extend([exp_key+'_intens', exp_key+'_intens_5',exp_key+'_intens_95'])
    gevs_csv_writer.writerow(head_row)
    risk_csv_writer.writerow(risk_head_row)
    # listing the models (including multi-model ensemble)
    models = data_dic[exp_list[0]].keys()
    for model in models: 
        # opening the figure and creating layout
        fig = plt.figure(constrained_layout=False)
        fig.set_size_inches(12., 8.)
        gs = fig.add_gridspec(nrows=2, ncols=6)
        ax_hist = fig.add_subplot(gs[:, :-2])
        ax_qq= fig.add_subplot(gs[0, -2:])
        ax_surv= fig.add_subplot(gs[1, -2:])
        model_row = [model]
        risk_model_row = [model]
        # looping over each experiment (climate)
        for exp_key in exp_list:
            # pulling the data for the ensemble
            ens_cubelist = data_dic[exp_key][model]
            # initializing the arrays to fit GEV distribution
            distrib_data = []
            weights = [] # weights in case model weighting is chosen
            for cube in ens_cubelist:
                # calculating model weight in case model weighting is chosen
                if (model == 'Multi-Model-Mean')&(cfg['model_weighting']):
                    cube_weight = cube.attributes['ensemble_weight']*cube.attributes['reverse_dtsts_n']
                else:
                    # otherwise each ensemble member has a weight of 1
                    # independent of the size of the ensembles for multi-model
                    cube_weight = 1
                for point in cube.data:
                    distrib_data.append(np.around(point, 1))
                    weights.append(cube_weight/len(cube.data))
            distrib_data = np.asarray(distrib_data)
            weights = np.asarray(weights)
            if cfg['model_weighting']:
                # saving the data for the weighted arrays, the behaviour is 
                # unsupported and will be re-arranged for in the future updates
                # current idea: populate an array based on the weights
                un_weights = np.unique(weights)
                rev_un_weights = np.asarray(1/un_weights).round(0).astype('int32')
                large_denom = np.gcd.reduce(rev_un_weights)
                dev_weights = rev_un_weights/large_denom
                least_mult = np.lcm.reduce(dev_weights.astype('int32'))
                un_factors = least_mult/dev_weights
                factors = np.zeros(len(weights))
                for n_w, un_wght in enumerate(un_weights):
                    factors[np.where(weights==un_wght)] = un_factors[n_w]
                factors = factors.astype('int32')
                upd_distr_data = list()
                new_weights = list()
                for n_dp, distrib_point in enumerate(distrib_data): 
                    for f in range(factors[n_dp]):
                        upd_distr_data.append(distrib_point)
                        new_weights.append(weights[n_dp]/factors[n_dp])
                distrib_data = np.asarray(upd_distr_data)
                weights = np.asarray(new_weights)
            # pulling an x for pdfs
            x_gev = uncert_band[exp_key][model]['x_gev']
            # fitting stationary GEV distribution to the model data, getting RP
            w_distr_par = cex.fit_gev(distrib_data, returnValue=event, getParams=True)
            # saving data from the fit, if fit unsucessful assign nans
            try:
                w_distr_loc = w_distr_par['mle'][0]; w_distr_scale = w_distr_par['mle'][1]; w_distr_shape = w_distr_par['mle'][2]
            except:
                w_distr_loc = np.nan ; w_distr_scale = np.nan ; w_distr_shape = np.nan
            # calculating pdf. -1 infront of shape due to opposite definitions
            # of shape in scipy and climextremes
            w_pdf = gev.pdf(x_gev, -1*w_distr_shape, loc=w_distr_loc, scale=w_distr_scale)
            # calculating survavila finction (needed for RP curve)
            w_survival = gev.sf(x_gev, -1*w_distr_shape, loc=w_distr_loc, scale=w_distr_scale)
            # calculating theoretical quantiles of the distribution
            theor_quants = gev(-1*w_distr_shape, loc=w_distr_loc, scale=w_distr_scale).ppf(quantile_measures)
            # calculating Kolmogorov-Smirnov and Cramer-von-Mises statistics
            ks_res = kstest(distrib_data, gev(-1*w_distr_shape, loc=w_distr_loc, scale=w_distr_scale).cdf)
            cvm_res = cramervonmises(distrib_data, gev(-1*w_distr_shape, loc=w_distr_loc, scale=w_distr_scale).cdf)
            # saving the GEV statistics data for individual model into the csv
            model_row.extend([w_distr_shape, uncert_band[exp_key][model]['shape_min'], uncert_band[exp_key][model]['shape_5']])
            model_row.extend([uncert_band[exp_key][model]['shape_95'], uncert_band[exp_key][model]['shape_max']])
            model_row.extend([w_distr_loc, uncert_band[exp_key][model]['loc_min'], uncert_band[exp_key][model]['loc_5']])
            model_row.extend([uncert_band[exp_key][model]['loc_95'], uncert_band[exp_key][model]['loc_max']])            
            model_row.extend([w_distr_scale, uncert_band[exp_key][model]['scale_min'], uncert_band[exp_key][model]['scale_5']])
            model_row.extend([uncert_band[exp_key][model]['scale_95'], uncert_band[exp_key][model]['scale_max']])            
            model_row.extend([ks_res.statistic, ks_res.pvalue, cvm_res.statistic, cvm_res.pvalue])
            # plotting histogram of all the TXNx values for the model
            n_bins = np.arange(border[0], border[1]+0.1, 1)
            ax_hist.hist(distrib_data, bins=n_bins, edgecolor=colors[exp_key],
                    facecolor = colors[exp_key], alpha=0.3, label=cfg['name_' + exp_key] , density=True, weights=weights, zorder = 2)
            # plotting on top the pdf of the fit
            ax_hist.plot(x_gev, w_pdf, c = colors[exp_key], ls = 'solid', label = 'GEV fit '+cfg['name_' + exp_key], zorder=3)
            pdf_perc_5 = uncert_band[exp_key][model]['pdf_5th_perc']
            pdf_perc_95 = uncert_band[exp_key][model]['pdf_95th_perc']
            sf_perc_5 = uncert_band[exp_key][model]['sf_5th_perc']
            sf_perc_95 = uncert_band[exp_key][model]['sf_95th_perc']
            # plotting uncertainties of the pdf, calculated with bootstrap
            ax_hist.fill_between(x_gev, pdf_perc_5, pdf_perc_95, color=colors[exp_key], alpha=0.3, linewidth=0, zorder=4)
            # plotting uncertainties of the survival function (RP curve)
            ax_surv.fill_between(x_gev, 1/sf_perc_5, 1/sf_perc_95, color=colors[exp_key], alpha=0.3, linewidth=0, zorder=4)
            # calculating the intensity of the event with same RP as observed
            # e.g., RP=100y, how warm will the 1 in 100-y event be in models?
            intens = x_gev[np.argmin(np.abs(w_survival-era_event_prob))]
            intens_95 = x_gev[np.argmin(np.abs(sf_perc_95-era_event_prob))]
            intens_5 = x_gev[np.argmin(np.abs(sf_perc_5-era_event_prob))]
            # saving return periods, intensities and their uncertainties to csv
            try:
                risk_model_row.extend([1/np.exp(w_distr_par['logReturnPeriod'][0]), np.exp(w_distr_par['logReturnPeriod'][0]), np.nanpercentile(uncert_band[exp_key][model]['return_periods_all'],5),
                                    np.nanpercentile(uncert_band[exp_key][model]['return_periods_all'],95, method='closest_observation'), intens, intens_5, intens_95])
            except:
                risk_model_row.extend([np.nan, np.nan, np.nanpercentile(uncert_band[exp_key][model]['return_periods_all'],5),
                                    np.nanpercentile(uncert_band[exp_key][model]['return_periods_all'],95, method='closest_observation'), intens, intens_5, intens_95])
            # calculating practical quantiles of the data (for QQ plot)
            pract_quants = np.quantile(distrib_data, quantile_measures)
            # plotting QQ plot (theortical vs practical quantiles)
            ax_qq.scatter(theor_quants, pract_quants, edgecolors=colors[exp_key], marker='o', facecolors='None', lw=0.75, label=cfg['name_' + exp_key], zorder=3)
            # plotting best estimate return period curve
            ax_surv.plot(x_gev, 1/w_survival, color=colors[exp_key], zorder=3)
        gevs_csv_writer.writerow(model_row)
        risk_csv_writer.writerow(risk_model_row)

        # technical aspects of the plot, limits, captions etc.
        ylims = ax_hist.get_ylim()
        ax_hist.set_ylim(*ylims)

        ax_hist.text(border[0]/1.22, ylims[1]*0.65,'  Number of\nrealisations ' +str(len(ens_cubelist)), fontsize='large')
        ax_hist.vlines(event, *ylims, color = 'indianred', linestyle = 'solid', lw=1.5, zorder=1, label = 'ERA5 ('+ str(cfg['analysis_year'])+')')

        ax_surv.vlines(event, 0.1, era_event_RP, linestyle = 'solid', color='indianred', zorder=2,  label = 'ERA5 ('+ str(cfg['analysis_year'])+')')
        ax_surv.hlines(era_event_RP, x_gev[0], event,linestyle = 'solid', color='indianred', zorder=2)

        ax_hist.legend(loc=2, fancybox=False, frameon=False)
        ax_qq.legend(loc=2, fancybox=False, frameon=False, handletextpad=0.01)
        ax_qq.plot(border, border, c='tab:grey', zorder=1)
        ax_qq.grid(color='silver', axis='both', alpha=0.5)
        ax_hist.set_xlim(border[0]/1.2, border[1]/1.2)
        ax_qq.set_xlim(border[0]/1.2, border[1]/1.2)
        ax_qq.set_ylim(border[0]/1.2, border[1]/1.2)
        ax_surv.set_ylim(1,10000)
        ax_surv.set_xlim(border[0]/1.2, border[1]/1.2)
        ax_surv.set_yscale('log')
        ax_surv.grid(color='silver', axis='both', alpha=0.5)
        ax_hist.set_title('(a) Probability density function')
        ax_qq.set_title('(b) Quality assessment')
        ax_qq.set_ylabel('Data quantile')
        ax_qq.set_xlabel('GEV quantile')
        ax_surv.set_title('(c) Return period')
        ax_surv.set_ylabel('years')
        ax_surv.set_xlabel(cfg['ax_var_label'] + ' anomaly, ' +cfg['var_units'])
        ax_hist.set_xlabel(cfg['ax_var_label'] + ' anomaly, ' +cfg['var_units'])
        ax_hist.set_ylabel('Number density')

        if model == 'Multi-Model-Mean':
            fig.suptitle(cfg['ax_var_label']+' anomalies in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ ' calculated from '+str(len(models) - 1) +' CMIP6 models' , fontsize = 'x-large')
        else:
            fig.suptitle(cfg['ax_var_label']+' anomalies in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ ' calculated from '+model, fontsize = 'x-large')
        fig.set_dpi(250)

        plt.tight_layout()
        
        fig.savefig(os.path.join(cfg['plot_dir'], 'figure_'+cfg['region']+'_'+cfg['ax_var_label'] +'_extremes_gev_'+model + diagtools.get_image_format(cfg)))

        # calculating uncertainties of the risk ratios (RR). Best estimate RR 
        # are calculated later after the run from the return periods csv 
        # now hard coded, will be updated in the next iterations.
        all_to_nat = list(); ssp_to_nat = list(); ssp_to_all = list()
        for i in range(len(uncert_band['all'][model]['return_periods_all'])):
                all_to_nat.append(uncert_band['nat'][model]['return_periods_all'][i]/uncert_band['all'][model]['return_periods_all'][i])
                ssp_to_nat.append(uncert_band['nat'][model]['return_periods_all'][i]/uncert_band['ssp245'][model]['return_periods_all'][i])
                ssp_to_all.append(uncert_band['all'][model]['return_periods_all'][i]/uncert_band['ssp245'][model]['return_periods_all'][i])
        all_to_nat_perc = np.nanpercentile(all_to_nat, [5,10,50,90,95],method='closest_observation')
        ssp_to_nat_perc = np.nanpercentile(ssp_to_nat, [5,10,50,90,95],method='closest_observation')
        ssp_to_all_perc = np.nanpercentile(ssp_to_all, [5,10,50,90,95],method='closest_observation')
        risk_uncert_row = [model]
        risk_uncert_row.extend(np.concatenate((all_to_nat_perc, ssp_to_nat_perc, ssp_to_all_perc)))
        risk_uncert_csv_writer.writerow(risk_uncert_row)
    
    risk_uncert_csv.close()

    return


def make_era_dist_figure(obs_info_dic, cfg, border):
    '''
    Plots the observational figure with GEV fit and timeseries (Ex: Fig. 3)

    Input: 
        obs_info_dic:
            dictionary with observational information
        cfg:
            config dictionary coming from ESMValTool
        border:
            minimum and maximum values encountered over the data
    '''

    # absolute TXNx values are in K converting to C
    abs_cube = obs_info_dic['abs_obs_cb'] - 273.15
    # deriving the values for the event of the interest
    ana_year_const = iris.Constraint(time = lambda cell: cell.point.year == cfg['analysis_year'])
    ana_year_value = abs_cube.extract(ana_year_const).data; ana_arg = np.max(np.where(abs_cube.data == ana_year_value)[0])

    # redefining the borders values (before anomalies, now absolute)
    border[0] = np.floor(abs_cube.data.min()*0.9)
    border[1] = np.ceil(abs_cube.data.max()*1.1)
    # initializing the x for survivor function and bins for histogram
    x_gev_fine = np.arange(border[0], border[1]+0.1, 0.1)
    n_bins = np.arange(border[0], border[1]+0.1, 1)

    # fitting non-stationary GEV (doing it again, because here it's absolute values)
    era_param = cex.fit_gev(abs_cube.data, obs_info_dic['smoothed_gsat'], locationFun=1, returnValue=float(ana_year_value), getParams=True)
    # obtaining ERA return period for the event of interest
    era_RP = np.exp(era_param['logReturnPeriod'][ana_arg])
    # calculating the location parameter for the year of interest
    era_loc = obs_info_dic['ana_gsat_value'] * era_param['mle'][1] + era_param['mle'][0] 
    era_scale = era_param['mle'][2] ; era_shape = era_param['mle'][3] 
    # calculating the RP curve for the year of interest, -1 due to opposite 
    # shape definitions in scipy and climextremes
    era_sf = gev.sf(x_gev_fine,  -1*era_shape, loc = era_loc, scale = era_scale)

    # getting the years in the observational cube
    tims = cf_units.num2pydate(abs_cube.coord('time').points, abs_cube.coord('time').units.origin, abs_cube.coord('time').units.calendar)
    years = np.asarray([t.year for t in tims])

    # creating the plot
    fig_era = plt.figure(constrained_layout=False)
    fig_era.set_size_inches(12., 8.)
    gs = fig_era.add_gridspec(nrows=2, ncols=5)
    ax_era_hist = fig_era.add_subplot(gs[0, :-2])
    ax_era_surv= fig_era.add_subplot(gs[0, -2:])
    ax_era_tseries= fig_era.add_subplot(gs[1, :])

    ax_era_hist.hist(abs_cube.data, bins=n_bins, edgecolor='indianred', facecolor='indianred', alpha=0.3, density=True)
    ax_era_hist.scatter(ana_year_value, 0, c='indianred', marker='x',lw=2, s=100, label=str(cfg['analysis_year']), clip_on=False, zorder=4)
    ax_era_hist.set_xlim(*border)
    ax_era_hist.set_xlabel(cfg['ax_var_label'] +', ' + cfg['var_units'])
    ax_era_hist.set_ylabel('Number density')
    ax_era_hist.set_title('(a) Histogram')

    ax_era_surv.plot(x_gev_fine[era_sf>0.00001], 1/era_sf[era_sf>0.00001], c='indianred')
    ax_era_surv.set_title('(b) ERA5 ' +  cfg['ax_var_label'] +' return period in '+ str(cfg['analysis_year']))
    ax_era_surv.set_xlabel(cfg['ax_var_label'] +', ' + cfg['var_units'])
    ax_era_surv.set_ylabel('years')
    ax_era_surv.scatter(ana_year_value, era_RP, c='indianred', marker='x',lw=2, s=100, label= str(cfg['analysis_year']), clip_on=False, zorder=4)
    ax_era_surv.text(border[1]*0.8, 5,'ERA5 return period\n    '+str(np.around(era_RP, 1))+ ' years', color='k')
    ax_era_surv.set_yscale('log')
    ax_era_surv.grid(color='silver', axis='both', alpha=0.5)
    ax_era_surv.set_ylim(1,10000)
    ax_era_surv.set_xlim(*border)

    ax_era_tseries.set_title('(c) ERA5 ' + cfg['ax_var_label'] +' timeseries')
    ax_era_tseries.plot(years, abs_cube.data, c='indianred')
    ax_era_tseries.grid(color='silver', axis='both', alpha=0.5)
    ax_era_tseries.set_xlabel('time')
    ax_era_tseries.set_ylabel(cfg['ax_var_label']+', ' + cfg['var_units'])
    ax_era_tseries.scatter(years[np.argmax(abs_cube.data)], abs_cube.data.max(), edgecolors='indianred', marker='o', facecolors='None', s=100, lw=2)
    ax_era_tseries.set_xlim(years[0]-0.5, years[-1]+0.5)
    ax_era_tseries.set_ylim(*border)
    # adding arrow for the event of interest
    ax_era_tseries.arrow(years[-1] - len(years)*0.1, abs_cube.data.mean()*0.2 + 0.8*abs_cube.data.max(), len(years)*0.09 ,
                                     0.19*abs_cube.data.max()-abs_cube.data.mean()*0.2, color='k', length_includes_head=True,  
                                                                                        head_width=0.5, head_length=0.5)
    ax_era_tseries.text(years[-1] - len(years)*0.16, abs_cube.data.mean()*0.2 + 0.77*abs_cube.data.max(), 
                        'ERA5 '+ str(cfg['analysis_year'])+': '+str(np.around(abs_cube.data[ana_arg],1))+' '+cfg['var_units'])

    fig_era.suptitle('ERA5 ' + cfg['ax_var_label'] + ' in ' + cfg['region'] + ' and its GEV fit', fontsize = 'x-large')

    plt.tight_layout()

    fig_era.savefig(os.path.join(cfg['plot_dir'], 'figure_'+cfg['region']+'_'+cfg['ax_var_label'] +'_era' + diagtools.get_image_format(cfg)))

    return


def main(cfg):

    # deriving the filepaths and metadata on the input files and sorting groups
    input_data = cfg['input_data']

    groups = group_metadata(input_data.values(), 'variable_group', sort=True)

    # processing observational information including fitting GEV, output: dictionary
    obs_info = obtain_obs_info(groups, cfg)

    # obtain filespaths for anomaly files. For now: a separate group due to memory
    anomalies = groups.pop('anomaly')

    groups_l = list(groups.keys())

    mins = list(); maxs = list()

    # initializing the data dictionary
    plotting_dic = {}

    # initializing the dictionary to have initial conditions for model GEV fit 
    # not required, but good information to keep in case fit not successful
    fit_param_apr = {}

    # looping over the groups, which are different climates
    for group in groups_l:
        plotting_dic[group] = {}
        group_data = groups[group]
        datasets = group_metadata(group_data, 'dataset')
        # initializing a CubeList for the multi-model ensemble
        ens_cubelist = iris.cube.CubeList()
        for dataset in datasets.keys():
            filepaths = list(group_metadata(datasets[dataset], 'filename').keys())
            # number of realisations used to calculate weights if applicable
            n_real = len(filepaths)
            # initializing CubeList for single model ensemble
            mod_cubelist = iris.cube.CubeList()
            for filepath in filepaths:
                mod_cb = iris.load_cube(filepath)
                # extracting the dates within a selected timerange and obtaining annual 
                # maxima in that time range
                mod_cb = eprep.annual_statistics(mod_cb.extract(obs_info['date_constr']), operator='max')
                # obtaining file metadata to get the realization number (e.g. r1i1p1f1)
                file_metadata = select_metadata(datasets[dataset], filename = filepath)
                ens = file_metadata[0]['ensemble']
                # loading corresponding anomaly filename
                anom_cb = iris.load_cube(select_metadata(anomalies, dataset=dataset, ensemble=ens)[0]['filename'])
                # extracting the dates within a selected timerange and obtaining annual 
                # maxima in that time range for anomaly period 
                anom_cb = eprep.annual_statistics(anom_cb.extract(obs_info['date_constr']), operator='max')
                # obtaining mean for anomaly timerange
                anom_cb = eprep.climate_statistics(anom_cb, operator='mean', period='full')
                # subtracting anomaly from the individual model cube
                mod_cb = mod_cb-anom_cb
                # saving min and max to use later to calculate the fit borders
                mins.append(mod_cb.collapsed('time', iris.analysis.MIN).data)
                maxs.append(mod_cb.collapsed('time', iris.analysis.MAX).data)
                # saving the weigth and number of models to calculate weights 
                mod_cb.attributes['ensemble_weight'] = 1 / n_real
                mod_cb.attributes['reverse_dtsts_n'] = 1/ len(datasets)
                # adding individual cube into the respective cubelists
                ens_cubelist.append(mod_cb)
                mod_cubelist.append(mod_cb)
            # saving the cubelist for the individual model 
            plotting_dic[group][dataset] = {'data': mod_cubelist}
            # calculating uncertanties for the GEV fit using bootstrap method
            plotting_dic[group][dataset]['uncert'] = bootstrap_gev(mod_cubelist, obs_info['ana_year_value'])
        # after all models in the group processed, saving multi-model ensemble
        plotting_dic[group]['Multi-Model-Mean'] = {'data' : ens_cubelist}
        plotting_dic[group]['Multi-Model-Mean']['uncert'] = bootstrap_gev(plotting_dic[group], obs_info['ana_year_value']) 
        # defining mean GEV parameters from bootstrap to use as initial 
        # conditions. Not used further, but good to have. 
        fit_param_apr[group] = {'loc': np.around(plotting_dic[group]['Multi-Model-Mean']['uncert']['loc'].mean(),3),
                                'scale': np.around(plotting_dic[group]['Multi-Model-Mean']['uncert']['scale'].mean(),3),
                                'shape': np.around(plotting_dic[group]['Multi-Model-Mean']['uncert']['shape'].mean(),3)}
    
    # addind observational dictionary to the main dictionary
    plotting_dic['obs_info'] = obs_info

    # calculating min and max over the whole processed data
    min_var = np.asarray(mins).min() ; max_var = np.asarray(maxs).max()  

    # borders which will be used for the plot and span of the GEV fit
    border = [np.floor(min_var*1.5), np.ceil(max_var*1.5)]

    # loading matplotlib style saved in ESMValTool folder
    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    # plotting every single bootstrap and calculating the uncertainty bands
    uncert_band = make_uncert_figures(plotting_dic, cfg, border)

    # making the main histogram/gev/return period/QQ plot and saving the data
    make_hist_figure(plotting_dic, cfg, uncert_band, border, fit_param_apr)

    # making the observational figure with return periods 
    make_era_dist_figure(obs_info, cfg, border)

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
