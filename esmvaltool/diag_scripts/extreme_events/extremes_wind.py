import csv
import esmvalcore.preprocessor as eprep
import iris
from iris.util import equalise_attributes
import cf_units
import cftime
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import genextreme as gev
from scipy.stats import gumbel_r as gumbel
from scipy.stats import kstest, cramervonmises

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, select_metadata, group_metadata, get_diagnostic_filename, save_data, ProvenanceLogger
import esmvaltool.diag_scripts.shared.plot as eplot
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import ProvenanceLogger

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def era_preproc_individual(era_ind_fname, aux_dir, cfg):

    era_orig_cb = iris.load_cube(era_ind_fname)

    era_seas_cb = eprep.extract_season(era_orig_cb, season=cfg['season'])

    if cfg['era_regridding_shape']:
        era_seas_cb.coord('latitude').guess_bounds()
        era_seas_cb.coord('longitude').guess_bounds()
        reg_cube = eprep.extract_shape(era_seas_cb, os.path.join(aux_dir,cfg['era_regridding_region']), method='contains', crop=True)
    else:
        reg_cube = eprep.extract_region(era_seas_cb, cfg['era_regridding_region'][0], cfg['era_regridding_region'][1], 
                        cfg['era_regridding_region'][2], cfg['era_regridding_region'][3])

    return reg_cube


def get_era_txx(cfg):

    analysis_year = cfg['analysis_year']
    aux_dir = cfg['auxiliary_data_dir']
    work_dir = cfg['work_dir']
    years = np.arange(cfg['era_year_span'][0], cfg['era_year_span'][1]+1)

    region_p = '_'
    for rp in  cfg['region'].split(' '):
        region_p += rp.lower()+'_'

    era_dir = os.path.join(work_dir, 'era5')
    if not os.path.exists(era_dir):
        os.makedirs(era_dir)

    era_ws_arr = np.zeros(len(years))
    for n, y in enumerate(years):
        u_fname = os.path.join(aux_dir, 'era5_10m_u_component_of_wind_'+str(y)+'_hourly_143W-45W_38N-85N.nc')
        v_fname = os.path.join(aux_dir, 'era5_10m_v_component_of_wind_'+str(y)+'_hourly_143W-45W_38N-85N.nc')
        
        u_cb = era_preproc_individual(u_fname, aux_dir, cfg)
        v_cb = era_preproc_individual(v_fname, aux_dir, cfg)

        ws_cb = (u_cb**2 + v_cb**2)**0.5
        ws_cb.var_name = '10m_wind'
        ws_cb.long_name = '10m wind speed'
        iris.save(ws_cb, os.path.join(era_dir, 'era5_10m_wind_speed'+region_p+cfg['season'].lower()+'_'+str(y)+'.nc'))

        era_daily_cb = eprep.daily_statistics(ws_cb, operator='mean')
        era_spatial_cb = eprep.area_statistics(era_daily_cb, 'max')
        era_max_cb = eprep.annual_statistics(era_spatial_cb, 'max')
        era_ws_arr[n] = era_max_cb.data[0]

    tims = [cftime.datetime(y, 7, 15, calendar='gregorian') for y in years]
    tim_dim = iris.coords.DimCoord(cftime.date2num(tims,'days since 1850-01-01', calendar='gregorian'), 
                                   standard_name='time', long_name='time', var_name='time',
                                    units=cf_units.Unit('days since 1850-01-01', calendar='gregorian'))
    
    era_big_cube = iris.cube.Cube(era_ws_arr, standard_name='wind_speed', long_name='10m wind speed', var_name='10m_ws', 
                                    units="m s-1", dim_coords_and_dims=[(tim_dim,0)])

    return era_big_cube

def bootstrap_gev(data_dic, distrib = 'gev'): 

    if type(data_dic) == dict:  
        # determining the max length of the model realisation, the number of bootstrap
        # iterations is this value * 100
        max_cblst_len = np.asarray([len(data_dic[model]['data']) for model in data_dic.keys()]).max()
        iter_pool = max_cblst_len *100
        n_real = len(data_dic.keys())
        n_years = np.asarray([data_dic[model]['data'][0].shape for model in data_dic.keys()]).max()
        pool_size = np.int(np.around(n_real*n_years))
    elif type(data_dic) == iris.cube.CubeList: 
        iter_pool = np.max([len(data_dic) *100, 1000])
        n_years = data_dic[0].shape[0]
        n_real = 3 # this is a number of realisations which shown to be enough to draw conclusions
        if len(data_dic)<n_real: 
            pool_size = len(data_dic) * n_years
        else: 
            pool_size = n_real * n_years

    pool_data = list()
    if type(data_dic) == dict: 
        for model in data_dic.keys():
            for i in range(len(data_dic[model]['data'])):
                pool_data.append(data_dic[model]['data'][i].data)
    elif type(data_dic) == iris.cube.CubeList:
        for cb in data_dic: 
            pool_data.append(cb.data)
    pool_data = np.asarray(pool_data)

    shapes = np.zeros(iter_pool)
    locs = np.zeros(iter_pool)
    scales = np.zeros(iter_pool)

    rng = np.random.default_rng(501)

    for i in range(0, iter_pool):
        sample_data = list()
        mod_idxs = rng.choice(pool_data.shape[0], size = pool_size)
        for mod_idx in mod_idxs: 
            y_idx = rng.choice(n_years)
            sample_data.append(pool_data[mod_idx, y_idx])
        sample_data = np.asarray(sample_data).flatten()
        if distrib.lower()=='gev':
            shapes[i], locs[i], scales[i] = gev.fit(sample_data, method='MLE')
        elif distrib.lower()=='gumbel':
            locs[i], scales[i] = gumbel.fit(sample_data, method='MLE')
    
    param_dic = {'loc': locs, 'scale': scales}    
    if distrib.lower()=='gev': 
        param_dic['shape'] = shapes

    return param_dic


def make_uncert_figures(data_dic, cfg, border, distrib='gev'):

    colors = {}
    colors['wind_now'] = (196 / 255, 121 / 255, 0)
    colors['wind_nat'] = (0, 79 / 255, 0)
    colors['wind_fut'] = (69 / 255, 118 / 255, 191 / 255)

    exp_list = list(data_dic.keys())  ; exp_list.remove('reanalysis') 
    models = list(data_dic[exp_list[0]].keys())

    x_gev = np.arange(border[0],border[1], 0.1)

    tlocs = {'wind_now': 0.04 , 'wind_nat': 0.01,  'wind_fut': 0.08}
    uncert_band = {}
    for exp in exp_list: 
        uncert_band[exp] = {}

    if distrib.lower() == 'gev': 
        gev_params = ['shape', 'loc', 'scale']
    elif distrib.lower() == 'gumbel':
        gev_params = ['loc', 'scale']

    for model in models: 

        # this is a figure where we will plot single distributions from bootstrap
        fig_single_bootstrap, ax_single_bootstrap = plt.subplots(1)
        fig_single_bootstrap.set_size_inches(12., 8.)

        # this a figure where we plot how the values for GEV params are distributed 
        fig_gev_distr, ax_gev_distr = plt.subplots(len(gev_params))
        fig_gev_distr.set_size_inches(8., 12.)

        for exp in exp_list: 
            gev_dic = data_dic[exp][model].pop('uncert')
            data_dic[exp][model] = data_dic[exp][model].pop('data')
            # here we plot distribution of single GEV params
            for n, gev_param in enumerate(gev_params):
                ax_gev_distr[n].hist(gev_dic[gev_param], bins=50, edgecolor='none',
                        facecolor = colors[exp], alpha=0.3, label = exp, density=True)
                ax_gev_distr[n].set_xlabel(gev_param)
                ax_gev_distr[n].set_ylabel('Number density')
                ax_gev_distr[n].set_title(distrib +' parameter: ' + gev_param)

            # this is an array where we'll throw all pdfs, to calculate 5/95 perc later 
            all_pdfs = np.zeros((len(x_gev), len(gev_dic['loc'])))
            all_sfs = np.zeros((len(x_gev), len(gev_dic['loc'])))

            # here we plot single distributions
            for i in range(len(gev_dic['loc'])):
                if distrib.lower() == 'gev':
                    gev_pdf = gev.pdf(x_gev, gev_dic['shape'][i],gev_dic['loc'][i], gev_dic['scale'][i])
                    gev_sf = gev.sf(x_gev, gev_dic['shape'][i],gev_dic['loc'][i], gev_dic['scale'][i])
                elif distrib.lower() == 'gumbel': 
                    gev_pdf = gumbel.pdf(x_gev, gev_dic['loc'][i], gev_dic['scale'][i])
                    gev_sf = gumbel.sf(x_gev, gev_dic['loc'][i], gev_dic['scale'][i])
                all_pdfs[:, i] = gev_pdf
                all_sfs[:, i] = gev_sf
                ax_single_bootstrap.plot(x_gev, gev_pdf, color = colors[exp], alpha=0.03)
            
            uncert_band[exp][model] = {'x_gev': x_gev, 'pdf_5th_perc' : np.percentile(all_pdfs, 5, axis = 1), 
                                                'pdf_95th_perc': np.percentile(all_pdfs, 95, axis = 1),
                                                'sf_5th_perc' : np.percentile(all_sfs, 5, axis = 1), 
                                                'sf_95th_perc' : np.percentile(all_sfs, 95, axis = 1),
                                                'sf_data_full' : all_sfs,
                                                'loc_min': gev_dic['loc'].min(), 'loc_max': gev_dic['loc'].max(),
                                                'loc_5': np.percentile(gev_dic['loc'], 5, interpolation='nearest'),
                                                'loc_95': np.percentile(gev_dic['loc'], 95, interpolation='nearest'),
                                                'scale_min': gev_dic['scale'].min(), 'scale_max': gev_dic['scale'].max(),
                                                'scale_5': np.percentile(gev_dic['scale'], 5, interpolation='nearest'),
                                                'scale_95': np.percentile(gev_dic['scale'], 95, interpolation='nearest')}
            if distrib.lower() == 'gev':
                uncert_band[exp][model].update({'shape_min': gev_dic['shape'].min(), 'shape_max': gev_dic['shape'].max(),
                                                'shape_5': np.percentile(gev_dic['shape'], 5, interpolation='nearest'),
                                                'shape_95': np.percentile(gev_dic['shape'], 95, interpolation='nearest')})
    
            param_str = '                             '+exp
            if distrib.lower() == 'gev':
                param_str += '\nshape mean:'+str(np.around(gev_dic['shape'].mean(),3))+', max:'+str(np.around(gev_dic['shape'].max(),3)) + ', min:' + str(np.around(gev_dic['shape'].min(),3)) 
            param_str += '\n loc mean:'+str(np.around(gev_dic['loc'].mean(),3))+', max:'+str(np.around(gev_dic['loc'].max(),3)) + ', min:' + str(np.around(gev_dic['loc'].min(),3)) \
                + '\nscale mean:'+ str(np.around(gev_dic['scale'].mean(),3))+', max:'+str(np.around(gev_dic['scale'].max(),3)) + ', min:' + str(np.around(gev_dic['scale'].min(),3))               
            
            ax_single_bootstrap.text(0.95*border[0], tlocs[exp], param_str, color = colors[exp])

        ax_gev_distr[0].legend(loc=0, fancybox=False, frameon=False)
        fig_gev_distr.suptitle('Distribution of ' + distrib+' parameters after bootstrap in ' +model,
                    fontsize = 'x-large')
        fig_gev_distr.set_dpi(250)
        plt.tight_layout()
        fig_gev_distr.savefig(os.path.join(cfg['plot_dir'], 'figure_'+cfg['region'].lower()+'_exreme_distr_param_'+distrib.lower() +'_'+ model + diagtools.get_image_format(cfg)))
        plt.close(fig_gev_distr)

        ax_single_bootstrap.legend(loc=0, fancybox=False, frameon=False)
        ax_single_bootstrap.set_xlim(border[0], border[1])
        ax_single_bootstrap.set_ylim(0, ax_single_bootstrap.get_ylim()[1])
        ax_single_bootstrap.set_xlabel(cfg['ax_var_label']+', '+cfg['var_units'])
        ax_single_bootstrap.set_ylabel('Number density')

        fig_single_bootstrap.suptitle('Estimation of ' + distrib +' fit uncertainty from '+model+' with Bootstrap method',
                    fontsize = 'x-large')
        fig_single_bootstrap.set_dpi(250)

        fig_single_bootstrap.savefig(os.path.join(cfg['plot_dir'], 'figure_'+cfg['region'].lower()+'_extremes_bootstrap_'+distrib.lower() +'_'+ model + diagtools.get_image_format(cfg)))
        plt.close(fig_single_bootstrap)
                            
    return uncert_band


def get_ref_params(ref_data_cb, cfg, distrib = 'gev'):

    era_2021 = ref_data_cb.data[-1]
    if distrib.lower() == 'gev':
        era_gev_params = gev.fit(ref_data_cb.data)
    elif distrib.lower() == 'gumbel':
        era_gev_params = gumbel.fit(ref_data_cb.data)
    
    era_ks = kstest(ref_data_cb.data, gev(*era_gev_params).cdf)
    era_cvm = cramervonmises(ref_data_cb.data, gev(*era_gev_params).cdf)
    
    orig_rp = np.around(1/gev.sf(era_2021, *era_gev_params), 1)
    bootstrap_rps = list()
    rng = np.random.default_rng(501)

    for i in range(1000): 
        for_fit = rng.choice(ref_data_cb.data[:-1], size=len(ref_data_cb.data) - 1, replace=True)
        for_fit = np.append(for_fit, era_2021)
        temp_gev = gev.fit(for_fit)
        temp_rp = np.around(1/gev.sf(era_2021, *temp_gev), 1)
        bootstrap_rps.append(temp_rp)
    
    bootstrap_rps = np.asarray(bootstrap_rps)

    rp_perc = np.percentile(bootstrap_rps, [5,10,50,90,95]).round(1)

    era_csv = open(os.path.join(cfg['work_dir'], distrib.lower()+'_era_data.csv'), 'w', newline='')
    era_csv_writer = csv.writer(era_csv, delimiter=',')
    era_csv_writer.writerow([str(cfg['analysis_year'])+' ERA value '+str(era_2021)])
    era_csv_writer.writerow(['ERA ' +distrib +' params'])
    if distrib.lower() == 'gev':
        era_csv_writer.writerow(['shape', 'loc', 'scale'])
    elif distrib.lower() == 'gumbel':
        era_csv_writer.writerow(['loc', 'scale'])
    era_csv_writer.writerow(era_gev_params)
    era_csv_writer.writerow(['KS test params: statistic', 'pvalue'])
    era_csv_writer.writerow([era_ks.statistic, era_ks.pvalue])
    era_csv_writer.writerow(['CvM test params: statistic', 'pvalue'])
    era_csv_writer.writerow([era_cvm.statistic, era_cvm.pvalue])
    era_csv_writer.writerow([str(cfg['analysis_year'])+' ERA return period', str(np.around(orig_rp,1))])
    era_csv_writer.writerow(['Bootstrapped uncertanties on ERA return period'])
    era_csv_writer.writerow(['5_perc', '10_perc', '50_perc', '90_perc', '95_perc'])
    era_csv_writer.writerow(rp_perc)
    era_csv.close()

    return era_gev_params


def make_hist_figure(data_dic, cfg, uncert_band, border, apr_param, distrib = 'gev'):

    era_cb = data_dic['reanalysis']
    event = era_cb.data[-1]

    era_params = get_ref_params(era_cb, cfg, distrib)

    risk_csv = open(os.path.join(cfg['work_dir'], distrib.lower()+'_risk_data.csv'), 'w', newline='')
    risk_csv_writer = csv.writer(risk_csv, delimiter=',')
    risk_head_row = ['model']

    risk_uncert_csv = open(os.path.join(cfg['work_dir'], distrib.lower()+'_uncert_risk_data.csv'), 'w', newline='')
    risk_uncert_csv_writer = csv.writer(risk_uncert_csv, delimiter=',')
    risk_uncert_head_row = ['model','all/nat_r_r_5', 'all/nat_r_r_10', 'all/nat_r_r_50','all/nat_r_r_90', 'all/nat_r_r_95',
                            'ssp/nat_r_r_5', 'ssp/nat_r_r_10', 'ssp/nat_r_r_50', 'ssp/nat_r_r_90', 'ssp/nat_r_r_95',
                            'ssp/all_r_r_5', 'ssp/all_r_r_10', 'ssp/all_r_r_50', 'ssp/all_r_r_90', 'ssp/all_r_r_95']
    risk_uncert_csv_writer.writerow(risk_uncert_head_row)

    exp_list = list(data_dic.keys()) ; exp_list.remove('reanalysis')    

    quantile_measures = np.arange(0, 1.01, 0.01); quantile_measures[0] = 0.001

    colors = {'wind_now' : (196 / 255, 121 / 255, 0), 
              'wind_nat' : (0, 79 / 255, 0), 
              'wind_fut' : (69 / 255, 118 / 255, 191 / 255)}

    csv_file = open(os.path.join(cfg['work_dir'], distrib.lower()+'_parameters.csv'), 'w', newline='')
    gevs_csv_writer = csv.writer(csv_file, delimiter=',')
    head_row = ['model']
    for exp_key in exp_list:
        if distrib.lower() == 'gev':
            head_row.extend(['shape_'+exp_key, 'shape_min_'+exp_key, 'shape_5_'+exp_key, 'shape_95_'+exp_key,'shape_max_'+exp_key])
        head_row.extend(['loc_'+exp_key,'loc_min_'+exp_key,'loc_5_'+exp_key,'loc_95_'+exp_key,'loc_max_'+exp_key])
        head_row.extend(['scale_'+exp_key,'scale_min_'+exp_key,'scale_5_'+exp_key,'scale_95_'+exp_key,'scale_max_'+exp_key])
        head_row.extend(['ks_stat_'+exp_key, 'ks_pvalue_'+exp_key, 'cvm_stat_'+exp_key, 'cvm_pvalue_'+exp_key])
        risk_head_row.append(exp_key+'_prob')
        risk_head_row.extend([exp_key+'_return_p', exp_key+'_return_p_5', exp_key+'_return_p_95', exp_key+'_return_p_5',exp_key+'_return_p_95'])
        risk_head_row.extend([exp_key+'_intens', exp_key+'_intens_5',exp_key+'_intens_95'])
    gevs_csv_writer.writerow(head_row)
    risk_csv_writer.writerow(risk_head_row)
    models = data_dic[exp_list[0]].keys()
    for model in models: 
        fig = plt.figure(constrained_layout=False)
        fig.set_size_inches(12., 8.)
        gs = fig.add_gridspec(nrows=2, ncols=6)
        ax_hist = fig.add_subplot(gs[:, :-2])
        ax_qq= fig.add_subplot(gs[0, -2:])
        ax_surv= fig.add_subplot(gs[1, -2:])
        model_row = [model]
        risk_model_row = [model]
        for exp_key in exp_list:
            ens_cubelist = data_dic[exp_key][model]
            distrib_data = []
            weights = []
            for cube in ens_cubelist:
                if (model == 'Multi-Model-Mean')&(cfg['model_weighting']):
                    cube_weight = cube.attributes['ensemble_weight']*cube.attributes['reverse_dtsts_n']
                else:
                    cube_weight = 1
                for point in cube.data:
                    distrib_data.append(np.around(point, 1))
                    weights.append(cube_weight/len(cube.data))
            distrib_data = np.asarray(distrib_data)
            weights = np.asarray(weights)
            if cfg['model_weighting']:
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
            x_gev = uncert_band[exp_key][model]['x_gev']
            if distrib.lower() == 'gev':
                w_distr_par = gev.fit(distrib_data, loc=apr_param[exp_key]['loc'], scale=apr_param[exp_key]['scale'], method='MLE')
                w_pdf = gev.pdf(x_gev, *w_distr_par)
                w_survival = gev.sf(x_gev, *w_distr_par)
                theor_quants = gev(*w_distr_par).ppf(quantile_measures)
                era_surv = gev.sf(x_gev, *era_params)
                ks_res = kstest(distrib_data, gev(*w_distr_par).cdf)
                cvm_res = cramervonmises(distrib_data, gev(*w_distr_par).cdf)
            elif distrib.lower() == 'gumbel':
                w_distr_par = gumbel.fit(distrib_data, loc=apr_param[exp_key]['loc'], method='MLE')
                w_pdf = gumbel.pdf(x_gev, *w_distr_par)
                w_survival = gumbel.sf(x_gev, *w_distr_par)
                theor_quants = gumbel(*w_distr_par).ppf(quantile_measures)
                era_surv = gumbel.sf(x_gev, *era_params)
                ks_res = kstest(distrib_data, gumbel(*w_distr_par).cdf)
                cvm_res = cramervonmises(distrib_data, gumbel(*w_distr_par).cdf)
            event_idx = np.argmin(np.abs(x_gev - event))
            era_event_prob = era_surv[event_idx]
            if distrib.lower() == 'gev':
                model_row.extend([w_distr_par[-3], uncert_band[exp_key][model]['shape_min'], uncert_band[exp_key][model]['shape_5']])
                model_row.extend([uncert_band[exp_key][model]['shape_95'], uncert_band[exp_key][model]['shape_max']])
            model_row.extend([w_distr_par[-2], uncert_band[exp_key][model]['loc_min'], uncert_band[exp_key][model]['loc_5']])
            model_row.extend([uncert_band[exp_key][model]['loc_95'], uncert_band[exp_key][model]['loc_max']])            
            model_row.extend([w_distr_par[-1], uncert_band[exp_key][model]['scale_min'], uncert_band[exp_key][model]['scale_5']])
            model_row.extend([uncert_band[exp_key][model]['scale_95'], uncert_band[exp_key][model]['scale_max']])            
            model_row.extend([ks_res.statistic, ks_res.pvalue, cvm_res.statistic, cvm_res.pvalue])
            n_bins = np.arange((border[0]//2)*2, border[1]+2, 2)
            ax_hist.hist(distrib_data, bins=n_bins, edgecolor=colors[exp_key],
                    facecolor = colors[exp_key], alpha=0.3, label=cfg['name_' + exp_key] , density=True, weights=weights, zorder = 2)
            ax_hist.plot(x_gev, w_pdf, c = colors[exp_key], ls = 'solid', label = distrib+' fit '+cfg['name_' + exp_key], zorder=3)
            pdf_perc_5 = uncert_band[exp_key][model]['pdf_5th_perc']
            pdf_perc_95 = uncert_band[exp_key][model]['pdf_95th_perc']
            sf_perc_5 = uncert_band[exp_key][model]['sf_5th_perc']
            sf_perc_95 = uncert_band[exp_key][model]['sf_95th_perc']
            ax_hist.fill_between(x_gev, pdf_perc_5, pdf_perc_95, color=colors[exp_key], alpha=0.3, linewidth=0, zorder=4)
            ax_surv.fill_between(x_gev, 1/sf_perc_5, 1/sf_perc_95, color=colors[exp_key], alpha=0.3, linewidth=0, zorder=4)
            intens = x_gev[np.argmin(np.abs(w_survival-era_event_prob))]
            intens_95 = x_gev[np.argmin(np.abs(sf_perc_95-era_event_prob))]
            intens_5 = x_gev[np.argmin(np.abs(sf_perc_5-era_event_prob))]
            risk_model_row.extend([w_survival[event_idx], 1/w_survival[event_idx], 1/sf_perc_5[event_idx], 1/sf_perc_95[event_idx], intens, intens_5, intens_95])
            pract_quants = np.quantile(distrib_data, quantile_measures)
            ax_qq.scatter(theor_quants, pract_quants, edgecolors=colors[exp_key], marker='o', facecolors='None', lw=0.75, label=cfg['name_' + exp_key], zorder=3)
            ax_surv.plot(x_gev, 1/w_survival, color=colors[exp_key], zorder=3)
        gevs_csv_writer.writerow(model_row)
        risk_csv_writer.writerow(risk_model_row)

        ylims = ax_hist.get_ylim()
        ax_hist.set_ylim(*ylims)

        ax_hist.text(border[0]/1.15, ylims[1]*0.65,'  Number of\nrealisations ' +str(len(ens_cubelist)), fontsize='large')
        ax_hist.vlines(event, *ylims, color = 'indianred', linestyle = 'solid', lw=1.5, zorder=1, label = 'ERA5 ('+str(cfg['analysis_year'])+')')

        ax_surv.vlines(event, 0.1, 1/era_event_prob, linestyle = 'solid', color='indianred', zorder=2,  label = 'ERA5 ('+str(cfg['analysis_year'])+')')
        ax_surv.hlines(1/era_event_prob, x_gev[0], event,linestyle = 'solid', color='indianred', zorder=2)

        ax_hist.legend(loc=0, fancybox=False, frameon=False)
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
        ax_hist.set_title('Probability density function')
        ax_qq.set_title('Quality assessment')
        ax_qq.set_ylabel('Data quantile')
        ax_qq.set_xlabel(distrib+' quantile')
        ax_surv.set_title('Return period')
        ax_surv.set_ylabel('years')
        ax_surv.set_xlabel(cfg['ax_var_label'] + ' anomaly, ' +cfg['var_units'])
        ax_hist.set_xlabel(cfg['ax_var_label'] + ' anomaly, ' +cfg['var_units'])
        ax_hist.set_ylabel('Number density')

        if model == 'Multi-Model-Mean':
            fig.suptitle(cfg['title_var_label']+' in '+cfg['region'] \
            + ' calculated from '+str(len(models) - 1) +' CMIP6 models' , fontsize = 'x-large')
        else:
            fig.suptitle(cfg['title_var_label']+' in '+cfg['region'] \
            +  ' calculated from '+model, fontsize = 'x-large')
        fig.set_dpi(250)

        plt.tight_layout()
        
        fig.savefig(os.path.join(cfg['plot_dir'], 'figure_'+cfg['region'].lower()+'_extremes_'+distrib.lower() +'_'+model + diagtools.get_image_format(cfg)))

        all_to_nat = list(); ssp_to_nat = list(); ssp_to_all = list()
        for i in range(uncert_band['wind_now'][model]['sf_data_full'].shape[1]):
            for j in range(uncert_band['wind_now'][model]['sf_data_full'].shape[1]):
                all_to_nat.append(uncert_band['wind_now'][model]['sf_data_full'][event_idx,i]/uncert_band['wind_nat'][model]['sf_data_full'][event_idx,j])
                ssp_to_nat.append(uncert_band['wind_fut'][model]['sf_data_full'][event_idx,i]/uncert_band['wind_nat'][model]['sf_data_full'][event_idx,j])
                ssp_to_all.append(uncert_band['wind_fut'][model]['sf_data_full'][event_idx,j]/uncert_band['wind_now'][model]['sf_data_full'][event_idx,i])
        all_to_nat_perc = np.percentile(all_to_nat, [5,10,50,90,95])
        ssp_to_nat_perc = np.percentile(ssp_to_nat, [5,10,50,90,95])
        ssp_to_all_perc = np.percentile(ssp_to_all, [5,10,50,90,95])
        risk_uncert_row = np.concatenate((all_to_nat_perc, ssp_to_nat_perc, ssp_to_all_perc))
        risk_uncert_csv_writer.writerow(risk_uncert_row)
    
    risk_uncert_csv.close()

    return


def make_era_dist_figure(abs_cube, cfg, border, distrib = 'gev'):

    abs_cube = abs_cube 
    border[0] = np.floor(abs_cube.data.min()*0.8)
    border[1] = np.ceil(abs_cube.data.max()*1.1)
    x_gev_fine = np.arange(border[0], border[1]+0.1, 0.1)
    n_bins = np.arange((border[0]//5)*2, border[1]+2,2)

    if distrib.lower() == 'gev':
        era_param = gev.fit(abs_cube.data)
        era_pdf= gev.pdf(x_gev_fine, *era_param)
        era_sf = gev.sf(x_gev_fine, *era_param)
    elif distrib.lower() == 'gumbel': 
        era_param = gumbel.fit(abs_cube.data)
        era_pdf= gumbel.pdf(x_gev_fine, *era_param)
        era_sf = gumbel.sf(x_gev_fine, *era_param)

    tims = cf_units.num2pydate(abs_cube.coord('time').points, abs_cube.coord('time').units.origin, abs_cube.coord('time').units.calendar)
    years = np.asarray([t.year for t in tims])

    fig_era = plt.figure(constrained_layout=False)
    fig_era.set_size_inches(12., 8.)
    gs = fig_era.add_gridspec(nrows=2, ncols=5)
    ax_era_hist = fig_era.add_subplot(gs[0, :-2])
    ax_era_surv= fig_era.add_subplot(gs[0, -2:])
    ax_era_tseries= fig_era.add_subplot(gs[1, :])

    ax_era_hist.hist(abs_cube.data, bins=n_bins, edgecolor='indianred', facecolor='indianred', alpha=0.3, density=True)
    ax_era_hist.plot(x_gev_fine, era_pdf, c='indianred')
    ylims = ax_era_hist.get_ylim()
    ax_era_hist.text(border[1]*0.9,ylims[1]*0.3,distrib + ' params:\n' + 'shape: '+ str(np.around(era_param[0],2))+'\n  loc: '+\
                                 str(np.around(era_param[1],2))+ '\nscale: '+str(np.around(era_param[2],2)), color='k')
    ax_era_hist.scatter(abs_cube.data[-1], 0, c='indianred', marker='x',lw=2, s=100, label=str(cfg['analysis_year']), clip_on=False, zorder=4)
    ax_era_hist.set_xlim(*border)
    ax_era_hist.set_xlabel(cfg['ax_var_label'] +', ' + cfg['var_units'])
    ax_era_hist.set_ylabel('Number density')
    ax_era_hist.set_title('Probability distribution function')

    ax_era_surv.plot(x_gev_fine[era_sf>0.00001], 1/era_sf[era_sf>0.00001], c='indianred')
    ax_era_surv.set_title('ERA5 ' +  cfg['ax_var_label'] +' return period')
    ax_era_surv.set_xlabel(cfg['ax_var_label'] +', ' + cfg['var_units'])
    ax_era_surv.set_ylabel('years')
    era_idx = np.argmin(np.abs(x_gev_fine-abs_cube.data[-1]))
    ax_era_surv.scatter(abs_cube.data[-1],1/era_sf[era_idx],c='indianred', marker='x',lw=2, s=100, label=str(cfg['analysis_year']), clip_on=False, zorder=4)
    ax_era_surv.text(border[1]*0.8, 5,'ERA5 return period\n    '+str(np.around(1/era_sf[era_idx], 1))+ ' years', color='k')
    ax_era_surv.set_yscale('log')
    ax_era_surv.grid(color='silver', axis='both', alpha=0.5)
    ax_era_surv.set_ylim(1,10000)
    ax_era_surv.set_xlim(*border)

    ax_era_tseries.set_title('ERA5 ' + cfg['ax_var_label'] +' timeseries')
    ax_era_tseries.plot(years, abs_cube.data, c='indianred')
    ax_era_tseries.grid(color='silver', axis='both', alpha=0.5)
    ax_era_tseries.set_xlabel('time')
    ax_era_tseries.set_ylabel(cfg['ax_var_label']+', ' + cfg['var_units'])
    ax_era_tseries.scatter(years[np.argmax(abs_cube.data)], abs_cube.data.max(), edgecolors='indianred', marker='o', facecolors='None', s=100, lw=2)
    ax_era_tseries.set_xlim(years[0]-0.5, years[-1]+0.5)
    ax_era_tseries.arrow(years[-1] - len(years)*0.1, abs_cube.data.mean()*0.2 + 0.8*abs_cube.data.max(), len(years)*0.09 ,
                                     0.19*abs_cube.data.max()-abs_cube.data.mean()*0.2, color='k', length_includes_head=True,  
                                                                                        head_width=0.5, head_length=0.5)
    ax_era_tseries.text(years[-1] - len(years)*0.16, abs_cube.data.mean()*0.2 + 0.77*abs_cube.data.max(), 
                        'ERA5 '+str(cfg['analysis_year'])+': '+str(np.around(abs_cube.data.max(),1))+' '+cfg['var_units'])

    fig_era.suptitle('ERA5 ' + cfg['title_var_label'] + ' in ' + cfg['region'] + ' and its ' + distrib + ' fit', fontsize = 'x-large')

    plt.tight_layout()

    fig_era.savefig(os.path.join(cfg['plot_dir'], 'figure_'+cfg['region'].lower()+'_era' + diagtools.get_image_format(cfg)))


    return


def main(cfg):

    input_data = cfg['input_data']

    groups = group_metadata(input_data.values(), 'variable_group', sort=True)

    wind_groups_l=[]
    for k in groups.keys():
        if 'wind' in k:
            wind_groups_l.append(k)
    
    abs_era_cube= get_era_txx(cfg)

    mins = list(); maxs = list()

    distrib_fit= cfg['fit_distribution']

    plotting_dic = {}

    fit_param_apr = {}

    for group in wind_groups_l:
        plotting_dic[group] = {}
        group_data = groups[group]
        datasets = group_metadata(group_data, 'dataset')
        ens_cubelist = iris.cube.CubeList()
        for dataset in datasets.keys():
            filepaths = list(group_metadata(datasets[dataset], 'filename').keys())
            n_real = len(filepaths)
            mod_cubelist = iris.cube.CubeList()
            for filepath in filepaths:
                mod_cb = iris.load_cube(filepath)
                mins.append(mod_cb.collapsed('time', iris.analysis.MIN).data)
                maxs.append(mod_cb.collapsed('time', iris.analysis.MAX).data)
                mod_cb.attributes['ensemble_weight'] = 1 / n_real
                mod_cb.attributes['reverse_dtsts_n'] = 1/ len(datasets)
                ens_cubelist.append(mod_cb)
                mod_cubelist.append(mod_cb)
            plotting_dic[group][dataset] = {'data': mod_cubelist}
            plotting_dic[group][dataset]['uncert'] = bootstrap_gev(mod_cubelist, distrib=distrib_fit)
        plotting_dic[group]['Multi-Model-Mean'] = {'data' : ens_cubelist}
        plotting_dic[group]['Multi-Model-Mean']['uncert'] = bootstrap_gev(plotting_dic[group], distrib=distrib_fit) 
        fit_param_apr[group] = {'loc': np.around(plotting_dic[group]['Multi-Model-Mean']['uncert']['loc'].mean(),3),
                                'scale': np.around(plotting_dic[group]['Multi-Model-Mean']['uncert']['scale'].mean(),3)}
        if distrib_fit.lower() == 'gev':
            fit_param_apr[group]['shape'] = np.around(plotting_dic[group]['Multi-Model-Mean']['uncert']['shape'].mean(),3)
    
    plotting_dic['reanalysis'] = abs_era_cube

    min_var = np.asarray(mins).min() ; max_var = np.asarray(maxs).max()  

    border = [np.floor(min_var/1.5), np.ceil(max_var*1.5)]

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    uncert_band = make_uncert_figures(plotting_dic, cfg, border, distrib = distrib_fit)

    make_hist_figure(plotting_dic, cfg, uncert_band, border, fit_param_apr, distrib = distrib_fit)

    make_era_dist_figure(abs_era_cube, cfg, border, distrib = distrib_fit)

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
