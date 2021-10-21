import cf_units
import cftime
import datetime
import pickle

import esmvalcore.preprocessor
import iris
from iris.util import equalise_attributes
import iris.plot as iplt
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import genextreme as gev

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, get_diagnostic_filename, save_data, Datasets, Variables, ProvenanceLogger
from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
from esmvalcore.preprocessor import regrid
import esmvaltool.diag_scripts.shared.plot as eplot
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import ProvenanceLogger

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def rename_variables(vars, datasets):

    # this function renames the variables in the model to run the script more effectively
    # tasmax is txx and pr is rx1day

    for var in vars.short_names():
        if var == 'tasmax':
            etccdi = 'txx'
        elif var =='pr':
            etccdi = 'rx1day'
        else:
            continue
        for dtst in datasets.get_dataset_info_list(short_name=var):
            dtst_info = datasets.get_dataset_info(dtst['filename'])
            updated_info = dtst_info
            updated_info ['short_name'] = vars.short_name(etccdi)
            updated_info ['long_name'] = vars.long_name(etccdi)
            updated_info['standard_name'] =vars.long_name(etccdi)
            datasets.set_data(data={}, path=dtst['filename'], dataset_info=updated_info)

    return

def sort_experiment_info(datasets, variable,project, exp_key):
    # as variable we mean short_name

    exps = set(datasets.get_info_list('exp', short_name = variable, project = project))

    for exp in exps:
        if exp_key.lower() == 'nat':
            if 'nat' in exp.lower():
                exp_name = exp
        elif exp_key.lower() == 'all':
            if (exp == 'historical') | ('historical-' in exp):
                exp_name = exp

    return (exp_name)

def get_era_txx(cfg):

    aux_dir = cfg['auxiliary_data_dir']
    cont_aux = glob.glob(aux_dir+'/era5_maximum_2m_temperature_since_previous_post_processing_*.nc')
    list_abs_cb = [ os.path.join(aux_dir, f) for f in cont_aux]

    era_cubelist = iris.load(list_abs_cb)
    equalise_attributes(era_cubelist)

    for n in range(len(era_cubelist)): 
        era_cubelist[n].data = era_cubelist[n].data.astype('float32')
        if era_cubelist[n].coord('time').units != era_cubelist[0].coord('time').units: 
            era_cubelist[n].coord('time').units = era_cubelist[0].coord('time').units

    era_cb = era_cubelist.concatenate_cube()
    txxs = esmvalcore.preprocessor.annual_statistics(era_cb, 'max')
     
    obs_cb = iris.load_cube('/home/acrnemr/esmvaltool_output/recipe_bc_extremes_rectang_20210924_050436/preproc/xcbox32/txx/OBS_HadEX3_clim_v20200928_yr_txx_1951-2017.nc')

    regrd_txx = esmvalcore.preprocessor.regrid(txxs, obs_cb, 'linear')

    reg_cb_txx = esmvalcore.preprocessor.extract_region(regrd_txx, -123.0, -119.0, 45.0, 52.0)

    anomal_era_cb = esmvalcore.preprocessor.anomalies(reg_cb_txx, 'full',
                        reference={'start_year': cfg['reference_period'][0],
                        'start_month': 1, 'start_day':1,
                        'end_year': cfg['reference_period'][1],
                        'end_month': 12, 'end_day':31} )
    crop_era_cb = esmvalcore.preprocessor.extract_time(anomal_era_cb,
                        start_year=cfg['era_plotting_period'][0],
                        start_month=1, start_day=1,
                        end_year=cfg['era_plotting_period'][1],
                        end_month=12, end_day=31)

    era_cb_txx = esmvalcore.preprocessor.area_statistics(crop_era_cb, 'mean')

    return era_cb_txx

def obtain_datasets(all_datasets, project, short_name, exp_key):

    if exp_key == 'OBS':
        exp = exp_key
        exp_dtsts = all_datasets.get_dataset_info_list(project=project, short_name=short_name)
        dataset_names = list(set([dtst['dataset'] for dtst in exp_dtsts]))
    else:
        exp = sort_experiment_info(all_datasets, short_name, project, exp_key)
        exp_dtsts = all_datasets.get_dataset_info_list(project=project, short_name=short_name, exp=exp)
        dataset_names = list(set([dtst['dataset'] for dtst in exp_dtsts]))

    return (dataset_names, exp)

def create_obs_mask(obs_filename, var):

    obs_data = iris.load_cube(obs_filename)

    n_lats = len(obs_data.coord('latitude').points)
    n_lons = len(obs_data.coord('longitude').points)

    obs_mask = np.ones((n_lats, n_lons), dtype = bool)

    n_years = len(obs_data.coord('time').points)
    t_param = round((n_years*0.7)+0.5)

    last_years_n= n_years - 5
    year_param=3

    for lat in range(n_lats):
        for lon in range(n_lons):
            if ((np.ma.count(obs_data.data[:, lat, lon])>=t_param)&(np.ma.count(obs_data.data[last_years_n:, lat, lon])
                                                                    >= year_param)):
                obs_mask[lat, lon] = False

    return (obs_mask)

def reform_exp_keys(project, exp_keys):
    #      this function is made in order to pass the correct keys into the next loop
    if project =='OBS':
        exp_keys = ['OBS']
    else:
        if 'OBS' in exp_keys:
            exp_keys.remove('OBS')

    return(exp_keys)

def obtain_filepaths(all_datasets, dataset, project, short_name, exp):

    if exp == 'OBS':
        filepaths = all_datasets.get_dataset_info_list(dataset=dataset, project=project, short_name=short_name)
    else:
        filepaths = all_datasets.get_dataset_info_list(dataset=dataset, project=project, short_name=short_name, exp=exp)

    return(filepaths)

def dataset_regriding(cube, exp, obs_filename):

    if exp =='OBS':
        regridded_cube = cube
    else:
        obs_cube = iris.load_cube(obs_filename)
        regridded_cube = regrid(cube, obs_cube, 'linear')

    return (regridded_cube)

def apply_obs_mask(cube, mask):

    if cube.shape[1:] == mask.shape:
        cube.data.mask = cube.data.mask | mask
    else:
        print('The regridding did not work correctly')

    return(cube)

def create_coords(cubelist, year_n):
    # dirty trick, we try to unify time to merge  the cubelist in the end

    cb = cubelist [0]

    n_t = len(cb.coord('time').points)

    tim = cb.coord('time')
    orig = tim.units.origin
    calendar = tim.units.calendar
    #  dest orig and calendar are converted, so in the end we can merge cubes
    dest_orig = 'days since 1850-1-1 00:00:00'
    dest_calendar = 'gregorian'
    cf_tim = cf_units.num2date(tim.points, orig, calendar)
    year = np.asarray([pnt.year for pnt in cf_tim])
    pnt_dts = np.asarray([datetime.datetime(yr, 7, 1) for yr in year])
    bds_dts = np.asarray([[datetime.datetime(yr, 1, 1), datetime.datetime(yr, 12, 31)] for yr in year])
    tim_coor = iris.coords.DimCoord(cf_units.date2num(pnt_dts, dest_orig, dest_calendar), standard_name='time',
                                    long_name='time', var_name='time', units=cf_units.Unit(dest_orig, dest_calendar),
                                    bounds=cf_units.date2num(bds_dts, dest_orig, dest_calendar))

    coord = [np.average(tim_coor.points[year_n*i:year_n*i + year_n]) for i in range(0, int(n_t / year_n))]
    bnds = [[tim_coor.bounds[year_n*i][0], tim_coor.bounds[year_n*i + (year_n - 1)][1]] for i in
            range(0, int(n_t / year_n))]
    if n_t%year_n != 0:
        # raise warning
        print('The n of years is not divisible by '+str(year_n))
        # coord.append(np.average(cb.coord('time').points[int(n_t / year_n):-1]))
        # bnds.append([cb.coord('time').bounds[int(n_t / year_n) * year_n][0], cb.coord('time').bounds[-1][1]])

    dcoord = iris.coords.DimCoord(np.asarray(coord), bounds=np.asarray(bnds), standard_name='time',
                                  units=cf_units.Unit(dest_orig, dest_calendar), long_name='time', var_name='time')

    return (dcoord)

def n_year_mean(cubelist, n):

    # the idea behind it is that we pass the cubelist with the same time coords

    n_aver_cubelist = iris.cube.CubeList()

    dcoord = create_coords(cubelist, n)

    for cube in cubelist:
        n_t = len(cube.coord('time').points)
        if n_t%n!=0:
            # add here a warning that the last is an average of n_t%n==0 years
            print('The n of years is not divisible by '+str(n)+' last '+str(n_t%n)+' years were not taken into account')
        data = np.asarray([np.average(cube.data[n * i:n * i + n], axis=0) for i in range(0, int(n_t / n))])
        n_aver_cube = iris.cube.Cube(data, long_name=cube.long_name + ', ' + str(n) + 'y mean',
                                     var_name=cube.var_name, units=cube.units, attributes=cube.attributes,
                                     dim_coords_and_dims=[(dcoord, 0)])

        n_aver_cubelist.append(n_aver_cube)

    return (n_aver_cubelist)

def ens_averaging(cubelist):

    if len(cubelist) >1:
        for n, cube in enumerate(cubelist):
            cube.add_aux_coord(iris.coords.AuxCoord(n, long_name='n_order', var_name='n_order'))
            if 'precip' in cube.var_name:
                cube.var_name = 'precip_ano'
        equalise_attributes(cubelist)
        ens_cube = cubelist.merge_cube()
        aver_ens_cube = ens_cube.collapsed('n_order', iris.analysis.MEAN)
        aver_ens_cube.remove_coord('n_order')
    else:
        aver_ens_cube = cubelist[0]

    return (aver_ens_cube)

def make_hist_figure(data_dic, cfg):

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))

    plt.style.use(st_file)

    obs_cb = data_dic['OBS']['OBS']['HadEX3'][0]
    era_cb = data_dic['reanalysis']

    proj_list = list(data_dic.keys())
    proj_list.remove('OBS') ; proj_list.remove('reanalysis')    

    colors = {}
    colors['ALL'] = (196 / 255, 121 / 255, 0)
    colors['NAT'] = (0, 79 / 255, 0)

    for proj in proj_list:
        exp_keys = list(data_dic[proj].keys())
        models = data_dic[proj][exp_keys[0]].keys()
        for model in models: 
            fig = plt.figure()
            fig.set_size_inches(12., 8.)
            for exp_key in exp_keys:
                ens_cubelist = data_dic[proj][exp_key][model]
                distrib_data = []
                weights = []
                for cube in ens_cubelist:
                    if model == 'Multi-Model-Mean':
                        cube_weight = cube.attributes['ensemble_weight']*cube.attributes['reverse_dtsts_n']
                    else:
                        cube_weight = cube.attributes['ensemble_weight']
                    for point in cube.data:
                        distrib_data.append(np.around(point, 2))
                        weights.append(cube_weight/len(cube.data))
                distrib_data = np.asarray(distrib_data)
                weights = np.asarray(weights)
                un_weights = np.unique(weights)
                rev_un_weights = np.asarray(1/un_weights).astype('int32')
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
                upd_distr_data = np.asarray(upd_distr_data)
                w_shape, w_loc, w_scale = gev.fit(upd_distr_data)
                t_shape, t_loc, t_scale = gev.fit(distrib_data)
                x_gev = np.arange(-15, 15.1, 0.1)
                w_pdf = gev.pdf(x_gev, w_shape, w_loc, w_scale)
                t_pdf = gev.pdf(x_gev, t_shape, t_loc, t_scale)
                n_bins = np.around(np.arange(-15, 15.1, 0.5), 1)
                plt.hist(distrib_data, bins=n_bins, edgecolor='none',
                        facecolor = colors[exp_key], alpha=0.3, label=exp_key + ' weighted', density=True, weights=weights) 
                plt.plot(x_gev, w_pdf, c = colors[exp_key], ls = 'solid', label = 'GEV '+exp_key+' weighted')
                if model == 'Multi-Model-Mean':
                    plt.hist(distrib_data, bins=n_bins, edgecolor=colors[exp_key],
                            facecolor = 'none', label=exp_key + ' unweighted', density=True)
                    plt.plot(x_gev, t_pdf, c = colors[exp_key], ls = 'dotted', label = 'GEV '+exp_key+' unweighted')
                    if exp_key == 'NAT':
                        plt.text(-5, 0.25, '   unwght\nshape='+str(np.around(t_shape,3))+ '\n loc='+ str(np.around(t_loc,3)) + '\nscale='+str(np.around(t_scale,3)), color= colors[exp_key])
                        plt.text(-5, 0.17, '     wght\nshape='+str(np.around(w_shape,3))+ '\n loc='+ str(np.around(w_loc,3)) + '\nscale='+str(np.around(w_scale,3)), color= colors[exp_key])
                    elif exp_key == 'ALL':
                        plt.text(4, 0.25, '   unwght\nshape='+str(np.around(t_shape,3))+ '\n loc='+ str(np.around(t_loc,3)) + '\nscale='+str(np.around(t_scale,3)), color= colors[exp_key])
                        plt.text(4, 0.17, '     wght\nshape='+str(np.around(w_shape,3))+ '\n loc='+ str(np.around(w_loc,3)) + '\nscale='+str(np.around(w_scale,3)), color= colors[exp_key])
                else: 
                    if exp_key == 'NAT':
                        plt.text(-5, 0.17, 'shape='+str(np.around(w_shape,3))+ '\n loc='+ str(np.around(w_loc,3)) + '\nscale='+str(np.around(w_scale,3)), color= colors[exp_key])
                        plt.text(7.5, 0.35, exp_key+' N realisations = ' + str(len(ens_cubelist)), fontsize = 'large')
                    elif exp_key == 'ALL':
                        plt.text(4, 0.17, 'shape='+str(np.around(w_shape,3))+ '\n loc='+ str(np.around(w_loc,3)) + '\nscale='+str(np.around(w_scale,3)), color= colors[exp_key])
                        plt.text(7.5, 0.325, exp_key+' N realisations = ' + str(len(ens_cubelist)), fontsize = 'large')
            obs_hist = np.histogram(obs_cb.data, bins=n_bins, density=True)
            obs_hist[0][obs_hist[0]==0] = np.nan
            era_hist = np.histogram(era_cb.data, bins=n_bins, density=True)
            era_hist[0][era_hist[0]==0] = np.nan
            plt.scatter(obs_hist[1][:-1] + np.diff(obs_hist[1])/2, obs_hist[0], marker='_', c = 'k', label = 'HadEX3', s=225, lw = 2.5)
            plt.scatter(era_hist[1][:-1] + np.diff(era_hist[1])/2, era_hist[0], marker='_', c = 'r', label = 'ERA5', s=225, lw = 2.5) 

            plt.legend(loc=2, fancybox=False, frameon=False)
            plt.xlim(-11,11)
            plt.xlabel('Temperature anomaly, C')
            plt.ylabel('Number density')

            fig.suptitle('Distribution of TXx anomalies in BC from 1991 to 2020 \n relative to 1951-1980 calculated from '+model +' ('+proj+')', 
                        fontsize = 'x-large')
            fig.set_dpi(250)

            ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_'+proj+'_'+model)
            ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_'+proj+'_'+model,
                                            img_ext='.png')
            plt.close()

    return

def make_timeseries_figure(data_dic, cfg):

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))

    plt.style.use(st_file)

    obs_cb = data_dic['OBS']['OBS']['HadEX3'][0]
    era_cb = data_dic['reanalysis']

    proj_list = list(data_dic.keys())
    proj_list.remove('OBS') ; proj_list.remove('reanalysis') 

    colors = {}
    colors['ALL'] = (196 / 255, 121 / 255, 0)
    colors['NAT'] = (0, 79 / 255, 0)

    for proj in proj_list:
        exp_keys = list(data_dic[proj].keys())
        models = data_dic[proj][exp_keys[0]].keys()
        for model in models:
            fig = plt.figure()
            fig.set_size_inches(12., 8.)
            for exp_key in exp_keys:
                ens_cubelist = data_dic[proj][exp_key][model]
                time_data = []
                weights = []
                for cube in ens_cubelist:
                    if model == 'Multi-Model-Mean':
                        cube_weight = cube.attributes['ensemble_weight']*cube.attributes['reverse_dtsts_n']
                    else:
                        cube_weight = cube.attributes['ensemble_weight']
                    weights.append([cube_weight]*len(cube.data))
                    time_data.append(cube.data)
                time_data = np.asarray(time_data)
                weights = np.asarray(weights)
                tim_coord = cube.coord('time')
                try:
                    tims=cf_units.num2pydate(tim_coord.points, tim_coord.units.origin, calendar=tim_coord.units.calendar)
                except:
                    tims=cf_units.num2pydate(tim_coord.points, tim_coord.units.origin, calendar='gregorian')
                aux_coord = iris.coords.DimCoord(np.arange(0, len(ens_cubelist)), standard_name=None, long_name='aux_ens_coord', var_name='aux_ens_coord')
                full_cube = iris.cube.Cube(time_data, long_name =cube.long_name, var_name = cube.var_name, units = cube.units,  dim_coords_and_dims = [(aux_coord, 0),(tim_coord, 1)])
                if len(ens_cubelist) > 1:
                    mean_cb = full_cube.collapsed('aux_ens_coord', iris.analysis.MEAN, weights = weights)
                    perc_cb = full_cube.collapsed('aux_ens_coord', iris.analysis.WPERCENTILE, percent=[5,95], weights = weights)
                    plt.fill_between(tims, perc_cb.data[0,:], perc_cb.data[1,:], color=colors[exp_key], alpha=0.2, lw=0)
                else: 
                    mean_cb = cube
                plt.plot(tims, mean_cb.data, c=colors[exp_key], label = exp_key, lw=1.5)
            plt.plot(cf_units.num2pydate(obs_cb.coord('time').points, obs_cb.coord('time').units.origin, calendar=obs_cb.coord('time').units.calendar), obs_cb.data, lw=1.5, c='k', label = 'HadEX3')
            plt.plot(cf_units.num2pydate(era_cb.coord('time').points, era_cb.coord('time').units.origin, calendar=era_cb.coord('time').units.calendar), era_cb.data, lw=1.5, c='r', label = 'ERA5')

            plt.legend(loc=2, fancybox=False, frameon=False)
            plt.ylim(-10,10)
            plt.ylabel('Temperature anomaly, C')

            fig.suptitle('Timeseries of TXx anomalies in BC from 1991 to 2020\nrelative to 1951-1980 from ' +model+' ('+proj+')', fontsize = 'x-large')
            fig.set_dpi(250)
            
            ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_timeseries_'+proj +'_'+model)
            ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_timeseries_'+proj +'_'+model,
                                            img_ext='.png')   
            plt.close()

    return


def main(cfg):

    vrbls = Variables(cfg)
    all_dtsts = Datasets(cfg)

    era_cube = get_era_txx(cfg)

    rename_variables(vrbls, all_dtsts)
    vrbl = list(set(all_dtsts.get_info_list('short_name')))[0]

    plotting_dic = {}

    raw_exp_keys = ['ALL', 'NAT', 'OBS']

    # for raw_exp_key in raw_exp_keys:
    #     plotting_dic[raw_exp_key] = {}

    projects = set(all_dtsts.get_info_list('project', short_name=vrbl))
    obs_filename = all_dtsts.get_dataset_info(project='OBS',
                                              short_name=vrbl)['filename']
    obs_mask = create_obs_mask(obs_filename, vrbl)
    for project in sorted(projects):
        plotting_dic[project] = {}
        exp_keys = reform_exp_keys(project, raw_exp_keys)
        for exp_key in sorted(exp_keys):
            plotting_dic[project][exp_key] = {}
            dtsts, exp = obtain_datasets(all_dtsts, project=project,
                                         short_name=vrbl, exp_key=exp_key)
            ens_cubelist = iris.cube.CubeList()
            for dtst in dtsts:
                flpths = obtain_filepaths(all_dtsts, dataset=dtst,
                                          project=project, short_name=vrbl,
                                          exp=exp)
                n_real = len(flpths)
                mod_cubelist = iris.cube.CubeList()
                for flfpth in flpths:
                    mod_cb = iris.load_cube(flfpth['filename'])
                    mod_cb = apply_obs_mask(mod_cb, obs_mask)
                    anomal_cb = esmvalcore.preprocessor.anomalies(mod_cb, 'full',
                                                        reference={'start_year': cfg['reference_period'][0],
                                                        'start_month': 1, 'start_day':1,
                                                        'end_year': cfg['reference_period'][1],
                                                        'end_month': 12, 'end_day':31} )
                    crop_cb = esmvalcore.preprocessor.extract_time(anomal_cb,
                                                    start_year=cfg['plotting_period'][0],
                                                    start_month=1, start_day=1,
                                                    end_year=cfg['plotting_period'][1],
                                                    end_month=12, end_day=31)
                    wght_mod_cb = esmvalcore.preprocessor.area_statistics(crop_cb, 'mean')
                    wght_mod_cb.attributes['ensemble_weight'] = 1 / n_real
                    wght_mod_cb.attributes['reverse_dtsts_n'] = 1/ len(dtsts)
                    provenance_rec= { 'authors' : 'malinina_elizaveta', 'statistics': 'max', 'ancestors': [flfpth['filename']]}
                    if flfpth['project'] == 'OBS':
                        basename = flfpth['short_name'] +'_'+ flfpth['dataset']+'_'+flfpth['project']
                    else:
                        basename = flfpth['short_name'] +'_'+ flfpth['dataset']+'_'+flfpth['exp'] + '_'+ flfpth['ensemble']
                    save_data(basename, provenance_rec, cfg, wght_mod_cb)
                    ens_cubelist.append(wght_mod_cb)
                    mod_cubelist.append(wght_mod_cb)
                plotting_dic[project][exp_key][dtst] = mod_cubelist
            if exp_key != 'OBS':
                plotting_dic[project][exp_key]['Multi-Model-Mean'] = ens_cubelist      

    era_cube = apply_obs_mask(era_cube, obs_mask)
    plotting_dic['reanalysis'] = era_cube

    make_hist_figure(plotting_dic, cfg)

    make_timeseries_figure(plotting_dic, cfg)                                   

    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
