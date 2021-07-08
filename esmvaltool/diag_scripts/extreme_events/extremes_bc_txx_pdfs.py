import cf_units
import cftime
import datetime

import esmvalcore.preprocessor
import iris
from iris.experimental.equalise_cubes import equalise_attributes
import iris.plot as iplt
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import genextreme as gev

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, Datasets, Variables
from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
from esmvalcore.preprocessor import regrid
import esmvaltool.diag_scripts.shared.plot as eplot
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools

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


def make_panel(data_dic, variable, exp_key, nrows, ncols, idx, ref_period, obs_cube):

    ax = plt.subplot(nrows, ncols, idx)

    ax.set_xlim(datetime.datetime(1948, 1, 1), datetime.datetime(2022, 1, 1))
    ax.plot([datetime.datetime(1948, 1, 1), datetime.datetime(2022, 1, 1)],
            [0, 0], c='silver', linewidth=0.7, linestyle='dashed')
    ax.fill_betweenx([15, -15], [datetime.datetime(ref_period[0], 1, 1),
                                 datetime.datetime(ref_period[0], 12, 31)],
                     [datetime.datetime(ref_period[1], 1, 1),
                      datetime.datetime(ref_period[1], 12, 31)], alpha=0.1,
                     color='lightgrey', linewidth=0)

    colors = {}

    colors['CMIP5'] = (37 / 255, 81 / 255, 204 / 255)
    colors['CMIP6'] = (204 / 255, 35 / 255, 35 / 255)
    colors['ribbon'] = (37 / 255, 81 / 255, 204 / 255)
    colors['lines'] = (204 / 255, 35 / 255, 35 / 255)

    if variable == 'rx1day':
        ax.set_ylabel('Rx1day (%)')
        if idx < 3:
            ax.set_title('Annual maximum 1-day precipitation \n (Rx1day)')
        ax.set_ylim(-5.5, 10.5)
        ax.set_yticks(np.arange(-5,11,5))
        txt_y = 3.12
        txt_x = datetime.datetime(1968, 7, 1)
        ax.text(datetime.datetime(ref_period[0]+4, 7, 1), -4.5, 'Reference period', color='grey')

    elif variable == 'txx':
        ax.set_ylabel('TXx ($^o$C)')
        if idx < 3:
            ax.set_title('Annual maximum daily \n maximum temperature (TXx)')
        ax.set_ylim(-1.2, 2.4)
        ax.set_yticks(np.arange(-1,3,1))
        txt_y = 0.6
        txt_x = datetime.datetime(1950, 1, 1)
        ax.text(datetime.datetime(ref_period[0] + 4, 7, 1), -1,
                'Reference period', color='grey')

    for proj in data_dic.keys():
        tim = data_dic[proj]['mean'].coord('time')
        conv_time = cftime.num2pydate(tim.points, tim.units.origin, tim.units.calendar)
        ax.fill_between(conv_time, data_dic[proj]['5_95_perc'][0].data,
                        data_dic[proj]['5_95_perc'][1].data, color=colors[proj],
                        alpha=0.2, linewidth=0)
        iplt.plot(data_dic[proj]['mean'], label=proj +' ('+
                                            str(data_dic[proj]['n_models'])+
                                            ')', c=colors[proj],
                  linestyle='solid', axes=ax)

    iplt.plot(obs_cube, c='k', linestyle='solid', axes=ax)
    ax.text(txt_x, txt_y, 'Observations:\n   '+obs_cube.attributes['title'].split()[0])


    if exp_key == 'ALL':
        label = 'Natural and Human Forcing'
    elif exp_key == 'NAT':
        label = 'Natural Forcing'

    leg = ax.legend(loc=2, frameon=False, ncol=1, title=label, handlelength=0,
                    fontsize='medium', title_fontsize='large', handletextpad=0.2)

    for nt, txt in enumerate(leg.get_texts()):
        col = leg.legendHandles[nt].get_color()
        txt.set_color(col)

    leg._legend_box.align = 'left'

    ax.set_xlabel('Year')

    return

def make_figure(data_dic, cfg):

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))

    plt.style.use(st_file)

    obs_cb = data_dic.pop('OBS')

    colors = {}
    colors['ALL'] = (196 / 255, 121 / 255, 0)
    colors['NAT'] = (0, 79 / 255, 0)

    fig = plt.figure()
    fig.set_size_inches(12., 8.)
    for exp_key in sorted(data_dic.keys()):
        ens_cubelist = data_dic[exp_key]
        distrib_data = []
        for cube in ens_cubelist:
            for point in cube.data:
                distrib_data.append(np.around(point - 273.15, 1))
        distrib_data = np.asarray(distrib_data)
        n_bins = int((np.max(distrib_data) - np.min(distrib_data))*10)
        plt.hist(distrib_data, bins=n_bins, color=colors[exp_key], label=exp_key)
    for n, obs_point in enumerate(obs_cb.data):
        plt.arrow(obs_point-273.15, 24, 0, -7, color='k')
        plt.text(obs_point-273.4, 24.5, str(10+n))

    plt.legend(loc=2, fancybox=False, frameon=False)
    plt.xlim(20,51)
    plt.ylim(0,25)
    plt.xlabel('Temperature, C')
    plt.ylabel('number of occurences')

    fig.suptitle('Distribution of maximal TXx in BC\n from 2010 to 2017', fontsize = 'x-large')
    fig.set_dpi(250)

    return

def main(cfg):

    vrbls = Variables(cfg)
    all_dtsts = Datasets(cfg)

    rename_variables(vrbls, all_dtsts)
    vrbl = list(set(all_dtsts.get_info_list('short_name')))[0]

    plotting_dic = {}

    raw_exp_keys = ['ALL', 'NAT', 'OBS']
    projects = set(all_dtsts.get_info_list('project', short_name=vrbl))
    obs_filename = all_dtsts.get_dataset_info(project='OBS',
                                              short_name=vrbl)['filename']
    obs_mask = create_obs_mask(obs_filename, vrbl)
    for project in sorted(projects):
        exp_keys = reform_exp_keys(project, raw_exp_keys)
        for exp_key in sorted(exp_keys):
            dtsts, exp = obtain_datasets(all_dtsts, project=project,
                                         short_name=vrbl, exp_key=exp_key)
            ens_cubelist = iris.cube.CubeList()
            for dtst in dtsts:
                flpths = obtain_filepaths(all_dtsts, dataset=dtst,
                                          project=project, short_name=vrbl,
                                          exp=exp)
                for flfpth in flpths:
                    mod_cb = iris.load_cube(flfpth['filename'])
                    mod_cb = dataset_regriding(mod_cb, exp, obs_filename)
                    mod_cb = apply_obs_mask(mod_cb, obs_mask)
                    wght_mod_cb = esmvalcore.preprocessor.area_statistics(mod_cb, 'max')
                    ens_cubelist.append(wght_mod_cb)
            if exp_key == 'OBS':
                plotting_dic[exp_key] = ens_cubelist[0]
            else:
                plotting_dic[exp_key] = ens_cubelist

    make_figure(plotting_dic, cfg)

    ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes')
    ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes',
                                      img_ext='.png')

    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
