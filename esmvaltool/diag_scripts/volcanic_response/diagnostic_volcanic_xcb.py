import logging
import os
import sys
import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import datetime as dt
import cftime
import esmvalcore.preprocessor as epreproc

from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
from esmvaltool.diag_scripts.shared import run_diagnostic, group_metadata, select_metadata
import esmvaltool.diag_scripts.shared.plot as eplot

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def calculate_nino34(cubelist, cfg):

    nino_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        tos_clim = epreproc.anomalies(cube, period='monthly')
        nino_idx = epreproc.area_statistics(tos_clim, operator='mean')
        std_tos = cube.data.std()
        nino_cube = nino_idx / std_tos
        if ('plot_start_year' in cfg.keys()) & ('plot_start_year' in cfg.keys()):
            nino_cube = epreproc.extract_time(nino_cube, cfg['plot_start_year'], 1, 1, cfg['plot_end_year']+1, 1, 1)
        nino_cube.var_name = 'nino34_sst'
        nino_cube.long_name = 'El Nino 3.4 sea surface temperature'
        nino_cubelist.append(nino_cube)

    return nino_cubelist


def derive_vars(variable, cubelist, cfg):

    if variable == 'siconc':
        sia_cubelist = ipcc_sea_ice_diag.calculate_siarea(cubelist)
        derived_cubelist = iris.cube.CubeList()
        start_year = cfg ['anomaly_period'][0]
        end_year = cfg ['anomaly_period'][0]
        for sia_cube in sia_cubelist:
            der_cube = epreproc.anomalies(sia_cube, period='monthly',
                                          reference = {'start_year': start_year, 'start_month': 1, 'start_day':1,
                                                       'end_year': end_year, 'end_month': 12, 'end_day':31} )
            der_cube = epreproc.extract_time(der_cube, start_year=cfg['plot_start_year'], start_month=1, start_day=1,
                                                       end_year=cfg['plot_end_year'], end_month=12, end_day=31)
            derived_cubelist.append(der_cube)
    elif variable == 'tos':
        derived_cubelist = calculate_nino34(cubelist, cfg)
    elif variable == 'tas':
        derived_cubelist = iris.cube.CubeList()
        for cube in cubelist:
            der_cube = epreproc.extract_time(cube,
                                             start_year=cfg['plot_start_year'],
                                             start_month=1, start_day=1,
                                             end_year=cfg['plot_end_year'],
                                             end_month=12, end_day=31)
            derived_cubelist.append(der_cube)
    else:
        derived_cubelist = cubelist



    return derived_cubelist

def make_plot(data_dic, cfg):

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))

    plt.style.use(st_file)
    fig = plt.figure()
    fig.set_size_inches(9., 9.)

    for nv, var in enumerate(data_dic.keys()):
        ax = plt.subplot(2,2,nv+1)
        mean = data_dic[var]['mean']
        p5 = data_dic[var]['p5']
        p95 = data_dic[var]['p95']
        time_raw = data_dic[var]['mean'].coord('time')
        times = cftime.num2pydate(time_raw.points, time_raw.units.origin,
                                  time_raw.units.calendar)
        # if var == 'tas':
        #     ax.plot(times, mean.data - 273.15, c='#1f77b4', linewidth = 1.5)
        #     [ax.plot(times, dtst.data - 273.15, c='#1f77b4', linewidth=0.5, alpha = 0.2) for
        #      dtst in data_dic[var]['all_data']]
        #     # ax.fill_between(times, (p5 - 273.15).data, (p95- 273.15).data, alpha = 0.2, linewidth=0, color='#1f77b4')
        if var == 'pr':
            ax.plot(times, mean.data / (86400*10e6), c='#1f77b4', linewidth=1.5)
            [ax.plot(times, dtst.data / (86400*10e6), c='#1f77b4', linewidth=0.5, alpha=0.2) for
             dtst in data_dic[var]['all_data']]
        else:
            ax.plot(times, mean.data, c = '#1f77b4', linewidth = 1.5)
            [ax.plot(times, dtst.data , c='#1f77b4', linewidth=0.5, alpha = 0.2) for
             dtst in data_dic[var]['all_data']]
        if var == 'tas':
            ax.set_title('GMST anomaly')
            ax.set_ylabel(r'$\Delta$ T ($^o$C)')
        elif var == 'tos':
            ax.set_title('Nino 3.4 SST')
            ax.set_ylabel(r'$\Delta$ T ($^o$C)')
            xlims = ax.get_xlim()
            ax.set_xlim(xlims[0], xlims[1])
            ax.fill_between((xlims[0], xlims[1]), -0.4, 0.4, color='silver', linewidth=0, alpha=0.3)
        elif var == 'siconc':
            ax.set_title('Arctic SIA anomaly')
            ax.set_ylabel(r'$\Delta$ sia (10$^6$ km$^2$)')
        elif var == 'pr':
            ax.set_title('SE Asia pr totals')
            ax.set_ylabel(r'pr_tot (10$^6$ mm)')
            ax.set_ylim(0, 1)
        ylims = ax.get_ylim()
        ax.set_ylim(ylims[0], ylims[1])
        ax.vlines(dt.datetime.fromisoformat(cfg['eruption_date']),
                  ylims[0], ylims[1]*1.1, alpha=0.7, colors='silver', linestyle='dashed')

    fig.suptitle('Changes in climate variables after '+cfg['volcano_name']+' eruption on ' + cfg['eruption_date'])

    plt.tight_layout()

    return

def main(cfg):

    input_data = cfg ['input_data'].values()
    variables = group_metadata(input_data, 'short_name').keys()

    plot_data_dict = dict()

    for var in variables:
        var_info_list = select_metadata(input_data, short_name=var)
        datasets = set([var_info_list[i]['dataset'] for i in range(len(var_info_list))])
        datasets_cubelist = iris.cube.CubeList()
        dataset_cubelist_perc = iris.cube.CubeList()
        for dataset in datasets:
            entries = select_metadata(input_data, short_name=var, dataset=dataset)
            data_cubelist = iris.load([entries[i]['filename'] for i in range(len(entries))])
            data_cubelist = derive_vars(var, data_cubelist, cfg)
            for d_cube in data_cubelist:
                dataset_cubelist_perc.append(d_cube)
            data_cube = epreproc.multi_model_statistics(data_cubelist, span='full', statistics=['mean'])
            try:
                datasets_cubelist.append(data_cube['mean'])
            except:
                datasets_cubelist.append(data_cube[0])
        var_cube = epreproc.multi_model_statistics(datasets_cubelist, span='full', statistics= ['mean','p5','p95'])
        var_cube.update({'all_data': datasets_cubelist})
        # perc_cubes = epreproc.multi_model_statistics(dataset_cubelist_perc, span='full', statistics=['p5','p95'])
        # plot_data_dict [var] = var.update(perc_cubes)
        plot_data_dict [var] = var_cube

    make_plot(plot_data_dict, cfg)

    ipcc_sea_ice_diag.figure_handling(cfg, name='fig_volc_xcb_timeseries')

    logger.info('Success')

if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
