; #############################################################################
; ESMValTool CMORizer for CloudSat Level 2 data
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    http://www.cloudsat.cira.colostate.edu/
;    ftp://ftp.cloudsat.cira.colostate.edu/
;
; Last access
;    20201107
;
; Download and processing instructions:
;
;    1) Create an account at the CloudSat Data Processing Center
;       (http://www.cloudsat.cira.colostate.edu/)
;
;    2) Download the datasets '2B-CWC-RO' and '2C-PRECIP-COLUMN' from
;       ftp.cloudsat.cira.colostate.edu (e.g. using wget):
;
;       wget -nd -N -r --user=<user name> --password=<password>
;          ftp://ftp.cloudsat.cira.colostate.edu/2B-CWC-RO.P1_R05/
;       wget -nd -N -r --user=<user name> --password=<passsword>
;          ftp://ftp.cloudsat.cira.colostate.edu/2C-PRECIP-COLUMN.P1_R05/
;
;    3) Put all files under a single directory (no subdirectories with
;       years and days). The wget commands given above do this automatically.
;
; Modification history
;    20201109-lauer_axel: written.
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")

begin
  ; Script name (for logger)
  diag_script = "cmorize_obs_cloudsat-l2.ncl"

  ; Source name
  obsname = "CLOUDSAT-L2"

  ; Tier
  tier = 2

  ; Period
  year1 = 2007
  year2 = 2007

  ; output variable (standard name)
  outvar = "clw"

  ; input variables
  var = "LO_RO_liquid_water_content_2B_CWC_RO"
  var_flag = "Precip_flag_2C_PRECIP_COLUMN"

  ; MIP
  mip = "Amon"

  ; Frequency
  freq = "mon"

  ; CMOR table
  CMOR_TABLE = getenv("cmor_tables") + "/cmip5/Tables/CMIP5_" + mip

  ; Type
  type = "sat"

  ; Version
  version = "P1-R05"

  ; Global attributes
  source = "http://www.cloudsat.cira.colostate.edu/"
  ref = ""
end

begin
  ; define 2x2 output grid

  nx = 180
  ny = 90
  dx = 360.0 / nx
  dy = 180.0 / ny

  ; alt40 from CMIP6 table "coordinates"
  gridhgt = (/240., 720., 1200., 1680., 2160., 2640., 3120., 3600., 4080.,   \
              4560., 5040., 5520., 6000., 6480., 6960., 7440., 7920., 8400., \
              8880., 9360., 9840., 10320., 10800., 11280., 11760., 12240.,   \
              12720., 13200., 13680., 14160., 14640., 15120., 15600., 16080.,\
              16560., 17040., 17520., 18000., 18480., 18960./)

  nz = dimsizes(gridhgt)
  nt = 12

  ; define output longitudes and latitudes

  gridlon = fspan(dx / 2.0, 360.0 - dx / 2.0, nx)
  gridlat = fspan(-90.0 + dy / 2.0, 90.0 - dy / 2.0, ny)

  ; define work arrays

  ; "in-cloud" (all points)
  grid = new((/nt, nz, ny, nx/), float)
  gridpts = new((/nt, nz, ny, nx/), integer)

  ; "grid-box average" (all points)
  grid_avg = new((/nt, nz, ny, nx/), float)
  gridpts_avg = new((/nt, nz, ny, nx/), integer)

  ; "in-cloud" (no precipitation)
  grid_noprecip = new((/nt, nz, ny, nx/), float)
  gridpts_noprecip = new((/nt, nz, ny, nx/), integer)

  ; "grid-box average" (no precipitation)
  grid_avg_noprecip = new((/nt, nz, ny, nx/), float)
  gridpts_avg_noprecip = new((/nt, nz, ny, nx/), integer)

  ; boundaries of vertical (height) bins

  gridhgt_bnds = new(nz + 1, float)
  gridhgt_bnds(0) = 0.
  do i = 1, nz - 1
    gridhgt_bnds(i) = (gridhgt(i - 1) + gridhgt(i)) / 2.0
  end do
  gridhgt_bnds(nz) = gridhgt(nz - 1) + gridhgt(nz - 1) - gridhgt_bnds(nz - 1)

  ; air density of vertical bins (from US standard atmosphere) in kg/m3
  tdp = stdatmus_z2tdp(gridhgt)
  air_density = tdp(1, :)
  delete(tdp)

  ; define coordinates

  grid!0 = "time"
  grid!1 = "alt40"
  grid&alt40 = gridhgt
  grid!2 = "lat"
  grid&lat = gridlat
  grid!3 = "lon"
  grid&lon = gridlon

  copy_VarMeta(grid, grid_avg)
  copy_VarMeta(grid, grid_noprecip)
  copy_VarMeta(grid, grid_avg_noprecip)
end

begin
  do calyear = year1, year2
    ; reset work arrays
    grid = 0.
    grid_avg = 0.
    grid_noprecip = 0.
    grid_avg_noprecip = 0.

    gridpts = 0
    gridpts_avg = 0
    gridpts_noprecip = 0
    gridpts_avg_noprecip = 0

    ; create time variable
    gridtime = create_timec(calyear, calyear)

    grid&time = gridtime
    grid_avg&time = gridtime
    grid_noprecip&time = gridtime
    grid_avg_noprecip&time = gridtime

    ; process all days of the year

    do dayofyear = 1, day_of_year(calyear, 12, 31)
      caldate = yyyyddd_to_yyyymmdd(calyear * 10000 + dayofyear)
      calmonth = mod(caldate / 100, 100)
      calday = mod(caldate, 100)

      time_idx = calmonth - 1

      fn = tostring(calyear) + sprinti("%0.3i", dayofyear)
      fs = systemfunc("ls " + input_dir_path + fn + "*2B-CWC-RO_GRANULE*.hdf")

      if (ismissing(fs(0))) then
         log_info("Warning: no input file(s) for " + tostring(calyear) + \
           "-" + sprinti("%03d", dayofyear) + ", skipping day.")
         delete(fs)
         continue
      end if

      norbits = dimsizes(fs)

      ; start loop over all orbits for one day

      do orbit = 0, norbits - 1
        ; combine filename for precipitation data
        ssplit = str_split(fs(orbit), "/")
        strs = str_split(ssplit(dimsizes(ssplit) - 1), "_")
        fs_precip = input_dir_path + strs(0) + "_" + strs(1) + \
          "_CS_2C-PRECIP-COLUMN_GRANULE_P1_R05_E02_F00.hdf"
        delete(ssplit)
        delete(strs)

        ; File extension must be .hdfeos for NCL to correctly read input
        ; files. File extension of cloudsat data downloaded from cloudsat
        ; data processing center is .hdf.
        ; --> workaround: set symbolic links with extension .hdfeos before
        ;     reading input files

        ssplit = str_split(fs(orbit), "/")
        slink1 = ssplit(dimsizes(ssplit) - 1) + "eos"
        delete(ssplit)

        ssplit = str_split(fs_precip, "/")
        slink2 = ssplit(dimsizes(ssplit) - 1) + "eos"
        delete(ssplit)

        ; remove old (possibly broken) links before creating new ones

        if (fileexists(slink1)) then
          system("rm -f " + link1)
        end if

        if (fileexists(slink2)) then
          system("rm -f " + link2)
        end if

        ; create symbolic links

        system("ln -s " + fs(orbit) + " " + slink1)
        system("ln -s " + fs_precip + " " + slink2)

        ;***************************************************************
        ; Read hdf-eos
        ;***************************************************************

        print("processing " + fs(orbit) + " + " + fs_precip + "...")

        f = addfile(slink1, "r")
        x = f->$var$

        if (isatt(x, "_FillValue")) then
          delete(x@_FillValue)
        end if
        if (isatt(x, "missing")) then
          delete(x@missing)
        end if

        dimd    = dimsizes(x)
        nray    = dimd(0)
        nbin    = dimd(1)

        f_precip = addfile(slink2, "r")
        x_flag = f_precip->$var_flag$

        ; remove '_FillValue' and 'missing' (unused) attributes to prevent NCL
        ; from handling missing values

        if (isatt(x_flag, "_FillValue")) then
          delete(x_flag@_FillValue)
        end if
        if (isatt(x_flag, "missing")) then
          delete(x_flag@missing)
        end if

        ; check if array dimensions match

        dimd2 = dimsizes(x_flag)

        if (dimd(0) .ne. dimd2) then
          print("Error: dimsize does not match. " + fs(orbit) + ", " + \
            fs_precip + " Abort.")
          exit
        end if

        lat     = f->Latitude_2B_CWC_RO
        lon     = f->Longitude_2B_CWC_RO
        hgt     = f->Height_2B_CWC_RO
;        time    = f->Profile_time_2B_CWC_RO

        ; convert longitudes from -180...180 to 0...360
        ; convert latitudes from -90...90 to 0...180

        lon = where(lon .lt. 0., lon + 360., lon)
        lat = lat + 90.

        ; --------------------------------------------------------------------

        do i = 0, nray - 1
          ; apply modulo function because some input files contain
          ; points with longitude = 360.0
          ; --> mapped to ilon = 0
          ilat = mod(toint(lat(i) / dy), ny)
          ilon = mod(toint(lon(i) / dx), nx)
          do ilev = 0, nz - 1
            iz = ind((hgt(i, :) .ge. gridhgt_bnds(ilev)) .and. (hgt(i, :) \
              .lt. gridhgt_bnds(ilev + 1)))
            if (.not.ismissing(iz(0))) then

              ; *** Calculate "in-cloud" average ***
              ; Filter invalid (x < 0) and missing values (x = 0) before
              ; averaging.

              iz_valid = ind(x(i, iz) .gt. 0.)
              if (.not.ismissing(iz_valid(0))) then
                idx = iz(iz_valid)
                vpts = dimsizes(idx)
                vsum = sum(x(i, idx))
                gridpts(time_idx, ilev, ilat, ilon) = \
                  gridpts(time_idx, ilev, ilat, ilon) + vpts
                grid(time_idx, ilev, ilat, ilon) = \
                  grid(time_idx, ilev, ilat, ilon) + vsum

                ; Now also filter cases that have been flagged with
                ; precipitation or as unknown. The following CloudSat flags
                ; are defined (see http://www.cloudsat.cira.colostate.edu/
                ; sites/default/files/
                ; products/files/2C-PRECIP-COLUMN_PDICD.P2_R04.20130124.pdf):
                ;
                ; flag 0 = no precipitation detected
                ;      1 = rain possible
                ;      2 = rain probable
                ;      3 = rain certain
                ;      4 = snow possible
                ;      5 = snow certain
                ;      6 = mixed precipitation possible
                ;      7 = mixed precipitation certain
                ;      9 = uncertain

                if (x_flag(i) .ne. 0) then
                  gridpts_noprecip(time_idx, ilev, ilat, ilon) = \
                    gridpts_noprecip(time_idx, ilev, ilat, ilon) + vpts
                  grid_noprecip(time_idx, ilev, ilat, ilon) = \
                    grid_noprecip(time_idx, ilev, ilat, ilon) + vsum
                end if

                delete(idx)
              end if
              delete(iz_valid)

              ; *** Calculate "gridbox average" ***
              ; Also include all points with missing cloud liquid water
              ; content (x = 0) when calculating the average. This assumes
              ; that all grid points with "missing" but not invalid values
              ; are cloud-free. Invalid values (x < 0) are filtered out.

              data = where(x(i, iz) .lt. 0., 0., x(i, iz))
              vsum = sum(data)
              vpts = dimsizes(iz)
              delete(data)

              grid_avg(time_idx, ilev, ilat, ilon) = \
                grid_avg(time_idx, ilev, ilat, ilon) + vsum
              gridpts_avg(time_idx, ilev, ilat, ilon) = \
                gridpts_avg(time_idx, ilev, ilat, ilon) + vpts

              ; Now also filter cases that have been flagged with
              ; precipitation or as unknown (see definition of flags above).

              if (x_flag(i) .ne. 0) then
                gridpts_avg_noprecip(time_idx, ilev, ilat, ilon) = \
                  gridpts_avg_noprecip(time_idx, ilev, ilat, ilon) + vpts
                grid_avg_noprecip(time_idx, ilev, ilat, ilon) = \
                  grid_avg_noprecip(time_idx, ilev, ilat, ilon) + vsum
              end if
            end if  ; if cloudsat point are available for this height bin
            delete(iz)
          end do  ; loop over all (ouput) height bins
        end do  ; loop over all points along satellite track

        delete(x)
        delete(x_flag)
            delete(lat)
        delete(lon)
        delete(hgt)
;        delete(time)

        ; remove temporary symbolic links

        system("rm -f " + slink1)
        system("rm -f " + slink2)

      end do  ; loop over all orbits per day
      delete(fs)
    end do  ; loop over days

    ; calculate monthly averages

    gridpts = where(gridpts .eq. 0, gridpts@_FillValue, gridpts)
    gridpts_avg = where(gridpts_avg .eq. 0, gridpts_avg@_FillValue, gridpts_avg)
    gridpts_noprecip = where(gridpts_noprecip .eq. 0, \
      gridpts_noprecip@_FillValue, gridpts_noprecip)
    gridpts_avg_noprecip = where(gridpts_avg_noprecip .eq. 0, \
      gridpts_avg_noprecip@_FillValue, gridpts_avg_noprecip)

    grid = grid / gridpts
    grid_avg = grid_avg / gridpts_avg
    grid_noprecip = grid_noprecip / gridpts_noprecip
    grid_avg_noprecip = grid_avg_noprecip / gridpts_avg_noprecip

    ; convert from mg/m3 to kg/kg using air density from
    ; US standard atmosphere

    grid = grid / conform(grid, air_density, 1) * 1.0e-6
    grid_avg = grid_avg / conform(grid_avg, air_density, 1) * 1.0e-6
    grid_noprecip = grid_noprecip / \
      conform(grid_noprecip, air_density, 1) * 1.0e-6
    grid_avg_noprecip = grid_avg_noprecip / \
      conform(grid_avg_noprecip, air_density, 1) * 1.0e-6

    ; ------------------------------------------------------------------------

    ; save results to files

    ; calculate coordinate bounds
    bounds = guess_coord_bounds(grid, freq)

    ; *** in-cloud, all ***

    format_coords(grid, calyear + "0101", calyear + "1231", freq)
    ; Set variable attributes
    output = format_variable(grid, outvar, CMOR_TABLE)
    delete(grid)
    ; Set global attributes
    comment = "Liquid-only Radar-only Liquid Water Content (in-cloud, all)"
    gAtt = set_global_atts(obsname, tier, source, ref, comment)
    ; output filename
    datestr = calyear + "01-" + calyear + "12"
    fout = output_dir_path + \
      str_join((/"OBS", obsname, type, version + "-in-cloud-all", \
                 mip, outvar, datestr/), "_") + ".nc"
    ; write variable
    write_nc(fout, outvar, output, bounds, gAtt)
    delete(output)
    delete(gAtt)

    ; *** gridbox average, all ***

    format_coords(grid_avg, calyear + "0101", calyear + "1231", freq)
    ; Set variable attributes
    output = format_variable(grid_avg, outvar, CMOR_TABLE)
    delete(grid_avg)
    ; Set global attributes
    comment = "Liquid-only Radar-only Liquid Water Content " \
      + "(gridbox average, all)"
    gAtt = set_global_atts(obsname, tier, source, ref, comment)
    ; output filename
    datestr = calyear + "01-" + calyear + "12"
    fout = output_dir_path + \
      str_join((/"OBS", obsname, type, version + "-gridbox-average-all", \
                 mip, outvar, datestr/), "_") + ".nc"
    ; write variable
    write_nc(fout, outvar, output, bounds, gAtt)
    delete(output)
    delete(gAtt)

    ; *** in-cloud, no precipitation only ***

    format_coords(grid_noprecip, calyear + "0101", calyear + "1231", freq)
    ; Set variable attributes
    output = format_variable(grid_noprecip, outvar, CMOR_TABLE)
    delete(grid_noprecip)
    ; Set global attributes
    comment = "Liquid-only Radar-only Liquid Water Content " \
      + "(in-cloud, no precipitation)"
    gAtt = set_global_atts(obsname, tier, source, ref, comment)
    ; output filename
    datestr = calyear + "01-" + calyear + "12"
    fout = output_dir_path + \
      str_join((/"OBS", obsname, type, version + "-in-cloud-noprecip", \
                 mip, outvar, datestr/), "_") + ".nc"
    ; write variable
    write_nc(fout, outvar, output, bounds, gAtt)
    delete(output)
    delete(gAtt)

    ; *** gridbox average, no precipitation only ***

    format_coords(grid_avg_noprecip, calyear + "0101", calyear + "1231", freq)
    ; Set variable attributes
    output = format_variable(grid_avg_noprecip, outvar, CMOR_TABLE)
    delete(grid_avg_noprecip)
    ; Set global attributes
    comment = "Liquid-only Radar-only Liquid Water Content " \
      + "(gridbox average, no precipitation)"
    gAtt = set_global_atts(obsname, tier, source, ref, comment)
    ; output filename
    datestr = calyear + "01-" + calyear + "12"
    fout = output_dir_path + \
      str_join((/"OBS", obsname, type, version + \
        "-gridbox-average-noprecip", mip, outvar, datestr/), "_") + ".nc"
    ; write variable
    write_nc(fout, outvar, output, bounds, gAtt)
    delete(output)
    delete(gAtt)

;    system("rm debug.nc")
;    debugfile = addfile("debug.nc", "c")
;    debugfile->clw = grid
;    debugfile->clw_avg = grid_avg
;    debugfile->clw_noprecip = grid_noprecip
;    debugfile->clw_avg_noprecip = grid_avg_noprecip
;    debugfile->pts = gridpts
;    x1d = ndtooned(x)
;    xreshape = reshape(x1d(0:4633999), (/2000, 2317/))
;    debugfile->x = xreshape

  end do  ; loop over years

end
