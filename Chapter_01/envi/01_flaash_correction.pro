; =============================================================================
; envi/01_flaash_correction.pro
; =============================================================================
; ENVI 5.6 FLAASH Atmospheric Correction - IDL Batch Script
; Study area: Torres del Paine, Patagonia, Chile (approx. 51 deg S)
;
; PURPOSE:
;   Automates FLAASH (Fast Line-of-sight Atmospheric Analysis of Spectral
;   Hypercubes) atmospheric correction for Landsat L1TP (TOA radiance) imagery.
;   FLAASH uses the MODTRAN5 radiative transfer model to convert at-sensor
;   radiance to ground surface reflectance.
;
; WHEN TO USE FLAASH vs. using pre-corrected products:
;   - Sentinel-2 L2A  : ALREADY corrected by ESA Sen2Cor. Do NOT use FLAASH.
;   - Landsat 9 L2SP  : ALREADY corrected by USGS LaSRC. Do NOT use FLAASH.
;   - Sentinel-2 L1C  : Raw TOA. USE FLAASH or Sen2Cor.
;   - Landsat 9 L1TP  : Raw TOA radiance. USE FLAASH or DOS1.
;   Applying FLAASH to already-corrected L2 products will corrupt the data.
;
; FLAASH PARAMETERS FOR TORRES DEL PAINE (~51 deg S, MARITIME):
;   Atmospheric model : Sub-Arctic Summer (closest match for 51 deg S summer)
;   Aerosol model     : Maritime (Pacific Ocean influence)
;   Initial visibility: 40 km (typical clear Patagonian atmosphere)
;   Water retrieval   : Enabled (1135 nm water absorption feature)
;   Aerosol retrieval : Enabled
;
; HOW TO RUN IN ENVI 5.6:
;   Option A - ENVI Toolbox:
;     1. Open ENVI 5.6
;     2. Toolbox > ENVI Modeler > Run IDL Script
;     3. Select this .pro file
;     4. Set INPUT_FILE to your Landsat L1TP radiance file
;
;   Option B - IDL Console:
;     1. Open ENVI 5.6 (which opens IDL console)
;     2. In IDL console: .compile 'D:\...\Chapter_01\envi\01_flaash_correction.pro'
;     3. In IDL console: flaash_patagonia
;
;   Option C - Python API wrapper:
;     See 01_flaash_correction.py in this same folder.
;
; OUTPUTS:
;   - {output_dir}/flaash_corrected.dat       (ENVI format reflectance)
;   - {output_dir}/flaash_corrected.hdr
;   - {output_dir}/flaash_corrected_rfl.tif   (GeoTIFF, ArcGIS/QGIS ready)
;   - {output_dir}/flaash_cloud_mask.dat
;   - {output_dir}/flaash_water_vapor.dat
;   - {output_dir}/flaash_report.txt
;
; ACADEMIC NOTE:
;   The Sub-Arctic Summer atmosphere is the MODTRAN standard profile that best
;   represents mid-latitude Southern Hemisphere summer conditions. Patagonia at
;   51 deg S has a maritime sub-polar climate, and the Sub-Arctic Summer profile
;   provides the closest match to observed radiosonde profiles for this latitude.
;
;   Maritime aerosol model: Sea salt aerosols dominate the western Patagonian
;   atmosphere due to persistent onshore Pacific winds. This contrasts with
;   Urban/Industrial or Rural aerosol models appropriate for inland scenes.
;
; =============================================================================
PRO flaash_patagonia

  COMPILE_OPT IDL2

  ; -------------------------------------------------------------------------
  ; USER CONFIGURATION - edit these paths before running
  ; -------------------------------------------------------------------------

  ; Input: Landsat L1TP stack (all bands as a single BSQ or BIL file)
  ; If your Landsat bands are separate files, stack them first with ENVI:
  ;   Raster Management > Layer Stacking
  input_file = 'D:\00_AI_Aplications\ClimateChange\Chapter_01\data\raw\landsat_l1tp\landsat9_l1tp_stack.dat'

  ; Output directory for all FLAASH products
  output_dir = 'D:\00_AI_Aplications\ClimateChange\Chapter_01\data\processed\envi_flaash'

  ; Scene center date (format: YYYY-MM-DD)
  scene_date = '2023-01-15'

  ; Sensor type - for Landsat 9 OLI use 'OLI'
  ; Options: 'OLI', 'TM', 'ETM+', 'MSI' (Sentinel-2)
  sensor_type = 'OLI'

  ; -------------------------------------------------------------------------
  ; Create output directory if it does not exist
  ; -------------------------------------------------------------------------
  IF ~FILE_TEST(output_dir, /DIRECTORY) THEN FILE_MKDIR, output_dir

  ; -------------------------------------------------------------------------
  ; Check that input file exists
  ; -------------------------------------------------------------------------
  IF ~FILE_TEST(input_file) THEN BEGIN
    PRINT, 'ERROR: Input file not found: ' + input_file
    PRINT, 'Stack individual Landsat bands first using ENVI Layer Stacking.'
    RETURN
  ENDIF

  ; -------------------------------------------------------------------------
  ; Open ENVI API
  ; -------------------------------------------------------------------------
  PRINT, '=========================================='
  PRINT, ' FLAASH Atmospheric Correction'
  PRINT, ' Torres del Paine, Patagonia, Chile'
  PRINT, '=========================================='

  e = ENVI()

  ; -------------------------------------------------------------------------
  ; Open the input L1TP radiance file
  ; -------------------------------------------------------------------------
  PRINT, 'Opening input file: ' + input_file
  raster = e.OpenRaster(input_file)

  IF OBJ_VALID(raster) EQ 0 THEN BEGIN
    PRINT, 'ERROR: Could not open input raster.'
    RETURN
  ENDIF

  PRINT, 'Input bands   : ' + STRTRIM(raster.NBANDS, 2)
  PRINT, 'Input rows    : ' + STRTRIM(raster.NROWS, 2)
  PRINT, 'Input columns : ' + STRTRIM(raster.NCOLUMNS, 2)
  PRINT, 'Input CRS     : ' + raster.SPATIALREF.COORD_SYS_STR

  ; -------------------------------------------------------------------------
  ; Set output paths
  ; -------------------------------------------------------------------------
  output_rfl     = output_dir + '\flaash_corrected.dat'
  output_rfl_tif = output_dir + '\flaash_corrected_rfl.tif'
  output_cloud   = output_dir + '\flaash_cloud_mask.dat'
  output_water   = output_dir + '\flaash_water_vapor.dat'
  output_report  = output_dir + '\flaash_report.txt'

  ; -------------------------------------------------------------------------
  ; FLAASH Parameters
  ; -------------------------------------------------------------------------
  ; Study scene centre (Torres del Paine approximate centre)
  scene_lat =  -51.0D   ; degrees, negative = Southern Hemisphere
  scene_lon =  -73.0D   ; degrees, negative = West

  ; Parse scene date for FLAASH day-of-year input
  date_parts = STRSPLIT(scene_date, '-', /EXTRACT)
  scene_year = FIX(date_parts[0])
  scene_mon  = FIX(date_parts[1])
  scene_day  = FIX(date_parts[2])

  ; Elevation of study area (mean terrain height, Torres del Paine valley floor)
  ; FLAASH uses this to adjust atmospheric path length
  scene_elevation_km = 0.15D   ; km above sea level

  ; Atmospheric model:
  ;   1 = Tropical, 2 = Mid-Lat Summer, 3 = Mid-Lat Winter
  ;   4 = Sub-Arctic Summer, 5 = Sub-Arctic Winter, 6 = US Standard 1962
  ; Sub-Arctic Summer (4) is best for Patagonian summer at 51 deg S
  atm_model = 4

  ; Aerosol model:
  ;   1 = Rural, 2 = Urban, 3 = Maritime, 4 = Tropospheric
  ; Maritime (3) - dominant aerosol type for windward Pacific coast
  aerosol_model = 3

  ; Initial aerosol visibility (km)
  ; 40 km = typical for clear Patagonian conditions
  ; Reduce to 20-25 km for hazy or smoky scenes
  initial_visibility = 40.0D

  ; Water vapour retrieval
  ; 1 = enabled (recommended), 0 = use fixed value
  water_retrieval = 1

  ; Water absorption wavelength (nm)
  ; 1135 nm feature used for water vapour retrieval in OLI/ETM+ spectra
  water_absorption_feature = 1135

  ; Aerosol retrieval
  ; 1 = 2-band retrieval (recommended for clear scenes)
  ; 0 = fixed initial visibility
  aerosol_retrieval = 1

  ; Multi-scattering mode
  ; 9 = Kaufman-Tanre 2-band retrieval (recommended)
  ; 5 = Trad 2-band with fixed aerosol
  multiscatter_model = 9

  ; Spatial subset for aerosol retrieval (pixels apart)
  ; Larger value = faster, less accurate aerosol map
  ; 10 = compute aerosol at every 10th pixel
  kwargs_aerosol_scale = 10

  ; -------------------------------------------------------------------------
  ; Run FLAASH task
  ; -------------------------------------------------------------------------
  PRINT, ''
  PRINT, 'Running FLAASH atmospheric correction...'
  PRINT, '  Atmospheric model : Sub-Arctic Summer (model 4)'
  PRINT, '  Aerosol model     : Maritime (model 3)'
  PRINT, '  Initial visibility: ' + STRTRIM(initial_visibility, 2) + ' km'
  PRINT, '  Scene latitude    : ' + STRTRIM(scene_lat, 2) + ' deg'
  PRINT, '  Water retrieval   : ' + (water_retrieval EQ 1 ? 'ENABLED' : 'DISABLED')
  PRINT, '  This may take 5-20 minutes depending on scene size...'
  PRINT, ''

  task = ENVITask('SpectralIndices')

  ; FLAASH is accessed via ENVITask('FLAASH')
  ; Note: Task name is 'FLAASH' in ENVI 5.x
  flaash_task = ENVITask('FLAASH')

  flaash_task.INPUT_RASTER                = raster
  flaash_task.OUTPUT_REFLECTANCE_RASTER_URI = output_rfl
  flaash_task.OUTPUT_CLOUD_RASTER_URI     = output_cloud
  flaash_task.OUTPUT_WATER_VAPOR_RASTER_URI = output_water
  flaash_task.OUTPUT_REPORT_URI           = output_report

  ; Scene geometry
  flaash_task.SCENE_CENTER_LAT   = scene_lat
  flaash_task.SCENE_CENTER_LON   = scene_lon
  flaash_task.SENSOR_TYPE        = sensor_type
  flaash_task.FLIGHT_DATE        = JULDAY(scene_mon, scene_day, scene_year)
  flaash_task.SCENE_ELEVATION    = scene_elevation_km

  ; Atmospheric models
  flaash_task.ATMOSPHERIC_MODEL  = atm_model
  flaash_task.AEROSOL_MODEL      = aerosol_model
  flaash_task.INITIAL_VISIBILITY = initial_visibility

  ; Retrieval options
  flaash_task.WATER_RETRIEVAL    = water_retrieval
  flaash_task.WATER_ABSORPTION_FEATURE = water_absorption_feature
  flaash_task.AEROSOL_RETRIEVAL  = aerosol_retrieval
  flaash_task.MULTI_SCATTER_MODEL = multiscatter_model
  flaash_task.AEROSOL_SCALE_FACTOR = kwargs_aerosol_scale

  ; Execute
  flaash_task.Execute

  ; -------------------------------------------------------------------------
  ; Export reflectance result as GeoTIFF for ArcGIS Pro / QGIS
  ; -------------------------------------------------------------------------
  PRINT, 'FLAASH complete. Exporting GeoTIFF...'
  rfl_raster = e.OpenRaster(output_rfl)

  export_task = ENVITask('ExportRasterToFormat')
  export_task.INPUT_RASTER = rfl_raster
  export_task.OUTPUT_URI   = output_rfl_tif
  export_task.DATA_TYPE    = 'FLOAT'
  export_task.Execute

  ; -------------------------------------------------------------------------
  ; Validate output
  ; -------------------------------------------------------------------------
  IF FILE_TEST(output_rfl_tif) THEN BEGIN
    fsize = (FILE_INFO(output_rfl_tif)).SIZE / 1e6
    PRINT, ''
    PRINT, '=========================================='
    PRINT, ' FLAASH CORRECTION COMPLETE'
    PRINT, '=========================================='
    PRINT, '  Reflectance TIF : ' + output_rfl_tif + ' (' + STRTRIM(LONG(fsize),2) + ' MB)'
    PRINT, '  Cloud mask      : ' + output_cloud
    PRINT, '  Water vapor     : ' + output_water
    PRINT, '  Report          : ' + output_report
    PRINT, ''
    PRINT, '  ArcGIS Pro: Add ' + output_rfl_tif + ' as raster layer'
    PRINT, '  Python    : import rasterio; src = rasterio.open("' + output_rfl_tif + '")'
    PRINT, '=========================================='
  ENDIF ELSE BEGIN
    PRINT, 'WARNING: GeoTIFF export may not have completed. Check ' + output_rfl
  ENDELSE

  ; -------------------------------------------------------------------------
  ; Cleanup ENVI API objects
  ; -------------------------------------------------------------------------
  OBJ_DESTROY, raster
  IF OBJ_VALID(rfl_raster) THEN OBJ_DESTROY, rfl_raster
  ; Note: do not close 'e' if running interactively inside ENVI

END
