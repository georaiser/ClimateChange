; =============================================================================
; GeoCascade Chapter 14 -- ENVI + IDL Native Reference Script
; 11_idl_reference.pro
;
; PURPOSE:
;   Native IDL/ENVI scripting reference for all Chapter 14 workflows.
;   This script uses the ENVI Classic API (IDL-based) and the modern
;   ENVI Task Engine API.
;
; STUDY AREA: Torres del Paine, Patagonia, Chile
; BBOX WGS84: [-73.5, -51.5, -72.5, -50.5]
;
; HOW TO RUN:
;   Option 1 — ENVI IDL Console:
;     IDL> .compile 'D:\00_AI_Aplications\ClimateChange\Chapter_14\11_idl_reference.pro'
;     IDL> geocascade_run_all
;
;   Option 2 — IDL Workbench:
;     Open this file in IDL Workbench (Ctrl+O)
;     Run > Run File
;
;   Option 3 — Command Line:
;     idl -e "geocascade_run_all" -args "D:\00_AI_Aplications\ClimateChange"
;
; REQUIREMENTS:
;   ENVI 5.6+  with IDL 8.9+
;   Spatial Analyst module (for classification)
;   Atmospheric Correction Module (for FLAASH)
;
; OUTPUTS:
;   Chapter_14\data\processed\envi_outputs\idl_ndvi.tif
;   Chapter_14\data\processed\envi_outputs\idl_classified.tif
;   Chapter_14\data\processed\envi_outputs\idl_change.tif
;   Chapter_14\data\processed\envi_outputs\idl_dos1_corrected.tif
;
; ARCGIS PRO NOTE:
;   All .tif outputs are GeoTIFF-compatible -- add directly as raster layers.
; =============================================================================


; =============================================================================
; PROCEDURE: geocascade_setup
;   Initialise paths and start ENVI without display (headless mode).
; =============================================================================
PRO geocascade_setup, base_dir, envi_out_dir
  COMPILE_OPT IDL2

  ; Detect base directory from script location if not provided
  IF N_ELEMENTS(base_dir) EQ 0 THEN BEGIN
    base_dir = FILE_DIRNAME(ROUTINE_FILEPATH()) + PATH_SEP() + '..'
    base_dir = FILE_EXPAND_PATH(base_dir)
  ENDIF

  envi_out_dir = base_dir + PATH_SEP() + 'Chapter_14' + PATH_SEP() + $
                 'data' + PATH_SEP() + 'processed' + PATH_SEP() + 'envi_outputs'

  ; Create output directory
  IF ~FILE_TEST(envi_out_dir, /DIRECTORY) THEN $
    FILE_MKDIR, envi_out_dir

  PRINT, '============================================================'
  PRINT, ' GeoCascade Ch14 -- ENVI + IDL Native Scripting'
  PRINT, '============================================================'
  PRINT, ' Base dir  : ' + base_dir
  PRINT, ' Output dir: ' + envi_out_dir

  ; Start ENVI without display (headless / batch mode)
  ; Comment out HEADLESS=1 if you want the GUI to appear
  e = ENVI(/HEADLESS)
  PRINT, ' ENVI started (headless). Version: ' + e.VERSION

END


; =============================================================================
; PROCEDURE: geocascade_spectral_analysis
;   Compute NDVI, NBR, NDWI, NDSI from Sentinel-2 L2A imagery using
;   ENVI Task Engine (modern API).
;
;   ENVI GUI equivalent:
;     Toolbox > Spectral > Vegetation Index Calculator
;     OR: Toolbox > Band Math
;         Expression: (float(b4)-float(b3))/(float(b4)+float(b3))
; =============================================================================
PRO geocascade_spectral_analysis, base_dir, envi_out_dir
  COMPILE_OPT IDL2

  PRINT, ''
  PRINT, '[1/4] Spectral Index Analysis (NDVI, NBR, NDWI, NDSI)'

  e = ENVI(/CURRENT)

  ; -- Locate Sentinel-2 input image ------------------------------------------
  ; Check for real imagery from Chapter 02
  s2_search = FILE_SEARCH(base_dir + PATH_SEP() + 'Chapter_02' + $
                           PATH_SEP() + '**' + PATH_SEP() + '*.tif', $
                           COUNT=n_found)

  IF n_found GT 0 THEN BEGIN
    in_raster_path = s2_search[0]
    PRINT, '  Input image: ' + in_raster_path
    raster = e.OpenRaster(in_raster_path)
  ENDIF ELSE BEGIN
    PRINT, '  [NOTE] No Sentinel-2 TIF found in Chapter_02.'
    PRINT, '         Using envi_outputs/sentinel2_dos1_corrected.tif if available.'
    dos1_path = envi_out_dir + PATH_SEP() + 'sentinel2_dos1_corrected.tif'
    IF FILE_TEST(dos1_path) THEN BEGIN
      raster = e.OpenRaster(dos1_path)
      PRINT, '  Using DOS1-corrected image: ' + dos1_path
    ENDIF ELSE BEGIN
      PRINT, '  [SKIP] No input image found. Run 07_envi_spectral_analysis.py first.'
      RETURN
    ENDELSE
  ENDELSE

  ; -- Compute Vegetation Index using ENVI Task Engine ------------------------
  ; ENVI Task: VegetationIndex
  ; Sentinel-2 band order: B2=1,B3=2,B4=3,B8=4,B11=5,B12=6 (1-indexed)

  ; NDVI = (NIR - Red) / (NIR + Red)  -> B8(4) and B4(3)
  task_ndvi = ENVITask('VegetationIndex')
  task_ndvi['INPUT_RASTER']     = raster
  task_ndvi['INDEX']            = 'Normalized Difference Vegetation Index'
  task_ndvi['OUTPUT_RASTER_URI'] = envi_out_dir + PATH_SEP() + 'idl_ndvi.tif'
  task_ndvi.Execute

  ; NBR = (NIR - SWIR2) / (NIR + SWIR2)  -> B8(4) and B12(6)
  ; ENVI Band Math for custom index
  task_bm = ENVITask('BandMath')
  task_bm['INPUT_RASTERS']      = LIST(raster)
  task_bm['EXPRESSION']         = '(float(b4)-float(b6))/(float(b4)+float(b6))'
  task_bm['OUTPUT_RASTER_URI']  = envi_out_dir + PATH_SEP() + 'idl_nbr.tif'
  task_bm.Execute
  PRINT, '  NBR computed: ' + task_bm['OUTPUT_RASTER_URI']

  ; NDWI = (Green - NIR) / (Green + NIR) -> B3(2) and B8(4)
  task_ndwi = ENVITask('BandMath')
  task_ndwi['INPUT_RASTERS']     = LIST(raster)
  task_ndwi['EXPRESSION']        = '(float(b2)-float(b4))/(float(b2)+float(b4))'
  task_ndwi['OUTPUT_RASTER_URI'] = envi_out_dir + PATH_SEP() + 'idl_ndwi.tif'
  task_ndwi.Execute
  PRINT, '  NDWI computed: ' + task_ndwi['OUTPUT_RASTER_URI']

  ; NDSI = (Green - SWIR1) / (Green + SWIR1) -> B3(2) and B11(5)
  task_ndsi = ENVITask('BandMath')
  task_ndsi['INPUT_RASTERS']     = LIST(raster)
  task_ndsi['EXPRESSION']        = '(float(b2)-float(b5))/(float(b2)+float(b5))'
  task_ndsi['OUTPUT_RASTER_URI'] = envi_out_dir + PATH_SEP() + 'idl_ndsi.tif'
  task_ndsi.Execute
  PRINT, '  NDSI computed: ' + task_ndsi['OUTPUT_RASTER_URI']

  ; Print NDVI statistics
  ndvi_raster = e.OpenRaster(task_ndvi['OUTPUT_RASTER_URI'])
  stats = ENVITask('StatisticsRaster')
  stats['INPUT_RASTER'] = ndvi_raster
  stats.Execute
  PRINT, '  NDVI statistics:'
  PRINT, '    Min  : ' + STRTRIM(STRING(stats['MIN_VALUES'][0], FORMAT='(F8.4)'), 2)
  PRINT, '    Max  : ' + STRTRIM(STRING(stats['MAX_VALUES'][0], FORMAT='(F8.4)'), 2)
  PRINT, '    Mean : ' + STRTRIM(STRING(stats['MEAN_VALUES'][0], FORMAT='(F8.4)'), 2)

  PRINT, '  [OK] Spectral indices saved to: ' + envi_out_dir

END


; =============================================================================
; PROCEDURE: geocascade_atm_correction
;   FLAASH atmospheric correction for Sentinel-2 using ENVI Task Engine.
;   Falls back to QUAC if FLAASH module not licensed.
;
;   ENVI GUI equivalent:
;     Toolbox > Radiometric Correction > Atmospheric Correction Module > FLAASH
;
;   FLAASH parameters for Torres del Paine, Patagonia:
;     Atmospheric Model : Sub-Arctic Summer (lat -51 deg)
;     Aerosol Model     : Maritime
;     Initial Visibility: 40 km
;     Ground Elevation  : 0.5 km
; =============================================================================
PRO geocascade_atm_correction, base_dir, envi_out_dir
  COMPILE_OPT IDL2

  PRINT, ''
  PRINT, '[2/4] Atmospheric Correction (FLAASH / QUAC)'

  e = ENVI(/CURRENT)

  ; Locate input image
  in_path = envi_out_dir + PATH_SEP() + 'sentinel2_dos1_corrected.tif'
  IF ~FILE_TEST(in_path) THEN BEGIN
    ; Try Chapter 02 output
    found = FILE_SEARCH(base_dir + PATH_SEP() + 'Chapter_02' + $
                         PATH_SEP() + '**' + PATH_SEP() + '*L2A*.tif', $
                         COUNT=n)
    IF n GT 0 THEN in_path = found[0] $
    ELSE BEGIN
      PRINT, '  [SKIP] No input image for atmospheric correction.'
      RETURN
    ENDELSE
  ENDIF

  raster = e.OpenRaster(in_path)
  PRINT, '  Input: ' + in_path

  ; ---- Attempt FLAASH (requires Atmospheric Correction Module license) ------
  flaash_ok = 0
  CATCH, flaash_err
  IF flaash_err NE 0 THEN BEGIN
    PRINT, '  [WARN] FLAASH not licensed -- falling back to QUAC'
    CATCH, /CANCEL
    flaash_ok = 0
  ENDIF ELSE BEGIN
    ; FLAASH Task
    task_flaash = ENVITask('FLAASH')
    task_flaash['INPUT_RASTER']           = raster
    task_flaash['SENSOR_TYPE']            = 'Sentinel-2A'
    task_flaash['FLIGHT_DATE']            = '2022-01-15'
    task_flaash['FLIGHT_TIME_UTC']        = '14:30:00'
    task_flaash['SCENE_CENTER_LAT']       = -51.0
    task_flaash['SCENE_CENTER_LON']       = -72.9
    task_flaash['INITIAL_VISIBILITY']     = 40.0     ; km
    task_flaash['GROUND_ELEVATION']       = 0.5      ; km
    task_flaash['ATMOSPHERIC_MODEL']      = 5        ; Sub-Arctic Summer
    task_flaash['AEROSOL_MODEL']          = 3        ; Maritime
    task_flaash['WATER_RETRIEVAL']        = 1        ; enable
    task_flaash['OUTPUT_RASTER_URI']      = envi_out_dir + PATH_SEP() + 'idl_flaash.tif'
    task_flaash.Execute
    PRINT, '  [OK] FLAASH corrected: ' + task_flaash['OUTPUT_RASTER_URI']
    flaash_ok = 1
  ENDELSE

  ; ---- QUAC fallback (Quick Atmospheric Correction, no license needed) ------
  IF ~flaash_ok THEN BEGIN
    task_quac = ENVITask('QUACCorrection')
    task_quac['INPUT_RASTER']     = raster
    task_quac['OUTPUT_RASTER_URI'] = envi_out_dir + PATH_SEP() + 'idl_quac.tif'
    task_quac.Execute
    PRINT, '  [OK] QUAC corrected: ' + task_quac['OUTPUT_RASTER_URI']
  ENDIF

END


; =============================================================================
; PROCEDURE: geocascade_classify
;   Unsupervised ISODATA classification + post-classification Majority Filter.
;
;   ENVI GUI equivalent:
;     Classification > Unsupervised > ISODATA
;     Classification > Post Classification > Majority Analysis
; =============================================================================
PRO geocascade_classify, base_dir, envi_out_dir
  COMPILE_OPT IDL2

  PRINT, ''
  PRINT, '[3/4] Land Cover Classification (ISODATA + MLC + Majority)'

  e = ENVI(/CURRENT)

  ; Use NDVI output as input to classification
  ndvi_path = envi_out_dir + PATH_SEP() + 'idl_ndvi.tif'
  IF ~FILE_TEST(ndvi_path) THEN BEGIN
    ; Try open-source Python output
    ndvi_path = envi_out_dir + PATH_SEP() + 'ndvi.tif'
    IF ~FILE_TEST(ndvi_path) THEN BEGIN
      PRINT, '  [SKIP] NDVI raster not found. Run spectral analysis first.'
      RETURN
    ENDIF
  ENDIF

  raster = e.OpenRaster(ndvi_path)
  PRINT, '  Input NDVI: ' + ndvi_path

  ; ---- ISODATA Unsupervised Classification ----------------------------------
  task_iso = ENVITask('ISODATAClassification')
  task_iso['INPUT_RASTER']          = raster
  task_iso['NUMBER_OF_CLASSES']     = 5
  task_iso['ITERATIONS']            = 20
  task_iso['MINIMUM_CLASS_SIZE']    = 10
  task_iso['CHANGE_THRESHOLD']      = 5.0   ; % pixels that must change to continue
  task_iso['SPLIT_THRESHOLD']       = 1.0
  task_iso['MERGE_THRESHOLD']       = 0.5
  task_iso['OUTPUT_RASTER_URI']     = envi_out_dir + PATH_SEP() + 'idl_isodata.tif'
  task_iso['OUTPUT_REPORT_URI']     = envi_out_dir + PATH_SEP() + 'idl_isodata_report.txt'
  task_iso.Execute
  PRINT, '  [OK] ISODATA classified: ' + task_iso['OUTPUT_RASTER_URI']

  ; Print class statistics
  cls_raster = e.OpenRaster(task_iso['OUTPUT_RASTER_URI'])
  PRINT, '  ISODATA converged. Classes: 5'
  PRINT, '  (Rename classes in ENVI: View > Class Legend > right-click > Edit)'

  ; ---- Majority Filter (post-classification smoothing) ----------------------
  task_maj = ENVITask('ClassificationAggregation')
  task_maj['INPUT_RASTER']     = cls_raster
  task_maj['KERNEL_SIZE']      = 3         ; 3x3 window
  task_maj['OUTPUT_RASTER_URI'] = envi_out_dir + PATH_SEP() + 'idl_majority.tif'
  task_maj.Execute
  PRINT, '  [OK] Majority filter applied: ' + task_maj['OUTPUT_RASTER_URI']

  ; ---- Class Statistics -----------------------------------------------------
  stat_task = ENVITask('StatisticsRaster')
  stat_task['INPUT_RASTER'] = e.OpenRaster(task_maj['OUTPUT_RASTER_URI'])
  stat_task.Execute
  PRINT, '  Post-filter class statistics:'
  FOR c = 0, 4 DO BEGIN
    PRINT, '    Class ' + STRTRIM(c+1, 2) + ': ' + $
           STRTRIM(STRING(stat_task['MIN_VALUES'][c], FORMAT='(F8.0)'), 2) + $
           ' - ' + STRTRIM(STRING(stat_task['MAX_VALUES'][c], FORMAT='(F8.0)'), 2)
  ENDFOR

END


; =============================================================================
; PROCEDURE: geocascade_change_detection
;   Multi-date change detection using ENVI Change Detection workflow.
;   Compares NDVI 2019 vs NDVI 2023.
;
;   ENVI GUI equivalent:
;     Toolbox > Change Detection > Image Change Workflow
;       Step 1: Select before/after images
;       Step 2: Define change threshold
;       Step 3: Export binary and directional change maps
; =============================================================================
PRO geocascade_change_detection, envi_out_dir
  COMPILE_OPT IDL2

  PRINT, ''
  PRINT, '[4/4] Change Detection (NDVI 2019 vs 2023)'

  e = ENVI(/CURRENT)

  ; Input rasters (from script 10 or IDL spectral analysis)
  path_2019 = envi_out_dir + PATH_SEP() + 'ndvi_2019.tif'
  path_2023 = envi_out_dir + PATH_SEP() + 'ndvi_2023.tif'

  IF ~FILE_TEST(path_2019) OR ~FILE_TEST(path_2023) THEN BEGIN
    PRINT, '  [SKIP] ndvi_2019.tif or ndvi_2023.tif not found.'
    PRINT, '         Run 10_envi_change_detection.py first to generate inputs.'
    RETURN
  ENDIF

  raster_2019 = e.OpenRaster(path_2019)
  raster_2023 = e.OpenRaster(path_2023)
  PRINT, '  Before: ' + path_2019
  PRINT, '  After : ' + path_2023

  ; ---- Image Difference Change Detection ------------------------------------
  ; ENVI Task: ImageBandDifference
  task_diff = ENVITask('ImageBandDifference')
  task_diff['INPUT_RASTER1']     = raster_2019
  task_diff['INPUT_RASTER2']     = raster_2023
  task_diff['OUTPUT_RASTER_URI'] = envi_out_dir + PATH_SEP() + 'idl_ndvi_diff.tif'
  task_diff.Execute
  PRINT, '  [OK] NDVI difference computed: ' + task_diff['OUTPUT_RASTER_URI']

  ; ---- Threshold Change Classification  -------------------------------------
  ; ENVI Band Math: classify as Loss (-1), NoChange (0), Gain (1)
  diff_raster = e.OpenRaster(task_diff['OUTPUT_RASTER_URI'])
  task_cls = ENVITask('BandMath')
  task_cls['INPUT_RASTERS']      = LIST(diff_raster)
  ; IDL ternary: loss=0, nochange=1, gain=2
  task_cls['EXPRESSION']         = '(b1 LT -0.10) * 0 + ' + $
                                    '(b1 GE -0.10 AND b1 LE 0.10) * 1 + ' + $
                                    '(b1 GT 0.10) * 2'
  task_cls['OUTPUT_RASTER_URI']  = envi_out_dir + PATH_SEP() + 'idl_change_class.tif'
  task_cls.Execute
  PRINT, '  [OK] Change classified: 0=Loss, 1=NoChange, 2=Gain'
  PRINT, '       Output: ' + task_cls['OUTPUT_RASTER_URI']

  ; ---- Change Area Statistics -----------------------------------------------
  ; Count pixels per class
  diff_raster = e.OpenRaster(task_cls['OUTPUT_RASTER_URI'])
  data = diff_raster.GetData()      ; reads full raster as array
  pixel_area_km2 = 0.0111 * 0.0111  ; ~1.1km pixels at -51 deg lat (0.01 deg)

  loss_n    = TOTAL(data EQ 0)
  nochange_n = TOTAL(data EQ 1)
  gain_n    = TOTAL(data EQ 2)
  total_n   = N_ELEMENTS(data)

  PRINT, ''
  PRINT, '  Change Area Summary:'
  PRINT, '  -------------------------------------------'
  PRINT, '  Vegetation Loss  : ' + STRTRIM(LONG(loss_n), 2) + ' px  ' + $
         STRTRIM(STRING(loss_n * pixel_area_km2, FORMAT='(F8.1)'), 2) + ' km2  ' + $
         STRTRIM(STRING(loss_n/total_n*100, FORMAT='(F5.1)'), 2) + '%'
  PRINT, '  No Change        : ' + STRTRIM(LONG(nochange_n), 2) + ' px  ' + $
         STRTRIM(STRING(nochange_n * pixel_area_km2, FORMAT='(F8.1)'), 2) + ' km2  ' + $
         STRTRIM(STRING(nochange_n/total_n*100, FORMAT='(F5.1)'), 2) + '%'
  PRINT, '  Vegetation Gain  : ' + STRTRIM(LONG(gain_n), 2) + ' px  ' + $
         STRTRIM(STRING(gain_n * pixel_area_km2, FORMAT='(F8.1)'), 2) + ' km2  ' + $
         STRTRIM(STRING(gain_n/total_n*100, FORMAT='(F5.1)'), 2) + '%'

  PRINT, ''
  PRINT, '  ArcGIS Pro: Add idl_change_class.tif -> Symbology > Unique Values'
  PRINT, '    Class 0 = Red (Loss), Class 1 = Grey (No Change), Class 2 = Green (Gain)'

END


; =============================================================================
; PROCEDURE: geocascade_run_all
;   Master entry point -- runs the complete Chapter 14 IDL pipeline.
; =============================================================================
PRO geocascade_run_all, base_dir
  COMPILE_OPT IDL2

  ; Default base dir
  IF N_ELEMENTS(base_dir) EQ 0 THEN $
    base_dir = 'D:\00_AI_Aplications\ClimateChange'

  ; Initialise
  geocascade_setup, base_dir, envi_out_dir

  ; Run all procedures
  geocascade_spectral_analysis, base_dir, envi_out_dir
  geocascade_atm_correction, base_dir, envi_out_dir
  geocascade_classify, base_dir, envi_out_dir
  geocascade_change_detection, envi_out_dir

  ; Final summary
  PRINT, ''
  PRINT, '============================================================'
  PRINT, ' GeoCascade IDL Pipeline COMPLETE'
  PRINT, '============================================================'
  PRINT, '  All outputs: ' + envi_out_dir
  PRINT, '  GeoTIFFs are ArcGIS Pro compatible -- add as raster layers.'
  PRINT, ''
  PRINT, '  Next: python Chapter_14/12_arcpy_full.py  (ArcGIS Pro pipeline)'
  PRINT, '============================================================'

  ; Close ENVI
  e = ENVI(/CURRENT)
  e.Close

END


; =============================================================================
; QUICK REFERENCE -- Common ENVI IDL Commands
; =============================================================================
;
; Open raster:
;   e = ENVI(/HEADLESS)
;   raster = e.OpenRaster('path/to/file.tif')
;
; Band math:
;   task = ENVITask('BandMath')
;   task['INPUT_RASTERS'] = LIST(raster)
;   task['EXPRESSION'] = '(float(b4)-float(b3))/(float(b4)+float(b3))'
;   task.Execute
;
; Classification:
;   task = ENVITask('ISODATAClassification')
;   task['INPUT_RASTER'] = raster
;   task['NUMBER_OF_CLASSES'] = 5
;   task.Execute
;
; Save raster:
;   out_raster = task['OUTPUT_RASTER']
;   out_raster.Export, 'output.tif', 'TIFF'
;
; ENVI Task list (all available tasks):
;   IDL> tasks = ENVITask.list()
;   IDL> PRINT, tasks
; =============================================================================
