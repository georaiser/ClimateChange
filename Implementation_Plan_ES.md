# Propuesta de Especialización Maestra: Análisis Integral del Clima, Glaciares y Cuencas

Esta propuesta detalla un plan de implementación académico masivo y unificado que fusiona los objetivos del análisis de Cambio Climático, Retroceso Glaciar y Modelamiento de Cuencas. El plan aprovecha **ArcGIS Pro (ArcHydro), ENVI, RStudio y entornos Python** para crear un pipeline analítico robusto.

Además, para elevar el proyecto a una arquitectura Geo-AI de nivel empresarial, la especialización avanza hacia temas avanzados que incluyen **Orquestación Agéntica (LangGraph), PyTorch Multitarea, Detección de Cambios Multitemporal, PostGIS y desarrollo web Full-Stack**.

El proceso de aprendizaje se divide en 11 **Capítulos** académicos secuenciales, asegurando una comprensión paso a paso de cómo construir y escalar estas complejas herramientas.

---

## Resumen de Recursos y Técnicas
**Recursos de Datos:**
*   **Sensores Ópticos y Termales:** Landsat, Sentinel-2, MODIS.
*   **Radar y Elevación:** Sentinel-1 SAR, DEMs SRTM / ALOS PALSAR.
*   **Datasets Climáticos:** WorldClim, CHIRPS.
*   **Datos In-Situ:** Estaciones meteorológicas locales (ej. CR2 Chile).

**Técnicas Analíticas Avanzadas:**
*   **IA Agéntica:** LangGraph para orquestación geoespacial autónoma.
*   **Deep Learning (PyTorch):** CNNs multitarea para segmentación simultánea de lagos, nieve y detección de cambios.
*   **Detección de Cambios Multitemporal:** Análisis de series temporales a lo largo de décadas utilizando datos SAR y Ópticos fusionados.
*   **Modelamiento Hidrológico (ArcHydro / PySheds):** Delimitación de cuencas y balance hídrico.
*   **Bases de Datos Espaciales:** PostgreSQL + PostGIS para almacenamiento vectorial escalable.
*   **SIG Web Full-Stack:** FastAPI + React + MapLibre GL JS para dashboards interactivos.

## Región de Interés (ROI) Seleccionada
**Parque Nacional Torres del Paine y Punta Arenas (Cuenca del Río Grey), Región de Magallanes, Chile.**

---

## Estructura Detallada del Proyecto y Flujo Académico

### CAPÍTULO 1: Variables Climáticas y Procesamiento de Imágenes
**Objetivo Académico:** Base en adquisición de datos, corrección atmosférica y modelado climático.

#### `Chapter_01/`
*   **`01_stac_multisensor_download.py`:** (Concepto: Adquisición programática). Utiliza la API de Copernicus y AWS STAC para descargar datos multisensores.
*   **`02_atmospheric_correction.py`:** (Concepto: Física de reflectancia TOA a BOA). Corrección radiométrica/atmosférica usando Python (`rasterio`/`Py6S`) y ENVI.
*   **`03_station_ml_interpolation.py`:** (Concepto: ML para datos faltantes). Entrenamiento de Random Forest con datos de estaciones meteorológicas para predecir superficies climáticas continuas.
*   **`04_precipitation_dual_analysis`:** (Concepto: Análisis espacial R vs Python). Anomalías históricas CHIRPS y clasificación de zonas climáticas con ArcGIS IsoCluster.
*   **`05_uhi_modis_mapping.py`:** (Concepto: Microclimas urbanos). Mapeo de Isla de Calor Urbana sobre Punta Arenas usando datos termales.

---

### CAPÍTULO 2: Retroceso Glaciar, Dinámica SAR e IA
**Objetivo Académico:** Seguimiento de la pérdida de hielo mediante enfoques multisensor (Radar/Óptico) y Deep Learning fundamental.

#### `Chapter_02/`
*   **`01_glacier_area_perimeter.py`:** (Concepto: Detección de cambios SIG vectorial). Script ArcGIS Pro/Python para calcular el retroceso histórico del Glaciar Grey.
*   **`02_sentinel1_sar_analysis.py`:** (Concepto: Física de radar penetrante de nubes). Clasificación de nieve seca, nieve húmeda y hielo mediante retrodispersión SAR.
*   **`03_xarray_ndwi_multitemporal.py`:** (Concepto: Índices espectrales). Mapeo de la expansión del Lago Grey a lo largo de décadas.
*   **`04_pytorch_lake_segmentation.py`:** (Concepto: Segmentación semántica). CNN U-Net para extracción automatizada de lagos glaciares.
*   **`05_arcpy_ndsi_snow_cover.py`:** (Concepto: Seguimiento de nieve). Script automatizado de ArcGIS Pro para mapeo NDSI.
*   **`06_multitemporal_change_detection.py`:** (Concepto: Fusión de sensores multitemporal). Fusión de series temporales Sentinel-1 SAR y Sentinel-2 Óptico para detectar algorítmicamente cambios en la cobertura terrestre y eventos de desprendimiento glaciar entre dos fechas (T1 vs T2).

---

### CAPÍTULO 3: Hidrología y Modelamiento de Cuencas
**Objetivo Académico:** Comprender la mecánica del flujo de agua, modelado del terreno y gestión de cuencas.

#### `Chapter_03/`
*   **`01_dem_hydro_conditioning.py`:** (Concepto: Corrección del terreno). Relleno de sumideros y aplicación de correcciones TIN a DEMs.
*   **`02_archydro_basin_delineation.py`:** (Concepto: Enrutamiento de flujo automatizado). Uso de ArcHydro y PySheds para delimitar la Cuenca del Río Grey.
*   **`03_morphometric_parameters.py`:** (Concepto: Rasgos físicos de la cuenca). Cálculo del factor de forma, pendiente media y Curva Hipsométrica.
*   **`04_drainage_and_strahler.py`:** (Concepto: Jerarquía de corrientes). Cálculo de densidad de drenaje y orden de corrientes de Strahler.
*   **`05_water_balance_isohyets.py`:** (Concepto: Entradas/salidas de hidrología espacial). Generación de Isoyetas e Isotermas para un modelo espacial de Balance Hídrico.

---

### CAPÍTULO 4: Vulnerabilidad e Impactos en los Ecosistemas
**Objetivo Académico:** Comprender las consecuencias biológicas y humanas de los cambios físicos.

#### `Chapter_04/`
*   **`01_maxent_modeling` (R/Python):** (Concepto: Modelado de Distribución de Especies). Mapeo de cómo migrará la flora/fauna a medida que cambia el clima de la cuenca.
*   **`02_vulnerability_index_mce.py`:** (Concepto: Evaluación Multicriterio). Combinar precipitación, retroceso glaciar y riesgos de inundación en un "Mapa de Calor de Vulnerabilidad Climática".

---

### CAPÍTULO 5: Cartografía y Planes de Manejo
**Objetivo Académico:** Comunicar datos espaciales claramente a través de salidas SIG estándar.

#### `Chapter_05/`
*   **`01_map_automation_layout.py`:** (Concepto: Automatización cartográfica). Script `arcpy.mp` generando mapas PDF estandarizados.
*   **`02_watershed_management_report.py`:** (Concepto: Reportes). Compilación de estadísticas, balance hídrico e índices de vulnerabilidad en un plan de manejo final.

---

### CAPÍTULO 6: Proyecto Capstone - El Pipeline Unificado Lineal
**Objetivo Académico:** Sintetizar módulos anteriores en un único código automatizado.

#### `Chapter_06/`
*   **`main_linear_pipeline.py`:** (Concepto: Integración de sistemas). Un script masivo en Python que activa linealmente descargas, preprocesamiento, inferencia de IA, delimitación de cuencas y generación de reportes en un bucle de ejecución continuo.

---

## ARQUITECTURAS AVANZADAS

### CAPÍTULO 7: Orquestación Agéntica (LangGraph)
**Objetivo Académico:** Pasar del scripting lineal y rígido a agentes de IA autónomos impulsados por máquinas de estado.

#### `Chapter_07/`
*   **Concepto:** En lugar de un script que se ejecuta de arriba a abajo, construimos agentes de IA especializados que "deciden" cuándo actuar según el estado de los datos.
*   **Acción:** Construir una red LangGraph con un `AcquisitionAgent`, `GlacierAgent`, `HydrologyAgent` y `ReportAgent`. Pasan de forma autónoma cargas de datos (como GeoTIFFs) entre sí, manejando errores y reintentos de forma inteligente.

### CAPÍTULO 8: El Efecto Cascada
**Objetivo Académico:** Pasar de métricas aisladas a modelar una reacción en cadena física.

#### `Chapter_08/`
*   **Concepto:** Los desastres no ocurren de forma aislada. Vincularemos programáticamente los modelos para demostrar causa y efecto.
*   **Acción:** Un framework de Python que rastrea específicamente la causalidad estadística: *Anomalía de Temperatura (Ch1) → Expansión del Lago (Ch2) → Balance Hídrico Alterado (Ch3) → Cambio de Nicho de Alta Vulnerabilidad (Ch4).*

### CAPÍTULO 9: Modelo PyTorch Multitarea
**Objetivo Académico:** Actualizar el Deep Learning estándar a arquitecturas avanzadas multicabezal.

#### `Chapter_09/`
*   **Concepto:** Ejecutar múltiples redes neuronales separadas es computacionalmente costoso. El aprendizaje multitarea predice varias cosas a la vez.
*   **Acción:** Actualizar la U-Net del Capítulo 2 a una **CNN Multicabezal**. Un único modelo PyTorch que ingiere tensores Ópticos, SAR y DEM simultáneamente para generar máscaras en una sola pasada: 
    *   (1) Segmentación del Lago Glaciar
    *   (2) Cobertura de Nieve/Hielo
    *   (3) Zonas de Riesgo de Inundación
    *   **(4) Cabezal de Detección de Cambios Multitemporal:** Puntuación de anomalías que resalta los cambios estructurales en el terreno entre dos años diferentes.

### CAPÍTULO 10: Integración de Base de Datos Espacial (PostGIS)
**Objetivo Académico:** Pasar de archivos planos (Shapefiles/GeoTIFFs) a bases de datos espaciales empresariales.

#### `Chapter_10/`
*   **Concepto:** Los archivos planos son difíciles de consultar a lo largo del tiempo. Las bases de datos permiten consultas SQL espaciales masivas.
*   **Acción:** Configurar una base de datos PostgreSQL + PostGIS a través de Docker. Escribir scripts en Python (`SQLAlchemy`, `GeoAlchemy2`) para ingerir los polígonos de salida (lagos, cuencas, zonas de vulnerabilidad) y metadatos ráster directamente en la base de datos.

### CAPÍTULO 11: Dashboard Interactivo (SIG Full-Stack)
**Objetivo Académico:** Pasar de mapas PDF estáticos a SIG Web interactivo en tiempo real.

#### `Chapter_11/`
*   **Concepto:** Las partes interesadas necesitan plataformas interactivas, no solo PDFs.
*   **Acción:** Construir un backend en **FastAPI** que consulte la base de datos PostGIS, y un frontend en **React + MapLibre GL JS** para visualizar la Cascada Climática, Retroceso Glaciar y cambios en la Cuenca de forma dinámica a través del navegador web.
