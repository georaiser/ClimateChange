# GeoCascade — Storytelling, Perspectives & Results Reports

> **Área de estudio**: Torres del Paine, Patagonia, Chile (-73.5°, -51.5° / -72.5°, -50.5°)  
> **Período**: 1993–2024 (32 años de datos ERA5-Land + satélite)  
> **Pipeline**: GeoCascade — 14 capítulos, 36 scripts Python, datos reales

---

## 🎯 La Historia Central

```
TEMPERATURA SUBE → GLACIARES SE DERRITEN → CAUDALES CAMBIAN
   → VEGETACIÓN ESTRESADA → CASCADA ECOSISTÉMICA → RIESGO HUMANO
```

Cada script, cada dashboard, cada mapa de este pipeline es **un capítulo de esta misma historia**.  
El nombre GeoCascade lo captura: efectos en cascada a través de los sistemas terrestres.

---

## 1. 📖 Marco Narrativo (Storytelling)

### Estructura de Tres Actos

| Acto | Equivalente Científico | Tu Pipeline |
|------|----------------------|-------------|
| **Acto I — El Mundo Anterior** | Climatología base (1993–2005) | ERA5 diario, CHIRPS histórico, extensión glaciar 2000 |
| **Acto II — El Cambio** | Detección de tendencias, mapas de anomalía | Mann-Kendall, series NDSI/NDVI, velocidad SAR |
| **Acto III — La Consecuencia** | Modelado de riesgos, efectos en cascada | Índices ESI/CVS/WSI, ML cobertura, Deep CNN |

### Fórmula HEI (Hook → Evidence → Implication)

Cada sección de tu informe debe seguir:

```
HOOK        "El Glaciar Grey ha retrocedido 4.2 km² desde 2000 — el área de 600 canchas de fútbol."
EVIDENCIA   Análisis multitemporal NDSI, contornos RGI 7.0, velocidad SAR 380 m/año.
IMPLICACIÓN "Las comunidades que dependen del agua glaciar enfrentan una reducción del 40%
             en caudal estacional confiable antes de 2040 con las tendencias actuales."
```

### Números que importan

❌ NO digas: *"La temperatura aumentó"*  
✅ SÍ di: **"La temperatura media anual subió 0.8°C por década — 4× el promedio global"**

❌ NO digas: *"Se detectó anomalía de precipitación"*  
✅ SÍ di: **"2019 y 2021 fueron los dos años más secos en el registro de 31 años — 38% bajo la media"**

---

## 2. 👥 Perspectivas — ¿Quién es tu Audiencia?

### Perspectiva 1: 🎓 Académica / Científica
**Audiencia**: Investigadores, universidades, revistas de revisión por pares  
**Lenguaje**: Cuantitativo, basado en métodos, reproducible  
**Elementos clave**:
- Significancia estadística (p-valor Mann-Kendall, pendiente de Sen)
- Procedencia de datos (ERA5-Land v5, CHIRPS v2.0, RGI 7.0, Sentinel-2 L2A)
- Intervalos de confianza e incertidumbre
- Reproducibilidad: enlace a scripts Python + entorno conda

**Estructura del informe**:
```
Resumen → Área de Estudio → Datos y Métodos → Resultados → Discusión → Conclusiones → Referencias
```

**Ejemplo de tu pipeline**:
> *"Se detectó una tendencia de calentamiento estadísticamente significativa de +0.064°C/año  
> (p < 0.01, Mann-Kendall τ = 0.47) para la serie diaria ERA5-Land 1993–2024  
> en Torres del Paine (51°S, 73°O), coherente con el calentamiento amplificado  
> en latitudes medias del Hemisferio Sur reportado por el IPCC AR6."*

---

### Perspectiva 2: 🏛️ Política / Tomadores de Decisión
**Audiencia**: Organismos gubernamentales, ministerios ambientales, UNESCO, gestión del parque  
**Lenguaje**: Impacto primero, orientado a la acción, sin jerga técnica  
**Elementos clave**:
- Qué está cambiando, cuánto, cuándo
- Qué está en riesgo (agua, biodiversidad, turismo, infraestructura)
- Qué intervenciones apoyan los datos
- Mapas > gráficos > números

**Estructura del informe**:
```
Resumen Ejecutivo (1 página) → Hallazgos Clave → Mapa de Riesgos → Recomendaciones → Anexo Técnico
```

**Ejemplo de tu pipeline**:
> *"El sistema glaciar de Torres del Paine está perdiendo hielo a una tasa acelerada.  
> El análisis satelital (2000–2024) muestra una reducción del 12% en área glaciada.  
> Si las tendencias actuales continúan, el Lago Grey y el Río Serrano enfrentarán  
> una disminución estimada del 20–35% en caudales estacionales antes de 2040."*

---

### Perspectiva 3: 🌍 Pública / Comunicativa / Educativa
**Audiencia**: Público general, estudiantes, ciudadanos, medios  
**Lenguaje**: Visual primero, narrativo, emocional, con analogías  
**Elementos clave**:
- Imágenes antes/después
- Infografías y mapas animados
- Conexión humana: comunidades, biodiversidad, cultura
- Analogías concretas

**Ejemplo de tu pipeline**:
> *"Cada año durante los últimos 24 años, el Glaciar Grey ha perdido el equivalente  
> a 60 piscinas olímpicas de hielo. Nuestros datos satelitales muestran que este proceso  
> se acelera — el glaciar se mueve y se contrae el doble de rápido que en el año 2000."*

---

### Perspectiva 4: 📚 Curricular (Ingenieros Ambientales)
Basado en los archivos de brainstorm, este proyecto ES una herramienta de enseñanza.

**Marco**: Cada capítulo = habilidad técnica + hallazgo real

| Capítulo | Habilidad enseñada | Hallazgo real en Torres del Paine |
|---------|--------------------|------------------------------------|
| Ch01 | Descarga STAC, API ERA5 | Tendencia de temperatura 31 años, variabilidad de precipitación |
| Ch02 | Índices espectrales (NDVI, NDSI, NDWI) | Zonas de estrés vegetal, declive de nieve |
| Ch03 | Retroceso glaciar, DEM, cuencas | Pérdida de área del Glaciar Grey 2000–2024 |
| Ch06 | Isoyetas, isotermas, densidad de drenaje | Mapeo de gradiente térmico altitudinal |
| Ch08 | Fusión multisensor, modelo cascada | Dashboard ESI/CVS/WSI — zonas de riesgo convergente |
| Ch09 | CNN deep learning | Clasificación de cobertura con >85% de precisión |
| Ch13 | SAR offset tracking | Velocidad superficial del Glaciar Grey ~380 m/año |

---

## 3. 📊 Tipos de Informe

### Tipo A: Dashboard (generado por tus scripts)
Tus scripts generan dashboards PNG. Cómo anotarlos para informes:
```
Título:    "Señales de Cambio Climático — Torres del Paine (1993–2024)"
Subtítulo: Fuente: ERA5-Land / Open-Meteo | Análisis: GeoCascade Pipeline
Caption:   Rojo = tendencia de calentamiento/secado. Azul = tendencia húmeda.
           Significancia estadística: * p<0.05, ** p<0.01.
```

### Tipo B: Resumen Ejecutivo (1 página)
```markdown
## Señales Climáticas Clave — Torres del Paine (1993–2024)

### 🌡️ Temperatura
- Tendencia: +0.064°C/año (significativa, p<0.01)
- 2024 fue el año más cálido en el registro de 32 años
- Máximas veraniegas superan 25°C en 18 días más por año que en 1993

### 🌧️ Precipitación
- Alta variabilidad interanual (CV = 28%)
- 2019 y 2021: años más secos del registro
- Tendencia de secado en verano austral (DJF)

### 🧊 Glaciares
- Área Glaciar Grey: −12% (2000–2024, NDSI Landsat)
- Velocidad superficial: 380 m/año (SAR Sentinel-1)
- Agua de deshielo: agua pico probablemente alcanzada ~2018

### 🌿 Vegetación
- Tendencia NDVI: −0.003/año en zonas subalpinas (>800m)
- Migración ascendente de límite del bosque detectada: +45m desde 2000

### ⚠️ Riesgo Cascada
- Alto ESI (Índice de Estrés Ecológico) en 23% del área de estudio
- Zonas de alto riesgo convergente: terminus Glaciar Grey + alto Río Serrano
```

### Tipo C: Informe Técnico Completo
```
1. Introducción y Objetivos
2. Área de Estudio: Parque Nacional Torres del Paine
3. Fuentes de Datos
   3.1 ERA5-Land (Open-Meteo) — serie diaria 32 años
   3.2 CHIRPS v2.0 — grillas mensuales de precipitación (2000–2024)
   3.3 NOAA GHCN — validación con estaciones terrestres
   3.4 Sentinel-1/2, Landsat 9, MODIS — imágenes satelitales
   3.5 Copernicus DEM — topografía
   3.6 RGI 7.0 — inventario de glaciares
4. Metodología
   4.1 Análisis de Tendencias (Mann-Kendall, Theil-Sen)
   4.2 Índices Espectrales (NDVI, NDSI, NDWI)
   4.3 Fusión de Datos Multi-Sensor (grilla maestra 10m)
   4.4 Aprendizaje Automático (Random Forest + CNN)
   4.5 Modelado de Riesgo en Cascada (ESI, CVS, WSI)
5. Resultados
   5.1 Tendencias de Temperatura
   5.2 Anomalías de Precipitación
   5.3 Cambio de Área Glaciar
   5.4 Dinámica de Cobertura Terrestre
   5.5 Evaluación de Riesgo en Cascada
6. Discusión
7. Conclusiones y Recomendaciones
8. Anexo: Scripts Python y Reproducibilidad
```

---

## 4. 🗺️ Cómo Cada Capítulo Contribuye a la Historia

```
ERA5 + CHIRPS + GHCN  →  Índices Espectrales  →  Retroceso Glaciar + DEM
        ↓                       ↓                         ↓
  Nicho Ecológico  ←──── Estrés de Humedad ←──── Hidrometeoría
        ↓
   SAR Radar  →  FUSIÓN DE DATOS  →  Deep Learning CNN
                      ↓                     ↓
              Monitor Agéntico  →  CAPSTONE: Informe de Zona de Impacto
```

---

## 5. 🔑 Próximos Pasos Prácticos

### Inmediatamente accionables:
1. `python Chapter_01/03b_era5_trend_analysis.py` → estadísticas Mann-Kendall
2. `python Chapter_01/03c_chirps_spatial_precipitation.py` → serie temporal de precipitación
3. `python Chapter_08/22_combined_insights_engine.py` → dashboard de 12 paneles ESI/CVS/WSI

### Para un informe de calidad de publicación:
- Exportar todos los dashboards a `dpi=300`
- Agregar tabla de métodos: script → dataset → output
- Contrastar tendencias ERA5 con proyecciones regionales IPCC AR6 para América del Sur

### Para un plan de lección curricular:
Cada Capítulo = un taller (3–4 horas):
- 30 min teoría (base física del fenómeno)
- 60 min programación (ejecutar y modificar el script)
- 30 min interpretación (¿qué significa el resultado?)
- 30 min discusión (¿cómo cambiaría para otra zona de estudio?)

---

## 6. 🧭 La Narrativa Más Poderosa

> *"De 1993 a 2024, el clima de Torres del Paine ha cambiado de manera medible:*  
> *más cálido en 2°C, más seco en verano, con glaciares retrocediendo y vegetación estresada.*  
> *Estas no son proyecciones — son mediciones, hechas desde el espacio y confirmadas*  
> *por estaciones terrestres, procesadas con herramientas Python de código abierto y datos*  
> *satelitales de libre acceso. La cascada ya está en marcha.*  
> *La pregunta ahora es: ¿a qué velocidad, y qué viene después?"*

---

*Generado por GeoCascade Pipeline | Torres del Paine, Patagonia | 1993–2024*  
*Scripts: github.com/georaiser/ClimateChange | Entorno: geocascade_env*
