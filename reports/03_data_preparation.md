# Fase 3 - Data Preparation

Dataset final: `data/processed/co2_dataset_preparado.csv` - **5,888 filas x 58 columnas**, 184 países, años 1993-2024.

## Bitácora de decisiones

| etapa | decision |
|---|---|
| Selección | Variables descartadas por > 30% de faltantes: energia_pc, fosil_pct |
| Selección | Variable descartada por fuga de información: co2_pc |
| Selección | Países conservados: 195 de 217 (cobertura de CO2 >= 90%) |
| Limpieza | Valores de PIB/PIB per cápita reconstruidos por identidad: 0 |
| Limpieza | Duplicados país-año eliminados: 0 |
| Limpieza | Valores fuera de rango convertidos a faltantes: 0 |
| Limpieza | Faltantes antes de imputar: 3522; imputados con mediana regional: 1629 |
| Limpieza | Filas eliminadas por objetivo faltante en extremos: 0 |
| Limpieza | Países eliminados por faltantes irrecuperables: AFG, CYM, DJI, ERI, FRO, GIB, NCL, PRK, TCA, VGB, YEM |
| Limpieza | Países eliminados por huecos temporales: ninguno |
| Outliers | Saltos anómalos detectados (|z robusto| > 3.5 y magnitud relevante): 986 -> {'cambio estructural (conservado)': 811, 'choque conocido (conservado)': 96, 'pico aislado (corregido)': 79} |
| Transformación | Variables creadas: log, crecimientos, 3 rezagos del objetivo, rezagos t-1 de covariables, tendencia e indicadoras de choque [2009, 2020] |
| Transformación | Filas iniciales sin historia suficiente eliminadas: 552 |
| Integración | Unión con metadatos de países (región, ingreso); 11 variables one-hot creadas |
| Partición | Partición temporal sugerida para el modelado: {'entrenamiento': 4416, 'validacion': 736, 'prueba': 736} |

## Outliers detectados (resumen)

| variable | tipo | n |
|---|---|---|
| carbon_elec_pct | cambio estructural (conservado) | 55 |
| carbon_elec_pct | choque conocido (conservado) | 7 |
| carbon_elec_pct | pico aislado (corregido) | 8 |
| co2_mt | cambio estructural (conservado) | 173 |
| co2_mt | choque conocido (conservado) | 23 |
| co2_mt | pico aislado (corregido) | 16 |
| electricidad_pct | cambio estructural (conservado) | 168 |
| electricidad_pct | choque conocido (conservado) | 5 |
| electricidad_pct | pico aislado (corregido) | 26 |
| industria_pct | cambio estructural (conservado) | 128 |
| industria_pct | choque conocido (conservado) | 9 |
| industria_pct | pico aislado (corregido) | 11 |
| pib | cambio estructural (conservado) | 93 |
| pib | choque conocido (conservado) | 23 |
| pib | pico aislado (corregido) | 7 |
| pib_pc | cambio estructural (conservado) | 91 |
| pib_pc | choque conocido (conservado) | 21 |
| pib_pc | pico aislado (corregido) | 6 |
| poblacion | cambio estructural (conservado) | 7 |
| poblacion | pico aislado (corregido) | 2 |
| renovable_pct | cambio estructural (conservado) | 91 |
| renovable_pct | choque conocido (conservado) | 8 |
| renovable_pct | pico aislado (corregido) | 3 |
| urbano_pct | cambio estructural (conservado) | 5 |

Detalle completo en `reports/03_outliers_detectados.csv`.

## Controles de calidad del dataset final

| control | valor | cumple |
|---|---|---|
| Sin valores nulos | 0 | True |
| Sin duplicados país-año | 0 | True |
| N.° de países >= 150 | 184 | True |
| Porcentajes en [0, 100] | ['urbano_pct', 'renovable_pct', 'electricidad_pct', 'industria_pct', 'carbon_elec_pct'] | True |
| Objetivo estrictamente positivo | 0.0012 | True |
| Sin valores infinitos | 0 | True |
| Años consecutivos en cada país | True | True |
| Sin fuga de información (derivadas del CO2 solo rezagadas) | ninguna | True |
| Todas las particiones con datos | {'entrenamiento': 4416, 'validacion': 736, 'prueba': 736} | True |

## Diccionario de datos

| columna | tipo | descripcion |
|---|---|---|
| iso3 | str | Código ISO3 del país |
| pais | str | Nombre del país |
| region | str | Región del Banco Mundial |
| nivel_ingreso | str | Nivel de ingreso del Banco Mundial |
| anio | int64 | Año |
| co2_mt | float64 | Emisiones de CO2 totales excl. LULUCF (Mt CO2e) |
| log_co2 | float64 | OBJETIVO transformado: ln(CO2 Mt) |
| pib | float64 | PIB a precios constantes de 2015 (US$) |
| pib_pc | float64 | PIB per cápita a precios constantes de 2015 (US$) |
| poblacion | float64 | Población total (habitantes) |
| urbano_pct | float64 | Población urbana (% del total) |
| renovable_pct | float64 | Consumo de energía renovable (% del consumo final) |
| electricidad_pct | float64 | Acceso a la electricidad (% de la población) |
| industria_pct | float64 | Industria incl. construcción, valor agregado (% del PIB) |
| carbon_elec_pct | float64 | Electricidad producida con carbón (% del total) |
| log_pib | float64 | Logaritmo natural de PIB a precios constantes de 2015 (US$) |
| log_pib_pc | float64 | Logaritmo natural de PIB per cápita a precios constantes de 2015 (US$) |
| log_poblacion | float64 | Logaritmo natural de Población total (habitantes) |
| crec_pib_pct | float64 | Crecimiento anual del PIB (%) |
| crec_poblacion_pct | float64 | Crecimiento anual de la población (%) |
| log_co2_rezago1 | float64 | log_co2 rezagado 1 año(s) |
| log_co2_rezago2 | float64 | log_co2 rezagado 2 año(s) |
| log_co2_rezago3 | float64 | log_co2 rezagado 3 año(s) |
| crec_co2_rezago1 | float64 | Variación anual del CO2 en t-1 (%, log) |
| log_co2_media3_rezago1 | float64 | Media móvil de 3 años de log_co2 (t-1, t-2, t-3) |
| intensidad_carbono_rezago1 | float64 | Intensidad de carbono en t-1 (kg CO2 / US$ de PIB) |
| log_pib_rezago1 | float64 | Logaritmo de PIB a precios constantes de 2015 (US$) en t-1 |
| log_pib_pc_rezago1 | float64 | Logaritmo de PIB per cápita a precios constantes de 2015 (US$) en t-1 |
| log_poblacion_rezago1 | float64 | Logaritmo de Población total (habitantes) en t-1 |
| urbano_pct_rezago1 | float64 | Población urbana (% del total) en t-1 |
| renovable_pct_rezago1 | float64 | Consumo de energía renovable (% del consumo final) en t-1 |
| electricidad_pct_rezago1 | float64 | Acceso a la electricidad (% de la población) en t-1 |
| industria_pct_rezago1 | float64 | Industria incl. construcción, valor agregado (% del PIB) en t-1 |
| carbon_elec_pct_rezago1 | float64 | Electricidad producida con carbón (% del total) en t-1 |
| tendencia | int64 | Años desde 1990 |
| choque_2009 | int64 | 1 si el año es 2009 (Crisis financiera global) |
| choque_2020 | int64 | 1 si el año es 2020 (Pandemia COVID-19) |
| reg_africa_subsahariana | int64 | 1 si el país pertenece a la región África Subsahariana |
| reg_america_latina_caribe | int64 | 1 si el país pertenece a la región América Latina y el Caribe |
| reg_america_norte | int64 | 1 si el país pertenece a la región América del Norte |
| reg_asia_oriental_pacifico | int64 | 1 si el país pertenece a la región Asia Oriental y Pacífico |
| reg_asia_sur | int64 | 1 si el país pertenece a la región Asia del Sur |
| reg_europa_asia_central | int64 | 1 si el país pertenece a la región Europa y Asia Central |
| reg_medio_oriente_norte_africa | int64 | 1 si el país pertenece a la región Medio Oriente, Norte de África, Afganistán y Pakistán |
| ing_alto | int64 | 1 si el nivel de ingreso del país es ingreso alto |
| ing_bajo | int64 | 1 si el nivel de ingreso del país es ingreso bajo |
| ing_medio_alto | int64 | 1 si el nivel de ingreso del país es ingreso medio alto |
| ing_medio_bajo | int64 | 1 si el nivel de ingreso del país es ingreso medio bajo |
| imputado_co2_mt | int64 | 1 si co2_mt fue imputado en la limpieza |
| imputado_pib | int64 | 1 si pib fue imputado en la limpieza |
| imputado_pib_pc | int64 | 1 si pib_pc fue imputado en la limpieza |
| imputado_poblacion | int64 | 1 si poblacion fue imputado en la limpieza |
| imputado_urbano_pct | int64 | 1 si urbano_pct fue imputado en la limpieza |
| imputado_renovable_pct | int64 | 1 si renovable_pct fue imputado en la limpieza |
| imputado_electricidad_pct | int64 | 1 si electricidad_pct fue imputado en la limpieza |
| imputado_industria_pct | int64 | 1 si industria_pct fue imputado en la limpieza |
| imputado_carbon_elec_pct | int64 | 1 si carbon_elec_pct fue imputado en la limpieza |
| particion | str | Partición temporal sugerida (entrenamiento/validacion/prueba) |