# Fase 2 - Data Understanding

## 2.1 Recolección inicial de datos

- Fuente: World Bank Open Data (API v2). Periodo 1990-2024.

- Registros crudos (incluye agregados regionales): 111,300

- Panel de países: 7,595 filas país-año, 217 países, 12 indicadores.

## 2.2 Exploración de variables y relaciones

Último año con datos de CO2: **2024**.

Top 10 emisores:

1. China - 13,124.7 Mt
2. United States - 4,632.2 Mt
3. India - 3,153.8 Mt
4. Russian Federation - 2,009.2 Mt
5. Japan - 972.3 Mt
6. Iran, Islamic Rep. - 829.0 Mt
7. Indonesia - 812.2 Mt
8. Saudi Arabia - 652.5 Mt
9. Korea, Rep. - 588.0 Mt
10. Germany - 579.9 Mt

Correlación de Spearman con `co2_mt`:

- `pib`: 0.944
- `poblacion`: 0.77
- `co2_pc`: 0.644
- `carbon_elec_pct`: 0.598
- `energia_pc`: 0.544
- `industria_pct`: 0.505
- `electricidad_pct`: 0.36
- `fosil_pct`: 0.344
- `urbano_pct`: 0.343
- `pib_pc`: 0.327
- `renovable_pct`: -0.198

Figuras: `fig01` a `fig06` en `reports/figures/`.

## 2.3 Evaluación de la calidad de los datos

| variable | descripcion | faltantes_pct | paises_con_datos | primer_anio | ultimo_anio | fuera_de_rango | outliers_iqr |
|---|---|---|---|---|---|---|---|
| co2_mt | Emisiones de CO2 totales excl. LULUCF (Mt CO2e) | 6.45 | 203 | 1990 | 2024 | 70 | 259 |
| co2_pc | Emisiones de CO2 per cápita excl. LULUCF (t CO2e/hab) | 6.45 | 203 | 1990 | 2024 | 70 | 210 |
| pib | PIB a precios constantes de 2015 (US$) | 6.46 | 213 | 1990 | 2024 | 0 | 47 |
| pib_pc | PIB per cápita a precios constantes de 2015 (US$) | 6.46 | 213 | 1990 | 2024 | 0 | 0 |
| poblacion | Población total (habitantes) | 0.00 | 217 | 1990 | 2024 | 0 | 0 |
| urbano_pct | Población urbana (% del total) | 0.00 | 217 | 1990 | 2024 | 0 | 0 |
| renovable_pct | Consumo de energía renovable (% del consumo final) | 11.18 | 212 | 1990 | 2022 | 0 | 0 |
| energia_pc | Uso de energía (kg equivalente de petróleo per cápita) | 31.59 | 179 | 1990 | 2024 | 1 | 4 |
| fosil_pct | Consumo de energía de combustibles fósiles (% del total) | 34.46 | 179 | 1990 | 2024 | 6 | 0 |
| electricidad_pct | Acceso a la electricidad (% de la población) | 11.17 | 216 | 1990 | 2024 | 0 | 760 |
| industria_pct | Industria incl. construcción, valor agregado (% del PIB) | 15.38 | 207 | 1990 | 2024 | 0 | 294 |
| carbon_elec_pct | Electricidad producida con carbón (% del total) | 19.20 | 209 | 1990 | 2024 | 0 | 715 |

### Problemas detectados

- **registros_panel**: 7595
- **paises**: 217
- **duplicados_pais_anio_indicador**: 0
- **paises_sin_ningun_dato_co2_valido**: 22
- **paises_con_co2_igual_a_cero**: ['NRU', 'TUV']
- **paises_con_co2_pc_menor_a_0.01_t**: ['ASM', 'FSM', 'GUM', 'MHL', 'MNP', 'VIR']
- **paises_cobertura_co2_menor_umbral**: 22
- **huecos_internos_co2**: 0
- **saltos_anomalos_co2_z_robusto**: 228
- **inconsistencias_co2_total_vs_per_capita**: 0
- **variables_sobre_umbral_faltantes**: ['energia_pc', 'fosil_pct']

### Factibilidad frente a los requisitos de la fase 1

| control | valor | cumple |
|---|---|---|
| Periodo disponible cubre el requerido | 1990-2024 | True |
| Países con cobertura de CO2 >= 90% | 195 | True |