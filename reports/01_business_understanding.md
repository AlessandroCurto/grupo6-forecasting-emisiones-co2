# Fase 1 - Business Understanding

**Tema:** Forecasting de emisiones de CO2  
**Área:** Medio Ambiente

## Contexto

Los países firmantes del Acuerdo de París deben reportar y cumplir sus Contribuciones Determinadas a Nivel Nacional (NDC). En el caso del Perú, la NDC actualizada en 2020 plantea reducir hasta 40 % de sus emisiones de GEI respecto al escenario tendencial (BAU) al 2030. Anticipar la trayectoria de las emisiones de CO2 permite a los ministerios de ambiente y energía evaluar si las políticas vigentes son suficientes y priorizar medidas de transición energética.

## Problema

Las cifras oficiales de emisiones se publican con 1-2 años de rezago y no dicen hacia dónde se dirige cada país. Se requiere un insumo cuantitativo para pronosticar las emisiones de CO2 por país a partir de sus determinantes económicos, demográficos y energéticos.

## Interesados

- Ministerios de Ambiente y de Energía y Minas (p. ej., MINAM y MINEM en el Perú)
- Organismos multilaterales y bancos de desarrollo (financiamiento climático)
- Investigadores y ONG ambientales

## Objetivo de negocio

Contar con estimaciones anticipadas (horizonte de 1 a 5 años) de las emisiones de CO2 por país para evaluar la brecha frente a las metas NDC y priorizar políticas de mitigación.

## Objetivo de minería de datos

Construir un dataset panel país-año (1990-2024) limpio, sin valores faltantes, con variables explicativas rezagadas y sin fuga de información, que permita en la siguiente fase entrenar modelos de regresión/series de tiempo que pronostiquen las emisiones totales de CO2 (Mt CO2e) de cada país.

## Criterios de éxito

- **Negocio:** Que el pronóstico permita identificar con al menos 1 año de anticipación a los países que se alejan de su trayectoria de reducción.
- **Machine learning:** MAPE < 10 % en el periodo de prueba temporal (2021-2024); se evaluará en la fase de modelado, fuera del alcance de este trabajo.
- **Economico:** Usar exclusivamente datos abiertos y software libre (costo de licencias = 0).

## Alcance

Fases 1 a 3 de CRISP-ML(Q). No se entrena ni ejecuta ningún modelo.

## Supuestos

- Las emisiones de CO2 dependen de la actividad económica (PIB), la población, la estructura energética y la urbanización (identidad de Kaya / modelo STIRPAT).
- Las series del Banco Mundial son comparables entre países y en el tiempo.

## Riesgos identificados

- Indicadores energéticos discontinuados (el Banco Mundial dejó de actualizar algunos).
- Choques exógenos (crisis 2009, COVID-19 2020) que alteran la tendencia.
- Mezcla de países con agregados regionales en la API (doble conteo).
- Escalas muy distintas entre países (China vs. pequeños estados insulares).
- Fuga de información si se usan variables derivadas del objetivo en el mismo año.

## Requisitos de calidad (QA gates)

```json
{
  "periodo": [
    1990,
    2024
  ],
  "min_paises": 150,
  "cobertura_minima_objetivo_por_pais": 0.9,
  "max_faltantes_por_variable": 0.3,
  "max_nulos_dataset_final": 0,
  "max_duplicados_pais_anio": 0,
  "rangos_validos": {
    "porcentajes": [
      0,
      100
    ],
    "niveles": "valores > 0",
    "co2_per_capita_minimo_t": 0.01
  },
  "consistencia_co2_total_vs_per_capita": 0.05,
  "series_sin_huecos_temporales": true,
  "sin_fuga_de_informacion": true
}
```