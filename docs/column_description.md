# Column Description / Descripcion de Columnas

Training dataset for BAP (Berth Allocation Problem) service time prediction at Puerto de San Antonio, Chile.

File: `data/training_dataset.csv` | 5,597 rows | 44 columns

**Naming convention:** Columns taken directly from the source data keep their original Spanish names (e.g. `Cód. nave`, `F. arribo`, `Terminal`). Derived or computed columns use English names (e.g. `vessel_type_group`, `vessel_avg_berth_stay`, `arrival_month`).

---

## Target Variables / Variables Objetivo

| Column | Spanish Original | Description | Calculation |
|--------|-----------------|-------------|-------------|
| `estadia_sitio_hours` | Estadía sitio | Berth occupation time in hours. Time the vessel physically occupies the berth, from first mooring line to last unmooring line. Primary target for berth allocation. / Tiempo de ocupación del sitio de atraque en horas. | `Última espía desatraque - 1era espía atraque` |
| `tiempo_en_puerto_hours` | Tiempo en puerto | Time the vessel is inside the port in hours, from pilot boarding (port entry) to departure. Secondary target for port channel scheduling. / Tiempo que la nave permanece dentro del puerto en horas. | `Zarpe - Fecha práctico atraque` |

---

## Direct Features / Características Directas

| Column | Source | Description | Calculation |
|--------|--------|-------------|-------------|
| `Cód. nave` | Direct | Unique vessel identifier (IMO number). / Código identificador de la nave. | Direct from source |
| `Nave` | Direct | Vessel name. / Nombre de la nave. | Direct from source |
| `T. nave` | Direct | Original vessel type (16 categories). / Tipo de nave original. | Direct from source |
| `vessel_type_group` | Derived | Aggregated vessel type (7 groups: Container, Dry Bulk, Vehicle Carrier, Liquid Bulk, General Cargo, Passenger, Other). / Tipo de nave agrupado en 7 categorías. | Mapped from `T. nave` via grouping dictionary |
| `TRG` | Direct | Gross registered tonnage. / Tonelaje de registro bruto. | Direct from source |
| `Terminal` | Direct | Port terminal (DP World, STI, PANUL, QC, EPSA). / Terminal portuario. | Direct from source |
| `agency` | Derived | Maritime agency name. / Nombre de la agencia marítima. | Extracted from `Agencia` ("RUT - NAME" format), taking the name part |
| `L. naviera` | Direct | Shipping line. "UNKNOWN" for missing values (30% of records, mostly non-container vessels). / Línea naviera. | Direct from source, NaN filled with "UNKNOWN" |
| `Servicio` | Direct | Shipping service/route name. "UNKNOWN" for missing values. / Nombre del servicio o ruta naviera. | Direct from source, NaN filled with "UNKNOWN" |
| `Puerto origen` | Direct | Port of origin. / Puerto de origen. | Direct from source |
| `Puerto destino` | Direct | Port of destination. / Puerto de destino. | Direct from source |
| `origin_region` | Derived | Geographic region of origin port (14 regions). / Región geográfica del puerto de origen. | Mapped from `Puerto origen` via `port_regions.py` dictionary |
| `dest_region` | Derived | Geographic region of destination port. / Región geográfica del puerto de destino. | Mapped from `Puerto destino` via `port_regions.py` dictionary |
| `C. arribo proa` | Direct | Bow draft on arrival in meters. / Calado de proa al arribo en metros. | Direct from source |
| `C. arribo popa` | Direct | Stern draft on arrival in meters. / Calado de popa al arribo en metros. | Direct from source |
| `draft_arrival_mean` | Derived | Mean arrival draft in meters. / Calado medio al arribo en metros. | `(C. arribo proa + C. arribo popa) / 2` |
| `max_arrival_draft` | Derived | Maximum arrival draft in meters. Relevant for berth depth constraints. / Calado máximo al arribo en metros. | `max(C. arribo proa, C. arribo popa)` |
| `draft_trim_arrival` | Derived | Arrival trim in meters. Positive means stern-heavy (loaded aft). Indicates vessel loading condition. / Trimado al arribo en metros. Positivo indica mayor calado en popa. | `C. arribo popa - C. arribo proa` |

---

## Temporal Features / Características Temporales

All derived from `F. arribo` (arrival at anchorage datetime). / Todas derivadas de la fecha de arribo al fondeadero.

| Column | Spanish gloss | Description | Calculation |
|--------|---------------|-------------|-------------|
| `arrival_month` | Mes de arribo | Month of arrival (1-12). / Mes del arribo. | `F. arribo.month` |
| `arrival_day_of_week` | Día de la semana | Day of week (0=Monday, 6=Sunday). / Día de la semana. | `F. arribo.weekday()` |
| `arrival_hour` | Hora de arribo | Hour of arrival (0-23). / Hora del arribo. | `F. arribo.hour` |
| `arrival_year` | Año de arribo | Year of arrival. Captures trends (e.g. COVID-era elevated stays). / Año del arribo. | `F. arribo.year` |
| `quarter` | Trimestre | Quarter of the year (1-4). / Trimestre del año. | `(month - 1) // 3 + 1` |
| `is_weekend_arrival` | Arribo fin de semana | Whether arrival falls on Saturday or Sunday (0/1). / Si el arribo es en fin de semana. | `1 if day_of_week >= 5 else 0` |

---

## Historical Vessel Features / Características Históricas por Nave

Computed using an expanding window over prior visits of the same vessel, sorted chronologically by first mooring time. Shifted by 1 to exclude the current visit and prevent data leakage. NaN for first visits. / Calculadas con ventana expansiva sobre visitas previas de la misma nave, ordenadas cronológicamente. Desplazadas en 1 para excluir la visita actual. NaN para primeras visitas.

| Column | Spanish gloss | Description | Calculation |
|--------|---------------|-------------|-------------|
| `vessel_avg_berth_stay` | Promedio estadía nave | Mean berth stay (hours) of this vessel's prior visits. / Promedio de estadía en sitio de visitas previas de esta nave. | `groupby("Cód. nave")["estadia_sitio_hours"].expanding().mean().shift(1)` |
| `vessel_median_berth_stay` | Mediana estadía nave | Median berth stay of prior visits. / Mediana de estadía en sitio de visitas previas. | `groupby("Cód. nave")["estadia_sitio_hours"].expanding().median().shift(1)` |
| `vessel_std_berth_stay` | Desv. est. estadía nave | Standard deviation of berth stay from prior visits. / Desviación estándar de estadía en sitio de visitas previas. | `groupby("Cód. nave")["estadia_sitio_hours"].expanding().std().shift(1)` |
| `vessel_visit_count` | Visitas previas nave | Number of prior visits by this vessel (0 = first visit). / Número de visitas previas de esta nave. | `groupby("Cód. nave").cumcount()` |
| `vessel_last_berth_stay` | Última estadía nave | Berth stay of this vessel's most recent prior visit. / Estadía en sitio de la última visita previa de esta nave. | `groupby("Cód. nave")["estadia_sitio_hours"].shift(1)` |
| `vessel_avg_berth_stay_at_terminal` | Promedio estadía nave en terminal | Mean berth stay of this vessel's prior visits at the same terminal. / Promedio de estadía en sitio de visitas previas en el mismo terminal. | `groupby(["Cód. nave", "Terminal"])["estadia_sitio_hours"].expanding().mean().shift(1)` |
| `vessel_visit_count_at_terminal` | Visitas previas nave en terminal | Number of prior visits by this vessel to this terminal. / Número de visitas previas de esta nave al mismo terminal. | `groupby(["Cód. nave", "Terminal"]).cumcount()` |

---

## Group-Level Features / Características a Nivel de Grupo

Expanding-window averages at higher grouping levels, shifted by 1. Useful as fallback predictions when vessel-specific history is unavailable. / Promedios con ventana expansiva a nivel de grupo, desplazadas en 1. Útiles como predicción base cuando no hay historial específico de la nave.

| Column | Spanish gloss | Description | Calculation |
|--------|---------------|-------------|-------------|
| `type_terminal_avg_stay` | Promedio tipo+terminal | Historical mean berth stay for this (vessel type group, terminal) combination. / Promedio histórico de estadía para esta combinación (tipo de nave, terminal). | `groupby(["vessel_type_group", "Terminal"])["estadia_sitio_hours"].expanding().mean().shift(1)` |
| `type_avg_stay` | Promedio tipo | Historical mean berth stay for this vessel type group. / Promedio histórico de estadía para este tipo de nave. | `groupby("vessel_type_group")["estadia_sitio_hours"].expanding().mean().shift(1)` |
| `terminal_avg_stay` | Promedio terminal | Historical mean berth stay for this terminal. / Promedio histórico de estadía para este terminal. | `groupby("Terminal")["estadia_sitio_hours"].expanding().mean().shift(1)` |

---

## Reference Columns / Columnas de Referencia

Not intended as model features. Included for traceability and analysis. / No son características para el modelo. Incluidas para trazabilidad y análisis.

| Column | Source | Description | Calculation |
|--------|--------|-------------|-------------|
| `F. arribo` | Direct | Vessel arrival at anchorage (outside port). / Fecha de arribo al fondeadero (fuera del puerto). | Direct from source |
| `Fecha práctico atraque` | Direct | Pilot boarding for berthing (marks port entry). / Fecha en que el práctico aborda para el atraque (marca la entrada al puerto). | Direct from source |
| `1era espía atraque` | Direct | First mooring line secured (berthing start). / Primera espía de atraque firme (inicio del atraque). | Direct from source |
| `Última espía desatraque` | Direct | Last mooring line cast off (berth vacated). / Última espía de desatraque largada (sitio desocupado). | Direct from source |
| `Zarpe` | Direct | Vessel departure from port. / Zarpe de la nave del puerto. | Direct from source |
| `espera_preatraque_hours` | Derived | Time waiting at anchorage before berthing in hours. Includes anchorage wait + channel transit. / Tiempo de espera en fondeadero antes del atraque en horas. | `(Zarpe - F. arribo) - estadia_sitio_hours` |
| `quality_flag` | Derived | Data quality flag: 1 if vessel dispatch datetime precedes reception datetime (12 records), 0 otherwise. Target variables remain valid. / Bandera de calidad: 1 si la fecha de despacho es anterior a la de recepción. | `1 if Fecha despacho < Fecha recepción, else 0` |
| `split` | Derived | Suggested temporal split for model training. / Partición temporal sugerida para entrenamiento. | train: before 2024-07-01, validation: 2024-07-01 to 2025-02-28, test: after 2025-03-01 |

---

## Vessel Type Aggregation / Agrupación de Tipos de Nave

| Group / Grupo | Original Types / Tipos Originales | Count / Registros |
|---------------|-----------------------------------|-------------------|
| Container | Contenedor | 3,066 |
| Dry Bulk | Carga Seca Granel, Mineral/Granel/Petrolero | 951 |
| Vehicle Carrier | Autero, Autotrasbordo | 749 |
| Liquid Bulk | Transporte Quimico, Transporte Liquido, Transporte de Asfalto, Petrolero | 587 |
| General Cargo | Tradicional, Carga de Proyecto, Chipero, Refrigerado, Otros | 171 |
| Passenger | Pasajeros | 80 |
| Other | Nave Armada | 1 |

## Port Regions / Regiones Portuarias

| Region / Región | Examples / Ejemplos |
|-----------------|---------------------|
| Chilean_North | Arica, Iquique, Antofagasta, Mejillones, Coquimbo |
| Chilean_Central | Valparaiso, San Antonio, Coronel, Lirquen, San Vicente |
| Chilean_South | Puerto Montt, Punta Arenas, Chacabuco |
| Peru | Callao, Ilo, Matarani, Paita |
| Brazil | Santos, Rio Grande, Paranagua, Rio de Janeiro |
| Argentina | Buenos Aires, Rosario, Zarate, Bahia Blanca, Montevideo |
| Ecuador_Colombia | Guayaquil, Buenaventura, Manta |
| Central_America | Balboa, Cristobal, Manzanillo, Lazaro Cardenas |
| North_America | Houston, Long Beach, Los Angeles, Vancouver |
| Asia | Shanghai, Busan, Singapore, Hong Kong, Yokohama |
| Europe | Rotterdam, Zeebrugge, London, Livorno |
| Africa | Durban, Capetown, Port Elizabeth |
| Oceania | Melbourne, Fremantle, Tauranga |
| Other | Unmapped ports / Puertos no mapeados |
