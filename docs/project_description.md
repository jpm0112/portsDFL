# BAP Service Time Prediction - Project Description

## Objective

Predict the service time of vessels at Puerto de San Antonio, Chile, to support a Berth Allocation Problem (BAP) optimizer. The port has 5 terminals (DP World, STI, PANUL, QC, EPSA) that independently plan their berth schedules but share a common port entry channel. Accurate service time predictions enable better berth scheduling and port entry coordination.

## Problem Context

The Berth Allocation Problem assigns incoming vessels to available berths while minimizing waiting times and maximizing berth utilization. A key input to the BAP is the expected service time: how long each vessel will occupy a berth. This project builds a training dataset to develop a predictive model for that service time, using only information available before the vessel berths.

### Port Operations Sequence

A vessel call at San Antonio follows this sequence:

1. **Arrival at anchorage** (`F. arribo`) - vessel arrives and anchors outside the port
2. **Waiting at anchorage** - vessel waits for berth availability (can be days for bulk carriers)
3. **Pilot boarding** (`Fecha práctico atraque`) - pilot boards, vessel enters the port channel
4. **Mooring** (`1era espía atraque` to `Última espía atraque`) - ~42 minutes average
5. **Cargo operations** - loading/unloading at berth
6. **Unmooring** (`1era espía desatraque` to `Última espía desatraque`)
7. **Departure** (`Zarpe`) - vessel exits the port

### Key Insight

`F. arribo` records arrival at the anchorage (outside the port), not entry into the port. Dry bulk vessels wait a median of 207 hours (~8.6 days) at anchor. The pilot boarding timestamp marks actual port entry, with a consistent ~42-minute transit from pilot boarding to first mooring line across all vessel types.

## Target Variables

Two service time metrics are modeled:

| Target | Formula | Use Case |
|--------|---------|----------|
| **Berth stay** (`estadia_sitio_hours`) | Last unmooring - First mooring | Berth allocation: how long the berth is occupied |
| **Port time** (`tiempo_en_puerto_hours`) | Departure - Pilot boarding | Channel scheduling: how long the vessel is inside the port |

## Data Source

- **File**: `data/BBDD limpia(1).xlsx`, sheet "Resume Naves Comerciales (4)"
- **Records**: 5,605 vessel calls (5,597 after cleaning)
- **Period**: December 2019 to August 2025
- **Completeness**: All timestamp and numeric columns are 100% complete. Only `L. naviera` (shipping line) and `Servicio` (service route) have missing values (~30%), concentrated in non-container vessel types.

## Feature Engineering

Features are designed to use only pre-berthing information (no data leakage):

- **Direct features**: vessel type, gross tonnage, terminal, agency, shipping line, arrival drafts, origin/destination ports and regions
- **Temporal features**: month, day of week, hour, year, quarter, weekend flag
- **Historical features**: expanding-window statistics of the same vessel's prior berth stays (mean, median, std, last visit, visit count), computed with chronological ordering and shift-by-1 to prevent leakage
- **Group-level features**: expanding-window averages at (vessel type, terminal), vessel type, and terminal levels

Berth assignment (`Sitio`) is excluded as a feature because it is the decision variable in the BAP.

## Vessel Type Grouping

The 16 original vessel types are aggregated into 7 operationally meaningful groups based on cargo handling methods and service time patterns:

| Group | Types | Count | Avg Berth Stay |
|-------|-------|-------|----------------|
| Container | Contenedor | 3,066 | 36h |
| Dry Bulk | Carga Seca Granel, Mineral/Granel/Petrolero | 951 | 74h |
| Vehicle Carrier | Autero, Autotrasbordo | 749 | 36h |
| Liquid Bulk | Transporte Quimico/Liquido/Asfalto, Petrolero | 587 | 22h |
| General Cargo | Tradicional, Carga de Proyecto, Chipero, Refrigerado, Otros | 171 | 36h |
| Passenger | Pasajeros | 80 | 19h |
| Other | Nave Armada | 1 | 65h |

## Data Cleaning

- 7 records with berth stay < 2 hours removed (aborted calls or data errors)
- 1 record with berth stay of 780 hours removed (anomalous)
- 12 records flagged where dispatch timestamp precedes reception (quality issue, targets remain valid)

## Suggested Train/Test Split

A temporal split prevents data leakage from the historical features:

| Split | Period | Records |
|-------|--------|---------|
| Train | Before 2024-07-01 | 4,413 |
| Validation | 2024-07-01 to 2025-02-28 | 688 |
| Test | After 2025-03-01 | 496 |

## External Data Enrichment

External datasets were collected to supplement the main vessel call database:

### Weather Data (Open-Meteo API, CC BY 4.0)
- **weather_daily.csv**: 2,056 days of temperature, precipitation, wind speed/gusts/direction (complete coverage, no gaps)
- **weather_hourly.csv**: 49,344 hours of wind and precipitation data (complete)
- **marine_weather_daily.csv**: 2,056 days of wave height, wave period, swell data (31% null before Oct 2021 due to ERA5 ocean reanalysis gap)
- Coordinates: San Antonio, Chile (-33.5933, -71.6167)
- Weather data is the most impactful enrichment source: wind and wave conditions directly affect mooring operations, port entry, and can delay or extend service times.

### Vessel Characteristics
- **vessel_dimensions.csv**: LOA (length overall) for 338 of 1,550 vessels (22% coverage), derived by cross-matching IMO codes through imo_vessel_codes.csv (GitHub, Public Domain) and vessel_information_ais.csv (AIS educational dataset).
- Coverage is limited; fuller vessel databases (Datalastic, MarineTraffic, Kaggle Global Cargo Ships) require accounts or paid access.

### Reference Datasets
- **port_noumea_stopovers.csv**: ~50K port call records from Port of Noumea (2002-2017) with vessel IMO, tonnage, and routes. 35 vessels overlap with our fleet. Useful for methodology comparison.
- **ship_movement_2015_2020.zip / 2021_2022.zip**: Berth mooring dynamics from University of A Coruna (Zenodo, CC BY). 14 ships with length, beam, DWT, and movement measurements under various weather conditions. Reference for BAP research.

All sources are documented in `external_data/sources.md`.

## Project Structure

```
portsDFL/
  data/
    BBDD limpia(1).xlsx              # Source data (unchanged)
    Analisis semana 2 v3(1).xlsx     # Week 2 berth planning analysis
    propuesta v1(1).xlsx             # Berth schedule proposal
    stats plan 2025(1).xlsx          # 2025 planning statistics
    training_dataset.csv             # Generated training dataset (5,597 x 44)
    column_description.pdf           # Column definitions (EN/ES)
    project_description.pdf          # Project overview
  external_data/
    weather_daily.csv                # Daily weather (Open-Meteo)
    weather_hourly.csv               # Hourly weather (Open-Meteo)
    marine_weather_daily.csv         # Wave/swell data (Open-Meteo Marine)
    imo_vessel_codes.csv             # 64K vessel IMO registry (GitHub)
    vessel_information_ais.csv       # 10K vessels with LOA (AIS dataset)
    vessel_dimensions.csv            # 338 of our vessels with LOA (derived)
    port_noumea_stopovers.csv        # Port of Noumea call records (Zenodo)
    ship_movement_2015_2020.zip      # Berth dynamics dataset (Zenodo)
    ship_movement_2021_2022.zip      # Berth dynamics dataset (Zenodo)
    ship_movement_vessel_chars.csv   # 14 ships: length, beam, DWT (derived)
    sources.md                       # All sources documented
  data_pipeline/                     # (was src/) standalone data-build scripts
    build_clean_dataset.py           # Source-faithful clean dataset builder
    build_training_dataset.py        # Training dataset builder script
    port_regions.py                  # Port-to-region mapping dictionary
    download_external_data.py        # External data downloader (weather)
    generate_pdfs.py                 # PDF documentation generator
  docs/
    column_description.md            # Column definitions source (markdown)
    project_description.md           # This file (markdown)
    literature/                      # (was DFL_Port_Literature_Review/) lit review
    meetings/latex/                  # (was meetings/) DFL explainer figures
```

## Known Limitations

1. **No cargo volume data** in the main database. Container counts exist only for ~90 vessels in `stats plan 2025`. Draft difference (arrival vs departure) could proxy for cargo volume but is only available post-berthing.
2. **No crane/equipment data**. Cannot normalize service time by handling resources.
3. **Vessel dimensions (LOA, beam, DWT)** available for only 22% of vessels from free sources. Fuller coverage requires Kaggle account (Global Cargo Ships dataset) or paid databases (Datalastic, MarineTraffic).
4. **30% missing shipping line/service** data for non-container types. These are structurally missing (bulk/tanker vessels don't use liner services).
5. **Weather data not yet joined** to training dataset. Available in external_data/ for integration.
6. **Marine wave data has gaps** before October 2021 (31% null) due to ERA5 ocean reanalysis coverage limitations.
