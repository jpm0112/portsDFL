"""
Download external datasets for BAP service time prediction enrichment.

Downloads:
  1. Historical daily weather (Open-Meteo)
  2. Historical hourly weather (Open-Meteo) - wind focus
  3. Marine weather - waves/swell (Open-Meteo Marine API)

All data for San Antonio, Chile (-33.5933, -71.6167).
Output saved to external_data/ folder.
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime

# San Antonio port coordinates (decimal degrees; negative = south/west).
LAT = -33.5933
LON = -71.6167

# Data period matching vessel records (ISO date strings: YYYY-MM-DD).
START_DATE = "2020-01-01"
END_DATE = "2025-08-17"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external_data")


def fetch_daily_weather():
    """
    Fetch daily weather from Open-Meteo Historical API.

    Variables: temperature, precipitation, wind speed/gusts/direction.
    Output: weather_daily.csv
    """
    print("Fetching daily weather data...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        # Open-Meteo expects the "daily" parameter as a comma-separated string.
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
        ]),
        "timezone": "America/Santiago",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["daily"])
    # Open-Meteo calls the date column "time"; rename it for clarity.
    df.rename(columns={"time": "date"}, inplace=True)

    filepath = os.path.join(OUTPUT_DIR, "weather_daily.csv")
    df.to_csv(filepath, index=False)
    print(f"  Saved {filepath} ({len(df)} days, {df.columns.tolist()})")
    return df


def fetch_hourly_weather():
    """
    Fetch hourly weather from Open-Meteo Historical API.

    Focused on wind (key factor for port operations).
    Downloads in yearly chunks to avoid API limits.
    Output: weather_hourly.csv
    """
    print("Fetching hourly weather data (in yearly chunks)...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    all_dfs = []

    # Split into yearly chunks to avoid large request limits. ISO date strings
    # compare correctly because YYYY-MM-DD sorts the same alphabetically as
    # chronologically.
    years = range(2020, 2026)
    for year in years:
        chunk_start = f"{year}-01-01"
        # Cap the end at END_DATE so the last year stops at the real cutoff.
        chunk_end = min(f"{year}-12-31", END_DATE)
        if chunk_start > END_DATE:
            break

        print(f"  Fetching {year}...")
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": ",".join([
                "temperature_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_gusts_10m",
                "wind_direction_10m",
            ]),
            "timezone": "America/Santiago",
        }

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["hourly"])
        all_dfs.append(df)
        time.sleep(1)  # polite delay between requests so we don't hammer the API

    result = pd.concat(all_dfs, ignore_index=True)
    # Hourly timestamp column "time" -> "datetime" (it has hours, not just dates).
    result.rename(columns={"time": "datetime"}, inplace=True)

    filepath = os.path.join(OUTPUT_DIR, "weather_hourly.csv")
    result.to_csv(filepath, index=False)
    print(f"  Saved {filepath} ({len(result)} hours)")
    return result


def fetch_marine_weather():
    """
    Fetch marine weather from Open-Meteo Marine API.

    Variables: wave height, wave period, wave direction, swell.
    Downloads in yearly chunks.
    Output: marine_weather_daily.csv
    """
    print("Fetching marine weather data (in yearly chunks)...")

    url = "https://marine-api.open-meteo.com/v1/marine"
    all_dfs = []

    years = range(2020, 2026)
    for year in years:
        chunk_start = f"{year}-01-01"
        chunk_end = min(f"{year}-12-31", END_DATE)
        if chunk_start > END_DATE:
            break

        print(f"  Fetching {year}...")
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "daily": ",".join([
                "wave_height_max",
                "wave_direction_dominant",
                "wave_period_max",
                "swell_wave_height_max",
                "swell_wave_direction_dominant",
                "swell_wave_period_max",
            ]),
            "timezone": "America/Santiago",
        }

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Marine API returns DAILY aggregates (key is "daily", unlike the hourly fn above).
        df = pd.DataFrame(data["daily"])
        all_dfs.append(df)
        time.sleep(1)

    result = pd.concat(all_dfs, ignore_index=True)
    result.rename(columns={"time": "date"}, inplace=True)

    filepath = os.path.join(OUTPUT_DIR, "marine_weather_daily.csv")
    result.to_csv(filepath, index=False)
    print(f"  Saved {filepath} ({len(result)} days, {result.columns.tolist()})")
    return result


def generate_sources_file():
    """
    Generate sources.md documenting all data sources, URLs, and access dates.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    content = f"""# External Data Sources

Data downloaded on: {today}
Location: San Antonio, Chile (lat={LAT}, lon={LON})
Period: {START_DATE} to {END_DATE}

---

## Downloaded Datasets

### 1. weather_daily.csv - Historical Daily Weather
- **Source**: Open-Meteo Historical Weather API
- **URL**: https://open-meteo.com/en/docs/historical-weather-api
- **API Endpoint**: https://archive-api.open-meteo.com/v1/archive
- **License**: CC BY 4.0 (free for non-commercial and commercial use)
- **Variables**: temperature_2m_max/min/mean, precipitation_sum, rain_sum, wind_speed_10m_max, wind_gusts_10m_max, wind_direction_10m_dominant
- **Resolution**: Daily aggregates, 9km spatial resolution
- **Data Basis**: ERA5 reanalysis (ECMWF)
- **Use Case**: Join to training dataset by date to add weather context. Wind and precipitation may affect mooring operations and service times.

### 2. weather_hourly.csv - Historical Hourly Weather
- **Source**: Open-Meteo Historical Weather API
- **URL**: https://open-meteo.com/en/docs/historical-weather-api
- **API Endpoint**: https://archive-api.open-meteo.com/v1/archive
- **License**: CC BY 4.0
- **Variables**: temperature_2m, precipitation, wind_speed_10m, wind_gusts_10m, wind_direction_10m
- **Resolution**: Hourly, 9km spatial resolution
- **Use Case**: Match to vessel arrival/mooring timestamps for hour-specific weather conditions during port operations.

### 3. marine_weather_daily.csv - Marine/Wave Weather
- **Source**: Open-Meteo Marine Weather API
- **URL**: https://open-meteo.com/en/docs/marine-weather-api
- **API Endpoint**: https://marine-api.open-meteo.com/v1/marine
- **License**: CC BY 4.0
- **Variables**: wave_height_max, wave_direction_dominant, wave_period_max, swell_wave_height_max, swell_wave_direction_dominant, swell_wave_period_max
- **Resolution**: Daily aggregates
- **Data Basis**: ERA5 ocean wave reanalysis
- **Use Case**: Wave conditions directly affect port entry restrictions and mooring safety. High waves may delay berthing and extend service times.

---

## Additional Sources Identified (Not Downloaded - Manual Access Required)

### Vessel Characteristics Databases
These sources can provide LOA (length overall), beam, DWT, and build year for the 1,550 unique vessels in our dataset:

#### JobMarineMan Ship Database
- **URL**: https://jobmarineman.com/ships/
- **Access**: Free, search by IMO number or vessel name
- **Data**: IMO, vessel name, flag, type, year of build, DWT, GT, dimensions
- **Limitation**: Web interface only, no bulk API. Manual search needed per vessel.

#### Equasis (European Quality Shipping Information System)
- **URL**: https://www.equasis.org/
- **Access**: Free registration required
- **Data**: IMO, vessel details, classification, inspections, dimensions
- **Limitation**: Web interface, limited to individual searches. Run by EMSA.

#### MarineTraffic
- **URL**: https://www.marinetraffic.com/
- **Access**: Free for basic vessel info, API requires paid subscription
- **Data**: 550,000+ vessels with LOA, beam, DWT, GT, draught, year built

#### Datalastic Maritime Database
- **URL**: https://datalastic.com/maritime-database/
- **Access**: Basic CSV download available (check free tier limits)
- **Data**: 600,000+ vessels with dimensions, tonnage, capacity
- **Format**: CSV download or API

### Port Statistics

#### UNCTAD Port Statistics
- **URL**: https://unctadstat.unctad.org/datacentre/
- **Access**: Free, downloadable
- **Data**: Container port throughput (TEU), port call arrivals, liner connectivity index
- **Use Case**: Port-level trends and benchmarking

#### Puerto San Antonio Official Data
- **URL**: https://www.puertosanantonio.com/
- **Access**: Free, public operational statistics
- **Data**: Cargo transfer stats, TEU movements, real-time vessel schedules
- **Real-time tracking**: https://eps-hmi.pcspuertosanantonio.cl/information/ship-trail/public

#### DIRECTEMAR (Chilean Maritime Authority)
- **URL**: https://www.directemar.cl/
- **Access**: Free, public
- **Data**: Maritime regulations, port authority procedures, vessel documentation

### AIS / Vessel Tracking (Historical)

#### MarineCadastre (NOAA/USCG)
- **URL**: https://hub.marinecadastre.gov/pages/vesseltraffic
- **Access**: Free, open data
- **Coverage**: U.S. waters only (not applicable to San Antonio)
- **Note**: Good methodological reference for AIS data processing

#### AISHub
- **URL**: https://www.aishub.net/
- **Access**: Free (requires sharing own AIS feed)
- **Coverage**: Global including South America
- **Data**: Real-time AIS positions, limited historical archive

### Kaggle Datasets

#### Maritime Port Performance Dataset
- **URL**: https://www.kaggle.com/datasets/jeleeladekunlefijabi/maritime-port-performance-dataset
- **Access**: Free with Kaggle account

#### Global Cargo Ships Dataset
- **URL**: https://www.kaggle.com/datasets/ibrahimonmars/global-cargo-ships-dataset
- **Access**: Free with Kaggle account
- **Data**: May contain vessel dimensions (LOA, beam, DWT)

#### Port of Los Angeles Shipment Dataset
- **URL**: https://www.kaggle.com/datasets/mikoajfish99/port-of-los-angeles
- **Access**: Free with Kaggle account
- **Note**: Different port but similar operational data structure
"""

    filepath = os.path.join(OUTPUT_DIR, "sources.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Saved {filepath}")


def main():
    print("=" * 60)
    print("External Data Download for BAP Service Time Prediction")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Wrapping each fetch separately means one failed download won't stop the others.
    # 1. Daily weather
    try:
        fetch_daily_weather()
    except Exception as e:
        print(f"  ERROR fetching daily weather: {e}")

    # 2. Hourly weather
    try:
        fetch_hourly_weather()
    except Exception as e:
        print(f"  ERROR fetching hourly weather: {e}")

    # 3. Marine weather
    try:
        fetch_marine_weather()
    except Exception as e:
        print(f"  ERROR fetching marine weather: {e}")

    # 4. Sources documentation
    print("\nGenerating sources documentation...")
    generate_sources_file()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            print(f"  {f}: {size / (1024*1024):.1f} MB")
        else:
            print(f"  {f}: {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
