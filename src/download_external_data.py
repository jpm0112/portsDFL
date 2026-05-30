"""
Download external datasets for BAP service time prediction enrichment.

Downloads:
  1. Historical daily weather (Open-Meteo)
  2. Historical hourly weather (Open-Meteo) - wind focus
  3. Marine weather - waves/swell (Open-Meteo Marine API)

All data for San Antonio, Chile (-33.5933, -71.6167).
Output saved to external_data/ folder.
"""

# `import X` loads a module so we can use its functions as `X.something(...)`.
import os        # file/folder paths and directory operations
import sys       # access to interpreter (imported but not used here)
import json      # JSON encode/decode (imported but not used here)
import time      # used for time.sleep() to pause between API calls
import requests  # third-party library for making HTTP requests (may need `pip install requests`)
import pandas as pd  # `as pd` gives the module a short nickname; pandas handles tables (DataFrames)
from datetime import datetime  # import just the `datetime` class from the datetime module

# These ALL-CAPS module-level names are constants (Python has no true constants;
# uppercase is just a convention meaning "don't change this").
# San Antonio port coordinates (decimal degrees; negative = south/west).
LAT = -33.5933
LON = -71.6167

# Data period matching vessel records (ISO date strings: YYYY-MM-DD).
START_DATE = "2020-01-01"
END_DATE = "2025-08-17"

# Build the output folder path relative to THIS script's location so it works
# no matter where you run the script from.
# __file__ = path to this .py file; abspath -> full path; dirname -> its folder;
# ".." means "go up one level"; os.path.join joins parts with the OS path separator.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external_data")


# `def name():` defines a function. The triple-quoted text right below is a
# "docstring" documenting what the function does. This function takes no
# arguments and returns a pandas DataFrame (the table it built).
def fetch_daily_weather():
    """
    Fetch daily weather from Open-Meteo Historical API.

    Variables: temperature, precipitation, wind speed/gusts/direction.
    Output: weather_daily.csv
    """
    print("Fetching daily weather data...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    # A dict (key: value pairs in {}) of query parameters sent in the URL.
    # requests turns this into "?latitude=...&longitude=..." automatically.
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        # ",".join([...]) glues the list items into one comma-separated string,
        # which is the format Open-Meteo expects for the "daily" parameter.
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

    # requests.get sends an HTTP GET; timeout=60 aborts if the server takes >60s.
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()  # raise an error if the HTTP status was 4xx/5xx (so we don't parse bad data)
    data = response.json()       # parse the JSON response body into a Python dict

    # The API returns a dict where data["daily"] is itself a dict of column-name -> list.
    # pd.DataFrame builds a table from that, one row per day.
    df = pd.DataFrame(data["daily"])
    # .rename changes a column name; inplace=True modifies df directly (no copy returned).
    # Open-Meteo calls the date column "time"; we rename it to "date" for clarity.
    df.rename(columns={"time": "date"}, inplace=True)

    filepath = os.path.join(OUTPUT_DIR, "weather_daily.csv")
    df.to_csv(filepath, index=False)  # index=False = don't write the row-number column
    # f-strings (f"...") let you embed values inside {} directly in the string.
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
    all_dfs = []  # an empty list; we'll append one DataFrame per year, then combine them

    # Split into yearly chunks to avoid large request limits.
    # range(2020, 2026) yields 2020,2021,...,2025 (the END value is EXCLUDED).
    years = range(2020, 2026)
    for year in years:
        # Build that year's date range. These are strings; comparing ISO date
        # strings (e.g. "2025-12-31" > "2025-08-17") sorts correctly because the
        # YYYY-MM-DD format is alphabetically ordered the same as chronologically.
        chunk_start = f"{year}-01-01"
        # min() caps the end at END_DATE so the last year stops at the real cutoff.
        chunk_end = min(f"{year}-12-31", END_DATE)
        if chunk_start > END_DATE:  # skip any year that starts after our data window
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

        df = pd.DataFrame(data["hourly"])  # one year's hourly rows as a table
        all_dfs.append(df)  # add this year's table to our list
        time.sleep(1)  # polite delay (1 second) between requests so we don't hammer the API

    # pd.concat stacks the list of yearly DataFrames into one big table.
    # ignore_index=True renumbers the rows 0..N instead of repeating each chunk's 0-based index.
    result = pd.concat(all_dfs, ignore_index=True)
    # Hourly data's timestamp column "time" -> "datetime" (it has hours, not just dates).
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

        # Marine API returns DAILY aggregates (note key is "daily", unlike the hourly fn above).
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
    # datetime.now() = current date/time; .strftime formats it as a string.
    # "%Y-%m-%d" -> e.g. "2026-05-30" (4-digit year, 2-digit month, 2-digit day).
    today = datetime.now().strftime("%Y-%m-%d")

    # A multi-line f-string (triple quotes). The {today}, {LAT}, etc. are filled in.
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
    # `with open(...) as f:` opens the file and GUARANTEES it is closed afterward,
    # even if an error happens inside the block. "w" = write (overwrites existing).
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Saved {filepath}")


def main():
    # "=" * 60 repeats the "=" character 60 times to draw a separator line.
    print("=" * 60)
    print("External Data Download for BAP Service Time Prediction")
    print("=" * 60)

    # Create the output folder; exist_ok=True means "don't error if it already exists".
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # try/except runs the risky code, and if it raises an error, jumps to except
    # instead of crashing. `as e` captures the error object so we can print it.
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
    # os.listdir gives the file names in a folder; loop over each and report its size.
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)  # size in bytes
        if size > 1024 * 1024:  # bigger than 1 MB -> show in MB
            # In an f-string, {value:.1f} formats the number to 1 decimal place.
            print(f"  {f}: {size / (1024*1024):.1f} MB")
        else:
            print(f"  {f}: {size / 1024:.0f} KB")  # otherwise show in KB (0 decimals)


# This block runs ONLY when you execute this file directly (python download_external_data.py).
# If another file imports this one, __name__ won't be "__main__", so main() won't auto-run.
if __name__ == "__main__":
    main()
