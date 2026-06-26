# Vessel service-time predictor

Feed it a CSV of **raw vessel data** and it returns each model's predicted **berth
service time (hours)** plus an ensemble mean. It auto-engineers the model features from
the raw fields and reuses the trained artifacts in `../artifacts` — no retraining, no
optimizer/DFL involved. `predict.py` is the tool; `sample_vessels.csv` is the template.

## Run

```bash
conda activate portsdfl

# Needs trained models in ../artifacts first (from scripts/train_all.py, e.g. on ASAX —
# see ../../hpc/README.md). Then, from this folder:
python predict.py                                    # runs on sample_vessels.csv
python predict.py --input my_vessels.csv --output preds.csv
python predict.py --input my_vessels.csv --models xgb,lgbm   # a subset of models
```

Output `predictions.csv` = your input rows + one predicted-hours column per model +
`ensemble_mean`.

## Input columns

One row per vessel — copy `sample_vessels.csv` and edit it. Categorical values are
matched against the training data; an **unlisted value still works**, it just falls back
to that field's overall average (so the closed lists below matter most for `Sitio`,
`Tipo nave`, and `Agencia`).

| Column | What to put |
|---|---|
| `Sitio` | Berth — one of: `C1`, `C2`, `Sitio 1`, `Sitio 2`, `Sitio 3`, `Sitio 4/5`, `Sitio 8`, `Sitio 9` |
| `Tipo nave` | Raw vessel type — see the full list below |
| `arrival_datetime` | When the vessel arrived, `YYYY-MM-DD HH:MM` (e.g. `2025-01-10 06:00`) |
| `berthing_datetime` | First mooring / berthing time, same format |
| `Puerto origen` | Origin port — examples below |
| `Puerto destino` | Destination port — examples below |
| `Agencia` | Agency as raw `RUT - NAME` — see the full list below |
| `Línea naviera` | Shipping line — examples below; leave blank → `NON-LINER` |
| `Servicio` | Service/route — examples below; leave blank → `NO SERVICE` |
| `TRG` | Gross tonnage, a number (data range ~500 – 155,000) |
| `draft_arrival_bow`, `draft_arrival_stern` | Arrival draft in metres, bow & stern (~2.5 – 14.6) |
| `draft_departure_bow`, `draft_departure_stern` | Departure draft in metres (same range). **Optional** — omit and `Calado diff` is set to 0 |

### `Tipo nave` — valid raw types (grouped by what the model sees)

- **Container:** `Contenedor`
- **Dry Bulk:** `Carga Seca Granel`, `Mineral/Granel/Petrolero`
- **Vehicle Carrier:** `Autero`, `Autotrasbordo`
- **Liquid Bulk:** `Transporte Quimico`, `Transporte Liquido`, `Transporte de Asfalto`, `Petrolero`
- **General Cargo:** `Tradicional`, `Carga de Proyecto`, `Chipero`, `Refrigerado`, `Otros`
- **Passenger:** `Pasajeros`
- **Other:** `Nave Armada`

An unlisted vessel type raises an error rather than guessing.

### `Agencia` — the agencies in the data (use the exact string)

`80992000-3 - ULTRAMAR` · `82728500-5 - IAN TAYLOR` · `96566940-k - AGUNSA` ·
`80925100-4 - SOMARCO` · `96707720-8 - MSC` · `78986820-4 - B&M` · `80010900-0 - AGENTAL` ·
`78610880-2 - INCHCAPE` · `76902117-5 - MTA` · `91256000-7 - A.J.BROOM` · `other_agencies`

### Ports / lines / services — common values (any value accepted; unknown → average)

- **`Puerto origen`** (41 in the data): `CALLAO`, `MEJILLONES`, `LIRQUEN`, `IQUIQUE`,
  `SAN LORENZO`, `QUINTERO`, `ANTOFAGASTA`, `ARICA`, `ROSARIO`, `SANTOS - SP`,
  `RIO GRANDE - RS`, … or `other_origins` for a rare one.
- **`Puerto destino`** (53): `CALLAO`, `SAN VICENTE`, `HONG KONG`, `CORONEL`, `BALBOA`,
  `MEJILLONES`, `SAN ANTONIO - CL`, `MANZANILLO - MX`, `PUNTA ARENAS`, `LIRQUEN`, `ULSAN`,
  … or `other_destinations`.
- **`Línea naviera`** (24): `MAERSK LINE`, `HAPAG-LLOYD CHILE SPA`, `MSC`, `COSCO GROUP`,
  `TRANSMARES`, `CMA CGM`, `EVERGREEN`, `HYUNDAI GLOVIS INC.`, `ULTRAGAS`, `NYK`, `ZIM`,
  … or `NON-LINER`.
- **`Servicio`** (35): `CAR CARRIERS`, `SERVICIO CONOSUR / ABAC`, `Eurosal`, `WSA3`,
  `699/ AC3`, `Andes`, `84Q ATACAMA FEEDER`, `U4Gabac`, `WSA`, `CABOTAJE SUR`, `CABOTAJE`,
  … or `NO SERVICE`.

## Caveats

The exact training pipeline for two features isn't in the repo, so they're reverse-engineered:

- **`covid_era`** is derived from `arrival_datetime` with approximate cutoffs (`pre` if
  before 2020-03-11, `during` if before 2023-05-05, else `post`). Negligible — any recent
  vessel is `post`.
- **`Calado diff`** = arrival − departure draft, so it needs the departure drafts (known
  only after the call). Omit them for a not-yet-departed vessel and it defaults to 0; it's
  one of 17 features, so a rough value mostly affects the tail of the estimate.

Everything else (the categoricals, `TRG`, arrival draft, and the six cyclical date
encodings) is reproduced exactly.
