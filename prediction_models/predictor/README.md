# Vessel service-time predictor

A small self-contained tool: feed it a CSV of vessels and it returns each model's
predicted **berth service time (hours)** plus an ensemble mean. It reuses the trained
artifacts in `../artifacts` — no retraining, no optimizer/DFL involved.

## Run

```bash
# from this folder, with the project env active (has torch/xgboost/lightgbm/etc.):
python predict.py                                    # runs on the bundled sample
python predict.py --input my_vessels.csv --output preds.csv
python predict.py --input my_vessels.csv --models xgb,lgbm   # a subset of models
```

Needs trained artifacts in `../artifacts` (produced by `scripts/train_all.py`, e.g. on
ASAX — see `../../hpc/README.md`). Output `predictions.csv` = your input columns + one
predicted-hours column per model + `ensemble_mean`.

## Raw-input mode (best-effort)

If you have *raw* vessel fields instead of the engineered columns, use
`predict_from_raw.py` — it auto-derives the 17 features (see `sample_raw_vessels.csv`)
and then predicts:

```bash
python predict_from_raw.py                                   # runs on sample_raw_vessels.csv
python predict_from_raw.py --input my_raw_vessels.csv --output preds.csv
```

Required raw columns: `Sitio`, `Tipo nave` (raw type, e.g. `Contenedor`),
`arrival_datetime`, `berthing_datetime` (first mooring), `Puerto origen`,
`Puerto destino`, `Agencia` (raw `RUT - NAME`), `Línea naviera`, `Servicio`, `TRG`,
`draft_arrival_bow`, `draft_arrival_stern`. Optional: `draft_departure_bow` +
`draft_departure_stern` (for `Calado diff`; omitted → `Calado diff = 0`).

⚠️ Best-effort: `covid_era` / `Calado diff` use the reverse-engineered logic below, and
rare ports/lines that the training data bucketed (`other_destinations` / `other_liners`)
can't be reproduced, so such values fall back to the model's prior. For faithful inputs,
engineer the features yourself and use `predict.py`.

## Input format (faithful mode)

`sample_vessels.csv` is a ready-to-edit template (5 real rows) for `predict.py`. Replace
the values with your vessels; keep the column names. The 17 columns:

**Categoricals** — provide the value as it appears in the data (rare values are bucketed,
e.g. `other_destinations`, `other_liners`; an unseen value still works but falls back to
a sensible average):

| Column | What | Notes |
|---|---|---|
| `Sitio` | berth label | e.g. `Sitio 1`, `Sitio 4/5` |
| `Tipo nave (agrupado)` | vessel type group | one of: Container, Dry Bulk, Vehicle Carrier, Liquid Bulk, General Cargo, Passenger, Other |
| `Puerto origen` / `Puerto destino` | port names | rare ports appear as `other_destinations` |
| `Agencia` | agency, **raw** `RUT - NAME` | kept as-is, e.g. `78610880-2 - INCHCAPE` |
| `Línea naviera` | shipping line | rare → `other_liners`; missing → `NON-LINER` |
| `Servicio` | service/route | missing → `NO SERVICE` |
| `covid_era` | `pre` / `during` / `post` | see formula below |

**Numerics** — if you have *raw* vessel data, compute these (formulas verified against the
training data):

```
TRG            = gross tonnage (as-is)
Calado arribo  = (arrival bow draft + arrival stern draft) / 2
Calado diff    = Calado arribo  -  (departure bow draft + departure stern draft) / 2

covid_era      = "pre"    if arrival_date <  2020-03-11
                 "during" if arrival_date <  2023-05-05
                 "post"   otherwise

# from the berthing (first-mooring) datetime t:
hour  = t.hour            # 0..23     atraque_hour_sin = sin(2*pi*hour/24),  _cos = cos(2*pi*hour/24)
dow   = t.weekday         # Mon=0..6  atraque_dayofweek_sin = sin(2*pi*dow/7),   _cos = cos(2*pi*dow/7)
month = t.month           # 1..12     atraque_month_sin = sin(2*pi*month/12),    _cos = cos(2*pi*month/12)
```

## Caveats (read before trusting `covid_era` / `Calado diff`)

The exact feature pipeline that built the training CSV is **not in the repo**, so two
columns are reverse-engineered:

- **`covid_era`** cutoff dates are approximate (the boundaries match the data ~99.8%).
  Negligible in practice — any present-day vessel is just `post`.
- **`Calado diff`** = arrival minus departure draft, so it needs the **departure** drafts,
  which are only known *after* the call. For a vessel that hasn't left yet you can't
  compute it exactly — set it to `0` (or a typical value) if unknown; it's one of 17
  features, so a rough value mostly affects the tail of the estimate.

Everything else (the categoricals, `TRG`, `Calado arribo`, and all six `atraque_*` cyclical
encodings) reproduces the training features exactly.
