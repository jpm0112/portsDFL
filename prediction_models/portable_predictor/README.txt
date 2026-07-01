VESSEL SERVICE-TIME PREDICTOR
=============================

Predicts how many hours a vessel will spend at berth, using three trained models
(rf, xgb, lgbm). rf is the most accurate.

WHAT YOU NEED
-------------
  * Windows
  * Python 3.11 installed  (https://www.python.org/downloads/ -- during install,
    tick "Add python.exe to PATH")
  * Internet, the FIRST time only (to download the dependencies)

No GPU is required. The models run on the CPU and predict in seconds.

HOW TO USE
----------
  1. Double-click  Predict.bat
     - The first run sets up a local environment (a few minutes). Later runs are instant.
  2. It lists the CSV files in this folder and asks which one to use
     (if there is only one, it picks it automatically).
  3. It asks which models to run -- press ENTER for rf (the most accurate).
     Type 'all' to run all three, or list them (e.g. rf,xgb).
  4. It writes  <yourfile>_predictions.csv  next to your input and shows the
     predictions on screen. If you run more than one model, the "ensemble_mean"
     column is their average.

YOUR INPUT CSV
--------------
Put your vessels in  vessels.csv  (it starts as a copy you can edit / replace).
sample_vessels.csv is a read-only reference so you can always see the exact
column names; the tool ignores it when choosing what to predict.

Required fields:
  Sitio, Tipo nave, arrival_datetime, berthing_datetime, Puerto origen,
  Puerto destino, Agencia, Linea naviera, Servicio, TRG,
  draft_arrival_bow, draft_arrival_stern
Optional: draft_departure_bow, draft_departure_stern (improve one feature),
and vessel_id (added automatically if you leave it out).

Dates use the format  YYYY-MM-DD HH:MM  (e.g. 2024-06-15 14:00).

NOTES
-----
  * Predictions are in hours and are ballpark: typical error is about 12 hours.
  * If a port / line / agency was never seen in training, the tool says so and
    falls back to that field's average -- the prediction is rougher for that row.
