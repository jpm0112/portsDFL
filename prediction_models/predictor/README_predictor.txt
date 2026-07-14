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

TWO WAYS TO USE IT
------------------
  A) WEEKLY SCHEDULES (Run_Weekly_Predictions.bat) -- drop the weekly Excel files you
     receive into the weekly_schedules\ folder and predict them all at once. The tool
     fills in the technical fields (tonnage, drafts, vessel type, berth...)
     automatically from each vessel's port history.
  B) SINGLE CSV (Run_Single_CSV_Prediction.bat) -- you provide every field yourself in one CSV.

A) WEEKLY SCHEDULES
-------------------
  1. Leave your weekly files in the  weekly_schedules\  folder (.xlsx or .csv, any name,
     as many as you want -- e.g. semana1.xlsx, semana2.xlsx).
     Each file needs at least these columns (extra columns are ignored):
       Nave      - vessel name (used to look up the vessel's history)
       E.T.A.    - estimated arrival, YYYY-MM-DD HH:MM
       Agencia   - agency name (e.g. ULTRAMAR)
       Carga     - cargo type (only used if the vessel is new to the port)
  2. Double-click  Run_Weekly_Predictions.bat
     - The first run sets up a local environment (a few minutes). Later runs are instant.
  3. Results: one  <name>_predictions.csv  per weekly file, in the
     predictions\  folder. Columns added to your original ones:
       rf, xgb, lgbm   - predicted hours at berth per model (rf is the best)
       ensemble_mean   - average of the three
       matched_history - True if the vessel was found in the port history;
                         False means its technical fields were estimated and
                         the prediction is rougher
       notes           - any caveats for that row

  NEEDED FILE: this mode reads the vessel-history workbook
    ..\..\data\BBDD limpia(1).xlsx   (sheet "Resume Naves Comerciales (4)")
  so it works out of the box inside the full repo. If you copy this folder
  elsewhere, bring that file along and point to it with:
    python predict_weeks.py --history "C:\path\to\BBDD limpia(1).xlsx"

  Caveat: schedules only carry the E.T.A., so it is used as the berthing time
  too (the models were trained on real berthing times). Predictions are a bit
  rougher because of this; every row's notes column records it.

B) SINGLE CSV
-------------
  1. Double-click  Run_Single_CSV_Prediction.bat
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
