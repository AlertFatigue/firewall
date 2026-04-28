Fixed thesis pipeline - run order
=================================

1) Optional: create a smaller filtered EVE file
-----------------------------------------------
python extract.py \
  --input ./trueHeartbleedLogs/eve.json \
  --output ./filtered_eve.json \
  --alert-contains Heartbleed \
  --max-normal-events 15000

For a specific IP/port attack selection you can also add:
  --attack-src-ip 172.16.0.1 --attack-dest-ip 192.168.10.51 --attack-dest-port 444

2) Sanity-test the data before calling Gemini
---------------------------------------------
python testLoader.py --input ./filtered_eve.json

3) Run the labeling pipeline
----------------------------
With Gemini:
  export GEMINI_API_KEY='your_key_here'
  python main.py --input ./filtered_eve.json --dataset heartbleed --output-dir ./outputs

Without Gemini, for debugging only:
  python main.py --input ./filtered_eve.json --dataset heartbleed --output-dir ./outputs --no-llm

4) Train CatBoost and measure latency
-------------------------------------
python trainModel.py --dataset heartbleed --output-dir ./outputs

This saves:
  outputs/heartbleed_catboost_model.cbm
  outputs/heartbleed_metrics.json
  outputs/heartbleed_feature_importance.csv

5) Build validation file with IP/port fields
--------------------------------------------
python validation.py --input ./filtered_eve.json --dataset heartbleed --output-dir ./outputs

6) Evaluate against ground truth, if you have the CIC ground-truth CSV
---------------------------------------------------------------------
python evaluate.py \
  --dataset heartbleed \
  --output-dir ./outputs \
  --groundtruth-file ./wednesdayGroundTruth.csv \
  --attack-label Heartbleed

Important fixes compared with the old files
===========================================
- HDBSCAN outliers are no longer treated as one single exemplar.
- testLoader now uses exemplars_df when creating exemplar contexts.
- main.py uses the correct outlier_contexts for outlier labels.
- Output filenames are controlled by --dataset, so heartbleed_train.csv and heartbleed_test.csv are produced consistently.
- extract.py is no longer locked to one hardcoded Heartbleed IP/port.
- trainModel.py saves the CatBoost model and measures latency/events per second.
- validation.py and evaluate.py are less hardcoded and safer for thesis evaluation.
