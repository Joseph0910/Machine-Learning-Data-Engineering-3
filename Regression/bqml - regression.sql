/***
model dataset
WITH
  raw AS (
  SELECT
    date,
    close,
    LAG(close, 1) OVER(ORDER BY date) AS min_1_close,
    LAG(close, 2) OVER(ORDER BY date) AS min_2_close,
    LAG(close, 3) OVER(ORDER BY date) AS min_3_close,
    LAG(close, 4) OVER(ORDER BY date) AS min_4_close
  FROM
    `ai4f.AAPL10Y`
  ORDER BY
    date DESC ),
  raw_plus_trend AS (
  SELECT
    date,
    close,
    min_1_close,
    IF (min_1_close - min_2_close > 0, 1, -1) AS min_1_trend,
    IF (min_2_close - min_3_close > 0, 1, -1) AS min_2_trend,
    IF (min_3_close - min_4_close > 0, 1, -1) AS min_3_trend
  FROM
    raw ),
  ml_data AS (
  SELECT
    date,
    close,
    min_1_close AS day_prev_close,
    IF (min_1_trend + min_2_trend + min_3_trend > 0, 1, -1) AS trend_3_day
  FROM
    raw_plus_trend )
SELECT
  *
FROM
  ml_data


***/


/***
build model
CREATE OR REPLACE MODEL `ai4f.aapl_model`
OPTIONS
  ( model_type='linear_reg',
    input_label_cols=['close'],
    data_split_method='seq',
    data_split_eval_fraction=0.3,
    data_split_col='date') AS
SELECT
  date,
  close,
  day_prev_close,
  trend_3_day
FROM
  `ai4f.model_data`


***/

/***
evaluate model performance 
SELECT * FROM ML.EVALUATE(MODEL `ai4f.aapl_model`)

***/

/***
make predictions
SELECT
  *
FROM
  ml.PREDICT(MODEL `ai4f.aapl_model`,
    (
    SELECT
      *
    FROM
      `ai4f.model_data`
    WHERE
      date >= '2019-01-01') )

***/



