# DSAI-HW-2021

It seems that the only data we will get when ranking is "Operating Reserve (**OR**)". Therefore, TML and DL methods based on full version data will not be adopted in Ranking Algorithm.

## How to use?
- Use `python app.py` directly. The prediction will be saved as `submission.csv`
- Or use `python app.py --output "predict_name.csv"` to specify a special output filename.


## Detail of app.py
The full version of `app.py` can be found in `1_model_selection.ipynb`, `2_1days_version.ipynb`, and `3_2days_version.ipynb`
- GradientBoostingRegression and 1-day preview approaches are used in `app.py`.

## Data Resources:
- **(2021/01/01 ~ 2022/01/31)** Taiwan Power Company_Past Power Supply and Demand Information (https://data.gov.tw/dataset/19995) was pre-download in `data/0_raw_elec.csv`
- **(2022/01/01 ~ )** Taiwan Power Company_Daily peak standby capacity rate for this year (https://data.gov.tw/dataset/25850)
will be download while exec.
- `app.py` is a causal model, which means it will predict future information without peeking. Details are in "Causality".

## Causality
For `app.py` algorithm, we will use the **OR of day_(N)** to predic the **OR of day_(N+1)**. Therefore, the model is causal.

## More experiments results:
- Some **data analysis** and **preprocessing** were adoped in folder `\1_traditinal_ML`
- Traditional Machine Learning including **K-Neighbor**, **SupportVevtor**, **NuSVR**, **DecisionTree**, **RandomForest**, **Adaboost**, **GradientBoosting**, and **ARIMA** approaches were structured in folder `\1_Traditional_ML`.
- Deep Learning approaches including **CNN**, **DNN**, **RNN**, **LSTM**, **BLSTM**, **CRNN**, **FCN** (Fully Convolutional Network), **Transformer** were structured in folder `\2_DL`.
