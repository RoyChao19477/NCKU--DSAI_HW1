# DSAI-HW-2021

It seems that the only data we will get when ranking is "Operating Reserve (**OR**)". Therefore, TML and DL methods based on full version data will not be adopted in Ranking Algorithm.

# How to use?
- Use `python app.py` directly. The prediction will be saved as `submission.csv`
- Or use `python app.py --output "predict_name.csv"` to specify a special output filename.

# Causality
For `app.py` algorithm, we will use the OR of day_(N) to predic the OR of day_(N+1). Therefore, the model is causal.

## More experiments results:
- Some **data analysis** and **preprocessing** were adoped in folder `\1_traditinal_ML`
- Traditional Machine Learning including **K-Neighbor**, **SupportVevtor**, **NuSVR**, **DecisionTree**, **RandomForest**, **Adaboost**, **GradientBoosting**, and **ARIMA** approaches were structured in folder `\1_Traditional_ML`.
- Deep Learning approaches including **CNN**, **DNN**, **RNN**, **LSTM**, **BLSTM**, **CRNN**, **FCN** (Fully Convolutional Network), **Transformer** were structured in folder `\2_DL`.
