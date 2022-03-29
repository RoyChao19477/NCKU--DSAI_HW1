import matplotlib
import numpy as np
import pandas as pd
import sklearn
import scipy
import seaborn as sns
import requests
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV                     
from sklearn.ensemble import GradientBoostingRegressor

sns.set( style="ticks" )
np.random.seed = 999



# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.

    df_raw = pd.read_csv("data/0_raw_elec.csv")     # This is original raw elec data from 2021/01/01 to 2022/01/29
    df_raw = df_raw[['日期', '淨尖峰供電能力(MW)', '尖峰負載(MW)', '工業用電(百萬度)', '民生用電(百萬度)']] # Filter out unnecessary features
    df_drop = df_raw.rename(columns=    # Renam columns
    {
        '日期' : 'date',
        '淨尖峰供電能力(MW)' : 'supply',
        '尖峰負載(MW)' : 'demand',
        '工業用電(百萬度)' : 'industry', 
        '民生用電(百萬度)' : 'civil'
    })
    df_dataset = df_drop.copy()
    df_dataset['OR'] = -1       # Target Feature : Tomorrow's OR
    df_dataset['today'] = -1    # Input Feature : Today's OR

    for idx in range( len(df_dataset) - 1 ):    # Generate Target/Input Feature
        df_dataset['today'][idx] = df_dataset['supply'][idx] - df_dataset['demand'][idx]
        df_dataset['OR'][idx] = df_dataset['supply'][idx + 1] - df_dataset['demand'][idx + 1]

    df_dataset.drop(df_dataset.tail(1).index, inplace=True)     # Drop last data while training
    min_max_scaler = MinMaxScaler()     # MinMaxScaler
    df_tmp = df_dataset.copy()
    min_max_scaler.fit(df_tmp[[         # Using MinMaxScaler to find outliers
            'supply', 'demand', 'industry', 'civil']])
    df_tmp[[                            # Useless > <
            'supply', 'demand', 'industry', 'civil']] = min_max_scaler.fit_transform(df_tmp[['supply', 'demand', 'industry', 'civil']])
    df_tmp.drop(columns='date', inplace=True)

    df_outlier = df_dataset[(np.abs(stats.zscore(df_dataset.drop(columns=['OR', 'date', 'today']))) < 3).all(axis=1)]   # Drop outliers

    min_max_scaler = MinMaxScaler()

    min_max_scaler.fit(df_outlier[[
            'supply', 'demand', 'industry', 'civil'
            ]])

    df_outlier[[
            'supply', 'demand', 'industry', 'civil'
            ]] = min_max_scaler.fit_transform(
                df_outlier[[
            'supply', 'demand', 'industry', 'civil'
            ]])

    df_outlier.to_csv("data/3_dataset_outlier_elec.csv")    # Save as backup file

    df_data = df_outlier.copy().drop(['supply', 'demand', 'industry', 'civil'], axis=1) # Drop useless features

    csv_url = "https://data.taipower.com.tw/opendata/apply/file/d006002/本年度每日尖峰備轉容量率.csv"   # Fetch .csv from URL
    req = requests.get(csv_url)
    url_content = req.content
    csv_file = open('data/df_more.csv', 'wb')       # Save fetched .csv file
    csv_file.write(url_content)
    csv_file.close()

    df_more = pd.read_csv("data/df_more.csv")
    df_more = df_more.rename(columns=
    {
        '日期' : 'date',
        '備轉容量(萬瓩)' : 'OR',
        '備轉容量率(%)' : 'p',
    })
    df_more.drop(['p'], axis=1, inplace=True)
    df_more['date'] = df_more['date'].str.replace("/", "").astype(int)
    df_more['OR'] = df_more['OR'] * 10
    df_more['today'] = -1

    for idx in range( len(df_more) - 1):
        df_more['today'][idx] = df_more['OR'][idx]
        df_more['OR'][idx] = df_more['OR'][idx+1]

    df_full = pd.concat([df_data, df_more[ df_more['date'] > 20220129 ]])
    df_full.to_csv("data/df_full.csv")

    df_train = df_full.iloc[:400]   # Until 2022/02/28
    df_test = df_full.iloc[400:428] # From 2022/02/28 to 2022/03/28

    train_x = df_train.drop(columns=['OR', 'date'])
    train_y = df_train['OR']

    test_x = df_test.drop(columns=['OR', 'date'])
    test_y = df_test['OR']

    param_grid = {
    'criterion':['friedman_mse', 'mse'],
    'learning_rate':[1, 0.1, 0.01, 0.001], 
    'n_estimators':[1, 10, 50, 100], 
    'subsample':[1], 
    'max_depth':[3, 5, 7],
    'random_state':[999]
    }

    # Split data into "trainning data", "testing data", and "validation data"
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=12)
    # Try different model parameters
    grid = GridSearchCV(GradientBoostingRegressor(), param_grid, verbose=5, n_jobs=-1)
    # Train model
    grid.fit(train_x, train_y)


    #df_training = pd.read_csv(args.training)
    model = GradientBoostingRegressor(
        criterion = grid.best_params_ ['criterion'],
        learning_rate = grid.best_params_['learning_rate'], 
        max_depth = grid.best_params_['max_depth'], 
        n_estimators = grid.best_params_['n_estimators'], 
        subsample = grid.best_params_['subsample'], 
        random_state = grid.best_params_['random_state']
        )
    model.fit(train_x, train_y)
    df_test = df_full[ (df_full['date'] >= 20220329) & (df_full['date'] <= 20220412) ].drop(columns=['OR', 'date'])
    df_result = model.predict(df_test)
    date = df_full[ (df_full['date'] >= 20220329) & (df_full['date'] <= 20220412) ]['date'] + 1
    df_output = pd.DataFrame(list(zip(date, df_result)), columns=['date', 'operating_reserve(MW)'])
    df_output.to_csv(args.output, index=False)