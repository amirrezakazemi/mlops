import pickle
import pandas as pd
import numpy as np
import sys



def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(df):

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f'Mean of predictions: {np.mean(y_pred)}')
    print(f'Std of predictions: {np.std(y_pred)}')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['pred'] = y_pred
    df_result = df[['ride_id', 'pred']]

    output_file = f'preds{year:04d}-{month:02d}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__=="__main__":

    if len(sys.argv) != 3:
        print("Usage: python your_script.py arg1 arg2")
    else:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        

    categorical = ['PULocationID', 'DOLocationID']
    filename = f'yellow_tripdata_{year:04d}-{month:02d}.parquet'

    df = read_data(filename)
    predict(df)









