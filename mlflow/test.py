#%%
import mlflow
import numpy as np
#%%
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment(experiment_name='diabetes')



#%%
##example from mlflow docs
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
#%%

seed = 32
df = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(df.data, df.target, random_state=seed)
#%%

with mlflow.start_run(run_name="train") as run:

    #%%

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    
    rf.fit(X_train, y_train)

    #%%

    y_pred = rf.predict(X_test)
    #%%
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    #%%
    mlflow.log_metrics({"rmse": rmse,
                        "mape": mape})
    mlflow.log_params(rf.get_params())


    mlflow.log_artifact(local_path='test.py', artifact_path='code')

mlflow.end_run()