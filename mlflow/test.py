#%%
import mlflow

#%%
##example from mlflow docs
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


mlflow.sklearn.autolog()
#%%


df = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(df.data, df.target)

#%%

rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
# MLflow triggers logging automatically upon model fitting
rf.fit(X_train, y_train)
