#%%
import mlflow

%run utils/utils.py
#%%
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment(experiment_name=experiment_id)


df = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(df.data, df.target)

#%%
d_train = xgb.DMatrix(X_train, label=y_train)
d_test = xgb.DMatrix(X_test, label=y_test)

#%%
def objective(trial):
    with mlflow.start_run(nested=True):
        # Define hyperparameters
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }

        if params["booster"] == "gbtree" or params["booster"] == "dart":
            params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            params["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )

        # Train XGBoost model
        bst = xgb.train(params, d_train)
        preds = bst.predict(d_test)
        error = mean_squared_error(y_test, preds)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("mse", error)
        mlflow.log_metric("rmse", math.sqrt(error))

    return error


#%%


# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(run_name=run_name, nested=True):
    # Initialize the Optuna study
    study = optuna.create_study(direction="minimize")

    # Execute the hyperparameter optimization trials.
    # Note the addition of the `champion_callback` inclusion to control our logging
    study.optimize(objective, n_trials=10, callbacks=[champion_callback])

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_mse", study.best_value)
    mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

    # Log tags
    mlflow.set_tags(
        tags={
            "project": "Apple Demand Project",
            "optimizer_engine": "optuna",
            "model_family": "xgboost",
            "feature_set_version": 1,
        }
    )

    # Log a fit model instance
    model = xgb.train(study.best_params, d_train)

    # Log the correlation plot
    #mlflow.log_figure(figure=correlation_plot, artifact_file="correlation_plot.png")

    # Log the feature importances plot
    #importances = plot_feature_importance(model, booster=study.best_params.get("booster"))
    #mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

    # Log the residuals plot
    #residuals = plot_residuals(model, dvalid, valid_y)
    #mlflow.log_figure(figure=residuals, artifact_file="residuals.png")

    #artifact_path = "model"

    mlflow.xgboost.log_model(
        xgb_model=model,
        model_format="ubj",
        artifact_path='code',
        metadata={"model_data_version": 1},
    )
    mlflow.log_artifact(local_path='train.py', artifact_path='code/train')



