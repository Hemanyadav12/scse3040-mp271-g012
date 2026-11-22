"""Model pipeline: impute + scale numbers, one-hot the text columns, regressor."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REGRESSORS = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def feature_columns(params):
    return params["features"]["numeric"] + params["features"]["categorical"]


def build_pipeline(model_name, model_params, params):
    kwargs = dict(model_params)

    if model_name != "ridge":
        kwargs["random_state"] = params["seed"]

    pre = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]),
            params["features"]["numeric"]
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            params["features"]["categorical"]
        ),
    ])

    return Pipeline([
        ("pre", pre),
        ("model", REGRESSORS[model_name](**kwargs))
    ])


def mae(pipe, df, params):
    return float(
        mean_absolute_error(
            df[params["target"]],
            pipe.predict(df[feature_columns(params)])
        )
    )
