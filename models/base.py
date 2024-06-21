from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


@dataclass
class RegressionModelResults:
    params: dict
    mse: float
    r2: float

    def __str__(self) -> str:
        return f""" Params:
    {json.dumps(self.params, indent=4)}
Results:
    MSE: {self.mse}
    R^2 score: {self.r2} """


class RegressionModelRunner:
    def __init__(
        self,
        data: pd.DataFrame,
        model_or_grid_search: RegressorMixin | GridSearchCV,
    ) -> None:
        self.X = data.iloc[:, 1:-1]
        self.y = data.iloc[:, -1]
        self._model = model_or_grid_search

        self.selector = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.results = None

    @property
    def feature_importances(self) -> ArrayLike:
        assert self.model is not None, f"Model hasn't been run!"
        if isinstance(self.model, (LinearRegression, SGDRegressor)):
            return self.model.coef_
        elif isinstance(self.model, (RandomForestRegressor, DecisionTreeRegressor)):
            return self.model.feature_importances_
        return NotImplemented

    @property
    def selector_scores(self) -> ArrayLike:
        assert self.selector is not None, f"Selection hasn't been done!"
        return self.selector.scores_

    @property
    def predictions(self) -> np.ndarray:
        assert self.y_pred is not None, f"Model hasn't been run!"
        return self.y_pred

    def split(self, test_size: int = 0.8, shuffle: bool = True) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, shuffle=shuffle
        )

    def scale(self) -> None:
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def select(self, k: int | Literal["all"] = 15) -> None:
        selector = SelectKBest(score_func=f_regression, k=k)
        self.X_train = selector.fit_transform(self.X_train, self.y_train)
        self.X_test = selector.transform(self.X_test)
        self.selector = selector

    def run(self) -> RegressionModelRunner:
        if self.X_train is None:
            self.split()
        self.scale()
        if self.selector is None:
            self.select()
        return self._run()

    def save_results(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(str(self.results))

    def _fit_model(self) -> RegressorMixin:
        self._model.fit(self.X_train, self.y_train)
        if isinstance(self._model, RegressorMixin):
            return self._model
        return self._model.best_estimator_

    def _run(self) -> RegressionModelRunner:
        self.model = self._fit_model()
        self.y_pred = self.model.predict(self.X_test)
        self.results = RegressionModelResults(
            params=self.model.get_params(),
            mse=mean_squared_error(self.y_test, self.y_pred),
            r2=r2_score(self.y_test, self.y_pred),
        )
        print(self.results)
        return self
