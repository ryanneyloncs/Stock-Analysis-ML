"""LSTM-based stock price prediction using technical indicators as features."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

try:
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not installed. ML predictions disabled.")
    print("   Install with: pip install tensorflow")

DEFAULT_FEATURES = ["Close", "Volume", "RSI", "MACD", "MA50", "MA200"]
MIN_DATA_BUFFER = 50
TRAIN_SPLIT_RATIO = 0.8


class MLPredictor:
    """LSTM predictor that forecasts next-day close price from historical indicators."""

    def __init__(self, lookback_days: int = 60, use_multiple_features: bool = True) -> None:
        self.lookback_days = lookback_days
        self.use_multiple_features = use_multiple_features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model: Any = None
        self.feature_columns: list[str] = []
        self._ref_closes: np.ndarray | None = None
        self._actual_closes: np.ndarray | None = None

    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.use_multiple_features:
            self.feature_columns = ["Close"]
            return data[["Close"]].copy()

        available_features = [col for col in DEFAULT_FEATURES if col in data.columns]

        if len(available_features) < 2:
            print("WARNING: Not enough features available, using Close price only")
            self.use_multiple_features = False
            self.feature_columns = ["Close"]
            return data[["Close"]].copy()

        self.feature_columns = available_features
        print(f"Using {len(available_features)} features: {', '.join(available_features)}")
        return data[available_features].copy()

    def _create_sequences(
        self, scaled_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a window over the data to build (X, y) sequence pairs."""
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_data)):
            X.append(scaled_data[i - self.lookback_days : i])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def _inverse_transform_returns(
        self, scaled_predictions: np.ndarray, n_features: int
    ) -> np.ndarray:
        """Convert scaled predictions back to return values."""
        pred_full = np.zeros((len(scaled_predictions), n_features))
        pred_full[:, 0] = scaled_predictions.flatten()
        return self.scaler.inverse_transform(pred_full)[:, 0]

    def prepare_data(
        self, data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Scale features, build sequences, and split 80/20 into train/test."""
        features_df = self._select_features(data).dropna()
        close_prices = features_df["Close"].values

        # Train on percentage returns instead of raw prices (scale-invariant)
        returns_df = features_df.pct_change().replace([np.inf, -np.inf], 0).dropna()

        min_required = self.lookback_days + MIN_DATA_BUFFER
        if len(returns_df) < min_required:
            raise ValueError(
                f"Not enough data. Need at least {min_required} rows, got {len(returns_df)}"
            )

        scaled_data = self.scaler.fit_transform(returns_df.values)
        X, y = self._create_sequences(scaled_data)

        n_preds = len(X)
        # Previous-day close for each prediction (to convert return -> price)
        self._ref_closes = close_prices[self.lookback_days : self.lookback_days + n_preds]
        # Actual close on the target day (for error metrics in dollar terms)
        self._actual_closes = close_prices[self.lookback_days + 1 : self.lookback_days + 1 + n_preds]

        split = int(TRAIN_SPLIT_RATIO * len(X))
        return X[:split], y[:split], X[split:], y[split:], scaled_data

    def build_model(self, n_features: int) -> Any:
        """3x LSTM (100, 100, 50) with dropout -> Dense(25) -> Dense(1)."""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.lookback_days, n_features)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def train(
        self,
        data: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> dict[str, Any] | None:
        """Train the model and return a metrics dict, or None on failure."""
        if not TENSORFLOW_AVAILABLE:
            return None

        print("Training multi-feature ML model...")

        try:
            X_train, y_train, X_test, y_test, _ = self.prepare_data(data)
            n_features = X_train.shape[2]
            split = len(X_train)

            self.model = self.build_model(n_features)
            self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=verbose,
            )

            # Convert predicted returns back to dollar prices
            train_returns = self._inverse_transform_returns(
                self.model.predict(X_train, verbose=0), n_features
            )
            test_returns = self._inverse_transform_returns(
                self.model.predict(X_test, verbose=0), n_features
            )

            train_pred = self._ref_closes[:split] * (1 + train_returns)
            test_pred = self._ref_closes[split:] * (1 + test_returns)
            train_actual = self._actual_closes[:split]
            test_actual = self._actual_closes[split:]

            metrics = {
                "train_mae": mean_absolute_error(train_actual, train_pred),
                "test_mae": mean_absolute_error(test_actual, test_pred),
                "train_rmse": np.sqrt(mean_squared_error(train_actual, train_pred)),
                "test_rmse": np.sqrt(mean_squared_error(test_actual, test_pred)),
                "train_predictions": train_pred,
                "test_predictions": test_pred,
                "n_features": n_features,
                "features_used": self.feature_columns,
            }

            print(
                f"Model trained - Test MAE: ${metrics['test_mae']:.2f}, "
                f"Test RMSE: ${metrics['test_rmse']:.2f}"
            )
            return metrics

        except Exception as e:
            print(f"Error training model: {e}")
            return None

    def predict_next_day(self, data: pd.DataFrame) -> float | None:
        """Predict tomorrow's closing price."""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return None

        try:
            features_df = self._select_features(data).dropna()
            last_close = features_df["Close"].iloc[-1]

            returns_df = features_df.pct_change().replace([np.inf, -np.inf], 0).dropna()
            last_returns = returns_df.values[-self.lookback_days :]
            last_returns_scaled = self.scaler.transform(last_returns)

            X_pred = last_returns_scaled.reshape(1, self.lookback_days, len(self.feature_columns))
            pred_scaled = self.model.predict(X_pred, verbose=0)

            predicted_return = self._inverse_transform_returns(
                pred_scaled, len(self.feature_columns)
            )[0]
            return float(last_close * (1 + predicted_return))

        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    def get_predictions_for_plotting(self, data: pd.DataFrame) -> pd.DataFrame | None:
        """Run predictions over the full history for chart overlay."""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return None

        try:
            features_df = self._select_features(data).dropna()
            close_prices = features_df["Close"].values

            returns_df = features_df.pct_change().replace([np.inf, -np.inf], 0).dropna()
            scaled_data = self.scaler.transform(returns_df.values)

            X_all, _ = self._create_sequences(scaled_data)
            predicted_returns = self._inverse_transform_returns(
                self.model.predict(X_all, verbose=0), len(self.feature_columns)
            )

            ref_closes = close_prices[self.lookback_days : self.lookback_days + len(X_all)]
            predicted_prices = ref_closes * (1 + predicted_returns)

            dates = returns_df.index[self.lookback_days :]
            return pd.DataFrame({"ML_Prediction": predicted_prices}, index=dates)

        except Exception as e:
            print(f"Error getting predictions for plotting: {e}")
            return None


def predict_stock_price(
    data: pd.DataFrame,
    lookback_days: int = 60,
    epochs: int = 50,
    verbose: int = 0,
    use_multiple_features: bool = True,
) -> tuple[MLPredictor | None, dict[str, Any] | None, float | None, pd.DataFrame | None]:
    """One-shot helper: train a predictor and return (predictor, metrics, next_day, predictions_df)."""
    if not TENSORFLOW_AVAILABLE:
        return None, None, None, None

    predictor = MLPredictor(
        lookback_days=lookback_days, use_multiple_features=use_multiple_features
    )
    metrics = predictor.train(data, epochs=epochs, verbose=verbose)

    if metrics is None:
        return None, None, None, None

    next_day_pred = predictor.predict_next_day(data)
    predictions_df = predictor.get_predictions_for_plotting(data)

    return predictor, metrics, next_day_pred, predictions_df
