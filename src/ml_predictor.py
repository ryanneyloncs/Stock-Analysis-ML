"""
Machine Learning Price Prediction Module.

Implements an LSTM neural network for stock price prediction
using multiple technical indicators as features.

Features used: Close price, Volume, RSI, MACD, Moving Averages
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# TensorFlow is optional - gracefully handle if not installed
try:
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not installed. ML predictions disabled.")
    print("   Install with: pip install tensorflow")

DEFAULT_FEATURES = ["Close", "Volume", "RSI", "MACD", "MA50", "MA200"]
MIN_DATA_BUFFER = 50  # Minimum extra rows needed beyond lookback period
TRAIN_SPLIT_RATIO = 0.8


class MLPredictor:
    """
    Multi-feature LSTM-based stock price predictor.

    Uses a deep LSTM neural network to predict next-day closing prices
    based on historical price data and technical indicators.

    Attributes:
        lookback_days: Number of historical days used for each prediction.
        use_multiple_features: Whether to use multiple features or just Close price.
        scaler: MinMaxScaler for normalizing input features.
        model: Trained Keras Sequential model.
        feature_columns: List of feature column names used by the model.
    """

    def __init__(self, lookback_days: int = 60, use_multiple_features: bool = True) -> None:
        """
        Initialize the predictor.

        Args:
            lookback_days: Number of past days to use for prediction.
            use_multiple_features: If True, uses multiple technical indicators;
                                   if False, uses only Close price.
        """
        self.lookback_days = lookback_days
        self.use_multiple_features = use_multiple_features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model: Any = None
        self.feature_columns: list[str] = []

    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select features for the model based on availability."""
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
        """Create sequences for time series prediction."""
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_data)):
            X.append(scaled_data[i - self.lookback_days : i])
            y.append(scaled_data[i, 0])  # Predict Close price (first column)
        return np.array(X), np.array(y)

    def _inverse_transform_predictions(
        self, predictions: np.ndarray, n_features: int
    ) -> np.ndarray:
        """Inverse transform scaled predictions to actual price values."""
        pred_full = np.zeros((len(predictions), n_features))
        pred_full[:, 0] = predictions.flatten()
        return self.scaler.inverse_transform(pred_full)[:, 0]

    def prepare_data(
        self, data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model training.

        Creates sequences of historical data for time series prediction.
        Splits data into 80% training and 20% testing sets.

        Args:
            data: Stock data with technical indicators.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, scaled_data).

        Raises:
            ValueError: If there is insufficient data for training.
        """
        features_df = self._select_features(data).dropna()

        min_required = self.lookback_days + MIN_DATA_BUFFER
        if len(features_df) < min_required:
            raise ValueError(
                f"Not enough data. Need at least {min_required} rows, got {len(features_df)}"
            )

        scaled_data = self.scaler.fit_transform(features_df.values)
        X, y = self._create_sequences(scaled_data)

        split = int(TRAIN_SPLIT_RATIO * len(X))
        return X[:split], y[:split], X[split:], y[split:], scaled_data

    def build_model(self, n_features: int) -> Any:
        """
        Build the LSTM model architecture.

        Architecture:
        - 3 LSTM layers (100, 100, 50 units) with dropout
        - 2 Dense layers (25, 1 units)
        - Adam optimizer with MSE loss
        """
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
        """
        Train the LSTM model on historical data.

        Args:
            data: Stock data with technical indicators.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch).

        Returns:
            Dictionary containing training metrics, or None if training failed.
        """
        if not TENSORFLOW_AVAILABLE:
            return None

        print("Training multi-feature ML model...")

        try:
            X_train, y_train, X_test, y_test, _ = self.prepare_data(data)
            n_features = X_train.shape[2]

            self.model = self.build_model(n_features)
            self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=verbose,
            )

            # Generate and inverse-transform predictions
            train_pred = self._inverse_transform_predictions(
                self.model.predict(X_train, verbose=0), n_features
            )
            test_pred = self._inverse_transform_predictions(
                self.model.predict(X_test, verbose=0), n_features
            )
            y_train_actual = self._inverse_transform_predictions(
                y_train.reshape(-1, 1), n_features
            )
            y_test_actual = self._inverse_transform_predictions(
                y_test.reshape(-1, 1), n_features
            )

            metrics = {
                "train_mae": mean_absolute_error(y_train_actual, train_pred),
                "test_mae": mean_absolute_error(y_test_actual, test_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train_actual, train_pred)),
                "test_rmse": np.sqrt(mean_squared_error(y_test_actual, test_pred)),
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
        """
        Predict the next day's closing price.

        Args:
            data: Stock data with technical indicators.

        Returns:
            Predicted closing price, or None if prediction failed.
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return None

        try:
            features_df = self._select_features(data).dropna()
            last_data = features_df.values[-self.lookback_days :]
            last_data_scaled = self.scaler.transform(last_data)

            X_pred = last_data_scaled.reshape(1, self.lookback_days, len(self.feature_columns))
            prediction_scaled = self.model.predict(X_pred, verbose=0)

            return float(
                self._inverse_transform_predictions(prediction_scaled, len(self.feature_columns))[0]
            )

        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    def get_predictions_for_plotting(self, data: pd.DataFrame) -> pd.DataFrame | None:
        """
        Generate predictions for all data points for visualization.

        Args:
            data: Stock data with technical indicators.

        Returns:
            DataFrame with Date index and ML_Prediction column,
            or None if prediction failed.
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return None

        try:
            features_df = self._select_features(data).dropna()
            scaled_data = self.scaler.transform(features_df.values)

            X_all, _ = self._create_sequences(scaled_data)
            predictions = self._inverse_transform_predictions(
                self.model.predict(X_all, verbose=0), len(self.feature_columns)
            )

            dates = features_df.index[self.lookback_days :]
            return pd.DataFrame({"ML_Prediction": predictions}, index=dates)

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
    """
    Convenience function to train model and generate predictions.

    Creates an MLPredictor, trains it, and returns predictions in one call.

    Args:
        data: Stock data with technical indicators.
        lookback_days: Number of historical days for prediction.
        epochs: Number of training epochs.
        verbose: Training verbosity level.
        use_multiple_features: Whether to use multiple features.

    Returns:
        Tuple of (predictor, metrics, next_day_prediction, predictions_df).
        All values are None if TensorFlow is not available or training failed.
    """
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
