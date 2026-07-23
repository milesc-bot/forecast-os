"""Exception types for forecast-os."""


class ForecastOSError(Exception):
    """Base class for all forecast-os errors."""


class DataContractError(ForecastOSError):
    """The input data does not satisfy the (unique_id, ds, y) panel contract."""


class NotFittedError(ForecastOSError):
    """A model method that requires fitting was called before ``fit``."""
