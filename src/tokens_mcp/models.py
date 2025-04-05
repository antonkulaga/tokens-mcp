from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator, field_validator
import numpy as np

class ExchangeInfo(BaseModel):
    """Exchange information from Token Metrics API"""
    exchange_id: Optional[str] = Field(None, description="Unique identifier for the exchange")
    exchange_name: Optional[str] = Field(None, description="Name of the exchange")
    
    model_config = {
        "extra": "allow",
    }

class CategoryInfo(BaseModel):
    """Category information from Token Metrics API"""
    category_id: Optional[int] = Field(None, description="Unique identifier for the category")
    category_name: Optional[str] = Field(None, description="Name of the category")
    category_slug: Optional[str] = Field(None, description="Category slug")
    
    model_config = {
        "extra": "allow",
    }
class TokenInfo(BaseModel):
    """Information about a cryptocurrency token."""
    TOKEN_ID: Optional[int] = Field(None, description="Unique token identifier")
    TOKEN_NAME: Optional[str] = Field(None, description="Name of the token")
    TOKEN_SYMBOL: Optional[str] = Field(None, description="Token symbol (e.g. BTC)")
    EXCHANGE_LIST: Optional[List[ExchangeInfo]] = Field(None, description="List of exchanges")
    CATEGORY_LIST: Optional[List[CategoryInfo]] = Field(None, description="List of categories")
    tm_link: Optional[str] = Field(None, description="Token Metrics link slug")
    contract_address: Optional[Dict[str, Any]] = Field(None, description="Contract addresses by blockchain")
    TM_LINK: Optional[str] = Field(None, description="Full Token Metrics URL")
    
    model_config = {
        "extra": "allow",
        "validate_assignment": True,
    }
    
    @field_validator('*', mode='before')
    @classmethod
    def convert_nan_to_none(cls, v):
        """Convert NaN values to None"""
        if isinstance(v, float) and np.isnan(v):
            return None
        return v
    
    @model_validator(mode="after")
    def parse_nested_structures(self):
        """Transform string representations of nested structures if needed"""
        # This is no longer needed as we handle this in the helpers.py parse_json_strings function
        return self

class PaginatedResponse(BaseModel):
    """Generic paginated response model."""
    data: List[Dict[str, Any]] = Field(..., description="Response data")
    meta: Optional[Dict[str, Any]] = Field(None, description="Response metadata")
    
    model_config = {
        "extra": "allow",
    }

class EmaData(BaseModel):
    """EMA data for a specific period"""
    values: List[float] = Field(..., description="List of EMA values")

class EmaResponse(BaseModel):
    """Response data from the get_emas endpoint"""
    symbol: str = Field(..., description="Token symbol")
    timeframe: str = Field(..., description="Timeframe for data (1D, 1H)")
    start_date: str = Field(..., description="Start date of the data")
    end_date: str = Field(..., description="End date of the data")
    dates: List[str] = Field(..., description="List of date strings")
    close: List[float] = Field(..., description="List of closing prices")
    open: List[float] = Field(..., description="List of opening prices")
    high: List[float] = Field(..., description="List of high prices")
    low: List[float] = Field(..., description="List of low prices")
    volume: List[float] = Field(..., description="List of volumes")
    emas: Dict[str, List[float]] = Field(..., description="Dictionary of EMA values by period")

    @field_validator('*', mode='before')
    @classmethod
    def transform_json_strings(cls, v):
        """Transform string representations of nested structures if needed"""
        # This is no longer needed as we handle this in the helpers.py parse_json_strings function
        return v 

class StrategyStats(BaseModel):
    """
    Model for trading strategy performance statistics.
    
    Attributes:
        total_return_pct: The strategy's total return as a percentage
        sharpe_ratio: The Sharpe ratio of the strategy
        max_drawdown_pct: Maximum drawdown as a percentage
        win_rate_pct: Win rate as a percentage
        trades_count: Total number of trades executed
    """
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades_count: int

class MACrossoverResponse(BaseModel):
    """
    Response model for Moving Average Crossover strategy results.
    
    Attributes:
        symbol: The token symbol
        dates: List of dates for the data points
        close_prices: List of close prices
        signals: List of signals (1 = buy, -1 = sell, 0 = hold)
        equity_curve: List of portfolio values over time
        ma_values: Dictionary containing short and long MA information
        stats: Strategy performance statistics
    """
    symbol: str
    dates: List[str]
    close_prices: List[float]
    signals: List[int]
    equity_curve: List[float]
    ma_values: Dict[str, Dict[str, Any]]  # Contains 'short' and 'long' keys with MA info
    stats: StrategyStats 