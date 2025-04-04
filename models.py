from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator, field_validator
import numpy as np

class ExchangeInfo(BaseModel):
    """Information about a cryptocurrency exchange."""
    exchange_id: str = Field(..., description="Exchange identifier")
    exchange_name: str = Field(..., description="Exchange name")
    
    model_config = {
        "extra": "allow",
    }

class CategoryInfo(BaseModel):
    """Information about a token category."""
    category_id: int = Field(..., description="Category identifier")
    category_name: str = Field(..., description="Category name") 
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