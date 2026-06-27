import pytest
import numpy as np
import pandas as pd
from features import _pct_return, _rsi, _rolling_std

def test_pct_return():
    """Test standard n-day percentage return logic"""
    data = {'A': [100, 105, 110, 108]}
    df = pd.DataFrame(data)
    result = _pct_return(df, 1)
    
    # 105/100 - 1 = 0.05
    assert np.isclose(result['A'].iloc[1], 0.05)
    # 110/105 - 1 = 0.047619
    assert np.isclose(result['A'].iloc[2], 0.047619, atol=1e-4)

def test_rolling_std():
    """Test rolling standard deviation computation"""
    data = {'A': [1, 2, 3, 4, 5, 6]}
    df = pd.DataFrame(data)
    result = _rolling_std(df, 3)
    assert len(result) == 6
    assert 'A' in result.columns

def test_rsi():
    """Test RSI computation against a known pattern"""
    data = {'A': [100, 102, 104, 103, 101, 99, 98, 100, 105, 110, 108, 109, 111, 115, 118, 120]}
    df = pd.DataFrame(data)
    
    # Calculate 14-period RSI
    result = _rsi(df, 14)
    
    assert len(result) == 16
    # The first 14 elements (0 to 13) should be NaN because min_periods=14
    assert np.isnan(result['A'].iloc[0])
    assert not np.isnan(result['A'].iloc[14])
    
    # Value should be bound between 0 and 100
    assert 0 <= result['A'].iloc[14] <= 100
