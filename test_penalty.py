"""Tests for kramers-kronig penalty"""

import pytest
import penalty
import numpy as np

@pytest.fixture
def example0():
    path = 'sample_data/fe.nff'
    return(penalty.parse_data(path))

@pytest.fixture
def example1():
    path = 'sample_data/mn.nff'
    return(penalty.parse_data(path))
  
def test_parse_example0_0(example0):
    """Test that input is parsed properly."""
    np.testing.assert_equal(example0[:4], np.array([[10.0000, -9999.00, 1.37852], 
                                                    [10.1617, -9999.00, 1.42961],
                                                    [10.3261, -9999.00, 1.48259],
                                                    [10.4931, -9999.00, 1.53754]]))
    
def test_parse_example0_1(example0):
    """Test that input is parsed properly."""
    np.testing.assert_equal(example0[-4:], np.array([[28590.2, 26.2151, 0.333497], 
                                                     [29052.6, 26.2100, 0.323310],
                                                     [29522.5, 26.2050, 0.313422],
                                                     [30000.0, 26.2000, 0.303827]]))

def test_get_f_p_example0(example0):
    f_p = penalty.get_f_p(example0[:,1])
    np.testing.assert_allclose(f_p,example0[:,2])
    
def test_get_f_dp_example0(example0):
    f_dp = penalty.get_f_dp(example0[:,2])
    np.testing.assert_allclose(f_dp,example0[:,1])

  