"""Tests for kramers-kronig penalty"""

import pytest
import penalty
import numpy as np

@pytest.fixture
def example0():
    path = 'sample_data/pf-rd-ox_fftkk.out'
    return(penalty.parse_data(path))

@pytest.fixture
def example1():
    path = 'sample_data/pf-rd-red_fftkk.out'
    return(penalty.parse_data(path))
  
def test_parse_example0_0(example0):
    """Test that input is parsed properly."""
    np.testing.assert_equal(example0[:4], np.array([[1006.0,-1.95442043311, 5.92009170594],
                                                    [1007.0, -2.81223119888, 8.52503033764],
                                                    [1008.0, -3.33235759037, 10.1156545543],
                                                    [1009.0, -3.56113395273, 10.8295300422]]))
    
def test_parse_example0_1(example0):
    """Test that input is parsed properly."""
    np.testing.assert_equal(example0[-4:], np.array([[24902.0,0.244888423888, 0.418575531827],
                                                     [24903.0,0.237396106135, 0.405759333696],
                                                     [24904.0,0.220802089543, 0.377388435958],
                                                     [24905.0,0.18454034239, 0.315405661584]]))

def test_get_f_p_example0(example0):
    f_p = penalty.get_f_p(example0[:,1])
    np.testing.assert_allclose(f_p,example0[:,2])
    
def test_get_f_dp_example0(example0):
    f_dp = penalty.get_f_dp(example0[:,2])
    np.testing.assert_allclose(f_dp,example0[:,1])

  