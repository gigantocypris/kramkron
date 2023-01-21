"""Tests for kramers-kronig core_functions"""

import pytest
import core_functions
import numpy as np

@pytest.fixture
def example0():
    path = "sample_data/pf-rd-ox_fftkk.out"
    return(core_functions.parse_data(path))

@pytest.fixture
def example1():
    path = "sample_data/pf-rd-red_fftkk.out"
    return(core_functions.parse_data(path))

@pytest.fixture
def example2():
    path = "sample_data/fe.nff"
    return(core_functions.parse_data(path,remove_first_line=True))

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

def test_parse_example2_0(example2):
    """Test that input is parsed properly."""
    np.testing.assert_equal(example2[:4], np.array([[10.0000, -9999.00, 1.37852], 
                                                    [10.1617, -9999.00, 1.42961],
                                                    [10.3261, -9999.00, 1.48259],
                                                    [10.4931, -9999.00, 1.53754]]))
    
def test_parse_example2_1(example2):
    """Test that input is parsed properly."""
    np.testing.assert_equal(example2[-4:], np.array([[28590.2, 26.2151, 0.333497], 
                                                     [29052.6, 26.2100, 0.323310],
                                                     [29522.5, 26.2050, 0.313422],
                                                     [30000.0, 26.2000, 0.303827]]))

def test_get_f_p_example0(example0):
    """Test that the Hilbert transform and the approach taken by the Sherrell thesis match."""
    padn=5000
    Z=26
    include_Z_term=False
    energy = example0[:,0]
    f_dp = example0[:,2]
    energy_interp,f_p_pred = core_functions.get_f_p(energy, f_dp, padn=padn,
                                     Z = Z, # atomic number
                                     include_Z_term=include_Z_term,
                                     hilbert_transform_func=core_functions.hilbert_transform)
    
    energy_interp,f_p_pred_sherrell = core_functions.get_f_p(energy, f_dp, padn=padn,
                                              Z = Z, # atomic number
                                              include_Z_term=include_Z_term,
                                              hilbert_transform_func=core_functions.hilbert_transform_sherrell)
    
    np.testing.assert_allclose(f_p_pred, f_p_pred_sherrell)

def test_get_f_p_get_f_dp_example0(example0):
    """Test that finding f_p and then finding f_dp yields the input f_dp."""
    padn=5000
    crop=500
    Z=26
    include_Z_term=False
    energy = example0[:,0]
    f_dp = example0[:,2]
    
    
    
    energy_interp,f_p_pred = core_functions.get_f_p(energy, f_dp, padn=padn,
                                     Z = Z, # atomic number
                                     include_Z_term=include_Z_term,
                                     hilbert_transform_func=core_functions.hilbert_transform)
    
    
    f_dp_interp = core_functions.INTERP_FUNC(energy, f_dp)(energy_interp)
    
    energy_interp,f_dp_pred = core_functions.get_f_dp(energy_interp, f_p_pred, padn=padn,
                                                      Z = Z, # atomic number
                                                      include_Z_term=include_Z_term,
                                                      hilbert_transform_func=core_functions.hilbert_transform)
    np.testing.assert_allclose(f_dp_interp[-crop:crop], f_dp_pred[-crop:crop])

def test_penalty_example0(example0):
    """Test that finding f_p and then calculating the penalty yields 0 penalty"""
    padn=5000
    Z=26
    include_Z_term=False
    energy = example0[:,0]
    f_dp = example0[:,2]
    
    
    
    energy_interp,f_p_pred = core_functions.get_f_p(energy, f_dp, padn=padn,
                                     Z = Z, # atomic number
                                     include_Z_term=include_Z_term,
                                     hilbert_transform_func=core_functions.hilbert_transform)
    
    
    f_dp_interp = core_functions.INTERP_FUNC(energy, f_dp)(energy_interp)
    
    mse,_,_,_ = core_functions.penalty(energy_interp, f_p_pred, f_dp_interp, 
                                       padn=padn, Z=Z, include_Z_term=include_Z_term,
                                       hilbert_transform_func=core_functions.hilbert_transform)
    np.testing.assert_allclose(mse, 0)
    
def test_get_f_p_get_f_dp_example2(example2):
    """Test that finding f_p and then finding f_dp yields the input f_dp."""
    padn=5000
    crop=500
    Z=26
    include_Z_term=False
    energy = example2[:,0]
    f_dp = example2[:,2]
    
    
    
    energy_interp,f_p_pred = core_functions.get_f_p(energy, f_dp, padn=padn,
                                     Z = Z, # atomic number
                                     include_Z_term=include_Z_term,
                                     hilbert_transform_func=core_functions.hilbert_transform)
    
    
    f_dp_interp = core_functions.INTERP_FUNC(energy, f_dp)(energy_interp)
    
    energy_interp,f_dp_pred = core_functions.get_f_dp(energy_interp, f_p_pred, padn=padn,
                                                      Z = Z, # atomic number
                                                      include_Z_term=include_Z_term,
                                                      hilbert_transform_func=core_functions.hilbert_transform)
    np.testing.assert_allclose(f_dp_interp[-crop:crop], f_dp_pred[-crop:crop])

def test_penalty_example2(example2):
    """Test that finding f_p and then calculating the penalty yields 0 penalty"""
    padn=5000
    Z=26
    include_Z_term=False
    energy = example2[:,0]
    f_dp = example2[:,2]
    
    
    
    energy_interp,f_p_pred = core_functions.get_f_p(energy, f_dp, padn=padn,
                                     Z = Z, # atomic number
                                     include_Z_term=include_Z_term,
                                     hilbert_transform_func=core_functions.hilbert_transform)
    
    
    f_dp_interp = core_functions.INTERP_FUNC(energy, f_dp)(energy_interp)
    
    mse,_,_,_ = core_functions.penalty(energy_interp, f_p_pred, f_dp_interp, 
                                       padn=padn, Z=Z, include_Z_term=include_Z_term,
                                       hilbert_transform_func=core_functions.hilbert_transform)
    np.testing.assert_allclose(mse, 0, atol=1e-7)