#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:09:48 2017

@author: jpoeppel
"""

import unittest
import warnings

import numpy as np

import random

from primo2.inference.factor import Factor
from primo2.inference.marginal import Marginal
from primo2.nodes import DiscreteNode

class MarginalTest(unittest.TestCase):
    
    def __init__(self, methodName, testFactor=None):
        super(MarginalTest, self).__init__(methodName)
        self.longMessage = True
        self.factor = testFactor      
    
    def test_create_from_factor(self):
        m = Marginal.from_factor(self.factor)
        
        self.assertEqual(m.variables, self.factor.variableOrder)
        for v in self.factor.values:
            self.assertEqual(m.values[v], self.factor.values[v])
        np.testing.assert_array_equal(m.probabilities, self.factor.potentials)
        pass
    
    def test_get_probabilities_entire_variable_str_dict(self):
        m = Marginal.from_factor(self.factor)
        varName = self.factor.checkVar
        res = m.get_probabilities(varName, returnDict=True)
        self.assertIsInstance(res, dict)
        for k in res:
            np.testing.assert_array_equal(res[k], self.factor.get_potential({varName:[k]}))
        pass
    
    def test_get_probabilities_entire_variable_str_array(self):
        m = Marginal.from_factor(self.factor)
        varName = self.factor.checkVar
        res = m.get_probabilities(varName, returnDict=False)
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, self.factor.potentials)
        
    def test_get_probabilities_entire_variable_list_dict(self):
        m = Marginal.from_factor(self.factor)
        varName = self.factor.checkVar
        res = m.get_probabilities({varName: self.factor.values[varName]}, returnDict=True)
        self.assertIsInstance(res, dict)
        for k in res:
            np.testing.assert_array_equal(res[k], self.factor.get_potential({varName:[k]}))
    
    def test_get_probabilities_entire_variable_list_array(self):
        m = Marginal.from_factor(self.factor)
        varName = self.factor.checkVar
        res = m.get_probabilities({varName: self.factor.values[varName]}, returnDict=False)
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, self.factor.potentials)
        
    def test_get_probabilities_entire_variable_empty_list_array(self):
        m = Marginal.from_factor(self.factor)
        varName = self.factor.checkVar
        res = m.get_probabilities({varName: []}, returnDict=False)
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, self.factor.potentials)
        
    def test_get_probabilities_entire_variable_empty_list_dict(self):
        m = Marginal.from_factor(self.factor)
        varName = self.factor.checkVar
        res = m.get_probabilities({varName: []}, returnDict=True)
        self.assertIsInstance(res, dict)
        for k in res:
            np.testing.assert_array_equal(res[k], self.factor.get_potential({varName: [k]}))
    
    def test_get_probabilities_part_variable_list_dict(self):
        m = Marginal.from_factor(self.factor)
        varName = self.factor.checkVar
        #Use random values to avoid having to create tests for all possible outcomes
        value = random.choice(self.factor.values[varName])
        res = m.get_probabilities({varName: [value]}, returnDict=True)
        self.assertIsInstance(res, dict)
        for k in res:
            np.testing.assert_array_equal(res[k], self.factor.get_potential({varName: [k]}), 
                                          err_msg="Failed for value: {} of variable {}".format(value, varName), 
                                          verbose=True)
    
    def test_get_probabilities_part_variable_list_array(self):
        m = Marginal.from_factor(self.factor)
        varName = self.factor.checkVar
        #Use random values to avoid having to create tests for all possible outcomes
        value = random.choice(self.factor.values[varName])
        res = m.get_probabilities({varName: [value]}, returnDict=False)
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, self.factor.get_potential({varName: [value]}), 
                                          err_msg="Failed for value: {} of variable {}".format(value, varName), 
                                          verbose=True)
        
    def test_get_probabilities_part_variable_str_dict(self):
        m = Marginal.from_factor(self.factor)
        varName = self.factor.checkVar
        #Use random values to avoid having to create tests for all possible outcomes
        value = random.choice(self.factor.values[varName])
        res = m.get_probabilities({varName: value}, returnDict=True)
        self.assertIsInstance(res, dict)
        for k in res:
            np.testing.assert_array_equal(res[k], self.factor.get_potential({varName: [k]}), 
                                          err_msg="Failed for value: {} of variable {}".format(value, varName), 
                                          verbose=True)
    
    def test_get_probabilities_part_variable_str_array(self):
        m = Marginal.from_factor(self.factor)
        varName = self.factor.checkVar
        #Use random values to avoid having to create tests for all possible outcomes
        value = random.choice(self.factor.values[varName])
        res = m.get_probabilities({varName: value}, returnDict=False)
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, self.factor.get_potential({varName: [value]}), 
                                          err_msg="Failed for value: {} of variable {}".format(value, varName), 
                                          verbose=True)
    
    def test_get_probabilitites_unknown_variable(self):
        m = Marginal.from_factor(self.factor)
        wrongVar = self.factor.wrongVar
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            
            m.get_probabilities(wrongVar)
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, RuntimeWarning)
            self.assertEqual(str(w[0].message), "The variable {} is not part of this marginal and will be ignored.".format(wrongVar))
        
        
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            
            m.get_probabilities({wrongVar:[]})
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, RuntimeWarning)
            self.assertEqual(str(w[0].message), "The variable {} is not part of this marginal and will be ignored.".format(wrongVar))
        
    def test_get_probabilitites_unknown_value(self):
        m = Marginal.from_factor(self.factor)
        checkVar = self.factor.checkVar
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            
            res = m.get_probabilities({checkVar: "unknown"})
            
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, RuntimeWarning)
            self.assertEqual(str(w[0].message), "Unknown value ({}) for variable {}. Ignoring this variable.".format("unknown", checkVar))
            np.testing.assert_array_equal(res, self.factor.potentials)
        
        
        
    def test_get_probability_simple(self):
        pass
    
    def test_get_probability_under_specified(self):
        pass
    
    def test_get_probability_fully_specified(self):
        pass
    
    def test_marginalize(self):
        m = Marginal.from_factor(self.factor)
        #Use random variables to avoid having to create tests for all possible outcomes
        checkVar = random.choice(self.factor.variableOrder)
        resm = m.marginalize(checkVar)
        checkList = list(m.variables)
        checkList.remove(checkVar)
        self.assertEqual(resm.variables, checkList)
        np.testing.assert_array_equal(resm.probabilities, self.factor.marginalize(checkVar).potentials)
    
    def test_marginalize_missing(self):
        m = Marginal.from_factor(self.factor)
        #Use random variables to avoid having to create tests for all possible outcomes
        checkVar = "NotThere"
        resm = m.marginalize(checkVar)
        checkList = list(m.variables)
        checkList.remove(checkVar)
        self.assertEqual(resm.variables, checkList)
        np.testing.assert_array_equal(resm.probabilities, self.factor.marginalize(checkVar).potentials)
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            res = m.marginalize(["A",checkVar])
            self.assertEqual(len(w),1)
            self.assertEqual(w[0].category, RuntimeWarning)
            self.assertEqual(str(w[0].message), "Variable {} will be ignored since it is not contained in this marginal.".format(checkVar))
            np.testing.assert_array_equal(res.probabilities, self.factor.marginalize("A").potentials)
    
######### Will not be run with all factors in setUp_test_factors, but with specific factors #########
    def test_get_probabilitites_multiple_variables_simple_dict(self):
        #Uses f2
        m = Marginal.from_factor(self.factor)
        res = m.get_probabilities({"A": "2", "B": "Apples"}, returnDict=True)
        self.assertIsInstance(res, dict)
        for k in res:
            for v in res[k]:
                np.testing.assert_array_equal(res[k][v], 
                        self.factor.get_potential({"A": ["2"], "B":["Apples"]}))
                
        res = m.get_probabilities({"A": "2", "B": ["Apples"]}, returnDict=True)
        self.assertIsInstance(res, dict)
        for k in res:
            for v in res[k]:
                np.testing.assert_array_equal(res[k][v], 
                        self.factor.get_potential({"A": ["2"], "B":["Apples"]}))
                
                
    def test_get_probabilitites_multiple_variables_complex_dict(self):
        #Uses f2
        m = Marginal.from_factor(self.factor)
        res = m.get_probabilities({"A": "2", "B": ["Apples", "Peaches"]}, returnDict=True)
        self.assertIsInstance(res, dict)
        for k in res:
            for v in res[k]:
                if k == "A":
                    np.testing.assert_array_equal(res[k][v], 
                          self.factor.get_potential({"A": ["2"], 
                                                     "B":["Apples","Peaches"]}))
                else:
                    np.testing.assert_array_equal(res[k][v], 
                          self.factor.get_potential({"A": ["2"], "B":[v]}))
                
                
        res = m.get_probabilities({"A": ["2","1"], "B": ["Apples", "Peaches"]}, returnDict=True)
        self.assertIsInstance(res, dict)
        for k in res:
            for v in res[k]:
                if k == "A":
                    np.testing.assert_array_equal(res[k][v], 
                          self.factor.get_potential({"A": [v], 
                                                     "B":["Apples","Peaches"]}))
                else:
                    np.testing.assert_array_equal(res[k][v], 
                          self.factor.get_potential({"A": ["2","1"], "B":[v]}))
                
    
    def test_get_probabilitites_multiple_variables_simple_array(self):
        #Uses f2
        m = Marginal.from_factor(self.factor)
        res = m.get_probabilities({"A": "2", "B": "Apples"}, returnDict=False)
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, self.factor.get_potential({"A": ["2"], "B":["Apples"]}))
                
        res = m.get_probabilities({"A": "2", "B": ["Apples"]}, returnDict=False)
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, self.factor.get_potential({"A": ["2"], "B":["Apples"]}))
                
    
    
    def test_get_probabilitites_multiple_variables_complex_array(self):
        #Uses f2
        m = Marginal.from_factor(self.factor)
        res = m.get_probabilities({"A": "2", "B": ["Apples", "Peaches"]}, returnDict=False)
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, 
                          self.factor.get_potential({"A": ["2"], 
                                                     "B":["Apples", "Peaches"]}))
                
                
        res = m.get_probabilities({"A": ["2","1"], "B": ["Apples", "Peaches"]}, returnDict=False)
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, 
                          self.factor.get_potential({"A": ["2","1"], 
                                                     "B":["Apples", "Peaches"]}))
    
    
    def test_get_probabilities_node(self):
        #Uses f
        m = Marginal.from_factor(self.factor)
        n = DiscreteNode("A")
        res = m.get_probabilities(n)
        np.testing.assert_array_equal(res, self.factor.potentials)
        res = m.get_probabilities({n:[]}, returnDict=True)
        for k in res:
            np.testing.assert_array_equal(res[k], self.factor.get_potential({"A":[k]}))
    
def setUp_test_factors():
    f = Factor()
    f.variableOrder = ["A"]
    f.values = {"A": ["True","False"]}
    f.variables = {"A":0}
    f.potentials = np.array([0.2,0.8])
    f.checkVar = "A"
    f.wrongVar = "B"
    
    f2 = Factor()
    f2.variableOrder = ["A", "B"]
    f2.values = {"A": ["1","2","3"], "B":["Apples", "Peaches"]}
    f2.variables = {"A":0, "B":1}
    f2.potentials = np.array([[0.2,0.1], [0.15,0.05], [0.27,0.23]])
    f2.checkVar = "B"
    f2.wrongVar = "C"
    return [f,f2]
    
def load_tests(loader, tests, pattern):
    test_cases = unittest.TestSuite()
    factors = setUp_test_factors()
    for f in factors:
        test_cases.addTest(MarginalTest("test_create_from_factor", f))
        test_cases.addTest(MarginalTest("test_get_probabilities_entire_variable_str_dict", f))
        test_cases.addTest(MarginalTest("test_get_probabilities_entire_variable_str_array", f))
        test_cases.addTest(MarginalTest("test_get_probabilities_entire_variable_list_dict", f))
        test_cases.addTest(MarginalTest("test_get_probabilities_entire_variable_list_array", f))
        test_cases.addTest(MarginalTest("test_get_probabilities_entire_variable_empty_list_array", f))
        test_cases.addTest(MarginalTest("test_get_probabilities_entire_variable_empty_list_dict", f))
        test_cases.addTest(MarginalTest("test_get_probabilities_part_variable_list_dict", f))
        test_cases.addTest(MarginalTest("test_get_probabilities_part_variable_list_array", f))
        test_cases.addTest(MarginalTest("test_get_probabilities_part_variable_str_dict", f))
        test_cases.addTest(MarginalTest("test_get_probabilities_part_variable_list_dict", f))
        
        test_cases.addTest(MarginalTest("test_get_probabilitites_unknown_variable", f))
        test_cases.addTest(MarginalTest("test_get_probabilitites_unknown_value", f))
#        test_cases.addTest(MarginalTest("test_get_probability_simple", f))
#        test_cases.addTest(MarginalTest("test_get_probability_under_specified", f))
#        test_cases.addTest(MarginalTest("test_get_probability_fully_specified", f))
        test_cases.addTest(MarginalTest("test_marginalize", f))
#        test_cases.addTest(MarginalTest("test_marginalize_missing", f))

    test_cases.addTest(MarginalTest("test_get_probabilities_node", factors[0]))
    test_cases.addTest(MarginalTest("test_get_probabilitites_multiple_variables_simple_dict", factors[1]))
    test_cases.addTest(MarginalTest("test_get_probabilitites_multiple_variables_simple_array", factors[1]))
    test_cases.addTest(MarginalTest("test_get_probabilitites_multiple_variables_complex_dict", factors[1]))
    test_cases.addTest(MarginalTest("test_get_probabilitites_multiple_variables_complex_array", factors[1]))
    return test_cases
    
    
if __name__ == "__main__":
    unittest.main()