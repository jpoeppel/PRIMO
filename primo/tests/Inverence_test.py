#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:47:59 2016

@author: jpoeppel
"""

import unittest
import numpy as np
from primo.network import BayesianNetwork
from primo.io import XMLBIFParser
from primo.inference.order import Orderer
from primo.inference.exact import VariableElimination
from primo.inference.exact import FactorTree

class EliminationOderTest(unittest.TestCase):
    
    def test_min_degree_elimination_order(self):
        bn = XMLBIFParser.parse("primo/tests/slippery.xbif")
        order = Orderer.get_min_degree_order(bn)
        self.assertEqual(order, ["slippery_road", "wet_grass", "sprinkler", "winter", "rain"])
        
        """
            TODO BETTER TEST WITH CERTAIN ORDER!
        """
        #Check error handling
        with self.assertRaises(TypeError) as cm:
            Orderer.get_min_degree_order("Not a Bayesian Network.")
        self.assertEqual(str(cm.exception), "Only Bayesian Networks are currently supported.")
        
    def test_random_elimination_order(self):
        bn = XMLBIFParser.parse("primo/tests/slippery.xbif")
        order = Orderer.get_random_order(bn)
        variables = ["slippery_road", "winter", "rain", "sprinkler", "wet_grass"]
        self.assertEqual(len(order), len(variables))
        for v in variables:
            self.assertTrue(v in order)
            
        #Check error handling
        with self.assertRaises(TypeError) as cm:
            Orderer.get_min_degree_order("Not a Bayesian Network.")
        self.assertEqual(str(cm.exception), "Only Bayesian Networks are currently supported.")

class VariableEliminationTest(unittest.TestCase):
    
    def setUp(self):
        self.bn = XMLBIFParser.parse("primo/tests/slippery.xbif")
        
        
    def test_naive_marginals(self):
        resFactor = VariableElimination.naive_marginals(self.bn, ["winter"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.6, 0.4]))
        
    def test_naive_marginal_evidence_trivial(self):
        resFactor = VariableElimination.naive_marginals(self.bn, ["rain"], {"winter": "true"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.8, 0.2]))
        
    def test_naive_marginal_evidence_trivial_multiple_evidence(self):
        resFactor = VariableElimination.naive_marginals(self.bn, ["wet_grass"], {"sprinkler": "true", "rain": "false"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.1, 0.9]))
        
    def test_naive_marginal_evidence(self):
        resFactor = VariableElimination.naive_marginals(self.bn, ["wet_grass"], {"winter": "true"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.668, 0.332]))
        
    def test_naive_marginal_evidence_multiple_evidence(self):
        resFactor = VariableElimination.naive_marginals(self.bn, ["wet_grass"], {"winter": "true", "rain": "false"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.02, 0.98]))
        

        
    def test_bucket_marginals(self):
        resFactor = VariableElimination.bucket_marginals(self.bn, ["winter"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.6, 0.4]))
#        
    def test_bucket_marginal_evidence_trivial(self):
        resFactor = VariableElimination.bucket_marginals(self.bn, ["rain"], {"wet_grass": "false"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.158858, 0.841142]))
        
    def test_bucket_marginal_evidence_trivial_multiple_evidence(self):
        resFactor = VariableElimination.bucket_marginals(self.bn, ["wet_grass"], {"sprinkler": "true", "rain": "false"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.1, 0.9]))
    
        
    def test_bucket_marginal_evidence(self):
        resFactor = VariableElimination.bucket_marginals(self.bn, ["wet_grass"], {"winter": "true"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.668, 0.332]))
        
    def test_bucket_marginal_evidence_multiple_evidence(self):
        resFactor = VariableElimination.bucket_marginals(self.bn, ["wet_grass"], {"winter": "true", "rain": "false"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.02, 0.98]))
        
    ### TODO check multiple marginals
#    def test_bucket_multiple_marginals(self):
#        resFactor = VariableElimination.bucket_marginals(self.bn, ["wet_grass", "rain"], {"winter": "true", "slippery_road": "false"})
        
        
class FactorEliminationTest(unittest.TestCase):
    
    
    def setUp(self):
        self.bn = XMLBIFParser.parse("primo/tests/slippery.xbif")
        
    def test_create_jointree(self):
        ft = FactorTree.create_jointree(self.bn)
        desiredCliques = ["slippery_roadrain", "wet_grasssprinklerrain", "sprinklerwinterrain"]
        self.assertEqual(len(ft.tree), 3)
        for n in ft.tree.nodes_iter():
            self.assertTrue(n in desiredCliques)
        
    def test_jointree_marginals(self):
        ft = FactorTree.create_jointree(self.bn)
        resFactor = ft.marginals(["winter"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.6, 0.4]))
        
    def test_jointree_marginals(self):
        ft = FactorTree.create_jointree(self.bn)
        resFactor = ft.marginals(["slippery_road"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.364, 0.636]))
#        
    def test_jointree_evidence_trivial(self):
        ft = FactorTree.create_jointree(self.bn)
        ft.set_evidence({"wet_grass": "false"})
        resFactor = ft.marginals(["rain"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.158858, 0.841142]))
        
    def test_jointree_marginal_evidence_trivial_multiple_evidence(self):
        ft = FactorTree.create_jointree(self.bn)
        ft.set_evidence({"sprinkler": "true", "rain": "false"})
        resFactor = ft.marginals(["wet_grass"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.1, 0.9]))
    
        
    def test_jointree_marginal_evidence(self):
        ft = FactorTree.create_jointree(self.bn)
        ft.set_evidence({"winter": "true"})
        resFactor = ft.marginals(["wet_grass"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.668, 0.332]))
        
    def test_jointree_marginal_evidence_multiple_evidence(self):
        ft = FactorTree.create_jointree(self.bn)
        ft.set_evidence( {"winter": "true", "rain": "false"})
        resFactor = ft.marginals(["wet_grass"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.02, 0.98]))
        
        
if __name__ == "__main__":
    #Workaround so that this script also finds the resource files when run directly
    # from within the tests folder
    import os
    os.chdir("../..")
    unittest.main()