#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of PRIMO2 -- Probabilistic Inference Modules.
# Copyright (C) 2013-2017 Social Cognitive Systems Group, 
#                         Faculty of Technology, Bielefeld University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Lesser GNU General Public License as 
# published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public 
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

import unittest
from primo2.networks import BayesianNetwork
from primo2.nodes import RandomNode, DiscreteNode

class BayesNetTest(unittest.TestCase):
    
    def setUp(self):
        self.bn = BayesianNetwork()
    
    def test_add_node(self):
        n = RandomNode("Node1")
        self.assertEqual(len(self.bn), 0)
        self.bn.add_node(n)
        self.assertEqual(len(self.bn), 1)
        
    def test_add_node_only_string(self):
        with self.assertRaises(TypeError) as cm:
            self.bn.add_node("I am not a node")
        self.assertEqual(str(cm.exception), "Only subclasses of RandomNode are valid nodes.")
        
    def test_add_node_same_name(self):
        n1 = RandomNode("Node1")
        n2 = RandomNode("Node2")
        n3 = RandomNode("Node1")
        self.assertEqual(len(self.bn), 0)
        self.bn.add_node(n1)
        self.assertEqual(len(self.bn), 1)
        with self.assertRaises(ValueError) as cm:
            self.bn.add_node(n3)
        self.assertEqual(str(cm.exception), "The network already contains a node called '{}'.".format(n1.name))
        self.assertEqual(len(self.bn), 1)
        self.bn.add_node(n2)
        self.assertEqual(len(self.bn), 2)
        
    def test_change_node_values_of_parent(self):
        n1 = DiscreteNode("Node1")
        n2 = DiscreteNode("Node2")
        self.bn.add_node(n1)
        self.bn.add_node(n2)
        self.bn.add_edge(n2,n1)
        self.assertEqual(n1.cpd.shape, (2,2))
        self.bn.change_node_values(n2, ["Value1","Value2","Value3"])
        self.assertEqual(n2.values, ["Value1","Value2","Value3"])
        self.assertEqual(n1.valid, False)
        self.assertEqual(n1.cpd.shape, (2,3))
        
                
#    def test_addEdge(self):
#        self.fail("TODO")
                
                
    """
        TODO: ADD MISSING TESTS
    """
        
    
    
if __name__ == "__main__":
    unittest.main()