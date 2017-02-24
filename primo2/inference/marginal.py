#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:37:05 2017

@author: jpoeppel
"""

import numpy as np
import warnings

class Marginal(object):
    
    """
        A class representing the inference results. This class holds the
        (joint) probability distribution after performing inference.
        
        TODO: Consider adding potentially used evidence and it's probability
        as well.
    """
    
    def __init__(self):
        self.variables = []
        self.values = {}
        self.probabilities = 0
        
        
    def copy(self):
        """
            Creates a (deep) copy of this marginal.
            
            Returns
            -------
                Marginal
                The copied marginal
        """
        res = Marginal()
        res.variables = list(self.variables)
        res.values = dict(self.values)
        res.probabilities = np.copy(self.probabilities)
        return res
    
    @classmethod
    def from_factor(cls, factor):
        """
            Creates a marginal from a factor. This method should only be used
            internally as a factor does not make any guarantees about the kind
            of potential it contains, thus calling it with factors not containing
            marginal probabilities will result in invalid marginals!
            
            Parameters
            ----------
            factor: Factor
                The factor whose potential is used to construct the marginal.
                
            Returns
            --------
                Marginal
                The created marginal representing the (joint) posterior
                marginal over all the variables in the factor.
        """
        res = cls()
        res.variables = factor.variableOrder
        res.values = dict(factor.values)
        res.probabilities = factor.potentials.copy()
        return res
    
    def get_probabilities(self, variables=None, returnDict=False):
        """
            Returns the probabilities for the specified variable(s), if specified,
            either as a compact numpy array (default), ordered according to the 
            variable and corresponding value orders in self.variables and 
            self.values, or as a dictionary.
            
            If variables is not specified it will return the probabilities for
            all included variables in the specfied form.
            
            Parameter
            ---------
            variables: Dict, RandomNode, String, optional.
                Dictionary containing the desired variables (the actual RandomNode
                or their Name) as keys and either an instantiation or a list 
                of instantiations of interest as values. An empty list will be
                interpreted as ALL values for that variable.
                For a marginal containing the binary variables A and B, 
                get_probabilities({"A":"True"}) and get_probabilities({"A":["True"]})
                will return the probabilties P(A=True, B=True) and 
                P(A=True, B=False). 
                Whereas get_probabilities({"A":"True", "B":"False"}) will only 
                return P(A=True, B=False).
                
                Any variable that is not part of the marginal will issue a
                warning and will be ignored. A variable is also ignored if an
                unknown instantiation was set for it.
                
                
            returnDict: Boolean, optional (default: False)
                Specifies if the probabilities should be returned as a dictionary
                of the form {variable: {value: probabilities}} (if set to true) or as
                a compact np.array with one dimension for each variable, according
                to the order given in self.variables. The entries within each
                dimension correspond to the values specified with the same 
                indices in self.values for that variable.
                In the simple case where only one variable is desired and a 
                dictionary should be returned, the outer dictionary is omitted.
                
            Returns
            -------
                dict or np.array
                The probabilities for the desired variables and their instantiations.
                See the optional returnDict parameter for more information about
                the return type.                
        """
        
        
        
        if not variables:
            variables = {}
        elif not isinstance(variables, dict):
            try:
                variables = {variables: self.values[variables]}
            except KeyError:
                variables = {variables: []}
                
        #Check variables in order to raise consistent warnings and make sure
        #values are in unified form
        for v in variables:
            if not v in self.variables:
                warnings.warn("The variable {} is not part of this marginal "\
                              "and will be ignored.".format(v),
                              RuntimeWarning)
            else:
                #For compatibility with both python2 and python3, we need both checks
                #since in python3 strings also have __iter__ which makes it not
                #possible to distinguish between list-like objects and strings
                #easily.
                if not hasattr(variables[v], "__iter__") or isinstance(variables[v],str):
                    #Catch the case of {"A":"True"}
                    variables[v] = [variables[v]]
                
        if returnDict:
            #If we want to return dicts, just call this method multiple
            #times to construct the partial matrizes that we want
            #TODO This is quite inefficient!!!
            res = {}            
            for var in variables:
                tmp = {}
                for val in variables[var]:
                    tmpVariables = dict(variables)
                    tmpVariables[var] = [val]
                    tmp[val] = self.get_probabilities(tmpVariables)
                res[var] = tmp
            
            if len(res) == 1:
                #Omit outer dictionary in the trivial case
                return res.values()[0]
            return res
            
        index = []        
        for v in self.variables:
            if v in variables and len(variables[v]) > 0:
                try:
                    #Otherwise we just take the indices of interest
                    index.append([self.values[v].index(value) for value in variables[v]])
                except ValueError:
                    warnings.warn("Unknown value ({}) for variable {}. "\
                                  "Ignoring this variable.".format(value, v), 
                                  RuntimeWarning)
                    index.append(range(len(self.values[v])))
            else:
                #If a variable is not specified or we have an empty list, 
                #we use the entire slice
                index.append(range(len(self.values[v])))
            
        res = np.squeeze(np.copy(self.probabilities[np.ix_(*index)]))
            
        return res


    def marginalize(self, variables):
        """
            Allows to marginalize out one or multiple variables.
            
            Parameters
            ----------
            variables: RandomNode, String, [RandomNode,], [String,]
                The variable(s) that should be marginalized out. Variables
                that were not part of the Marginal will be ignored but will
                raise a Warning.
                
            Returns
            --------
                Marginal
                The marginal resulting in summing out the given variables.
        """
        #For compatibility with both python2 and python3, we need both checks
        #since in python3 strings also have __iter__ which makes it not
        #possible to distinguish between list-like objects and strings
        #easily.
        if not hasattr(variables, "__iter__") or isinstance(variables,str):
            variables = [variables]
            
        res = self.copy()
        for v in variables:
            try:
                res.probabilities = np.sum(res.probabilities, axis=res.variables.index(v))
                del res.values[v]
                res.variables.remove(v)
            except ValueError:
                warnings.warn("Variable {} will be ignored since it is not " \
                              "contained in this marginal.".format(v), RuntimeWarning)
            
        return res