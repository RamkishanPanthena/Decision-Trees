# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:49:05 2018

@author: Krishna
"""

import math
import pandas as pd

# Decision Trees
class DecisionTree:
    def __init__(self, entropy_threshold = 0.3, rowcount_threshold = 10, depth_threshold = 7):
        self.df = None
        self.columnname = None
        self.colsplitvalue = None
        self.trueRes = None
        self.falseRes = None
        self.informationGain = None
        self.depth = 0
        self.currEntropy = 0
        self.trueEntropy = 0
        self.falseEntropy = 0
        self.finalRes = 0
        self.percRes = 0
        self.rowCount = 0
        self.trueCond = None
        self.falseCond = None
        self.entropy_threshold = entropy_threshold
        self.rowcount_threshold = rowcount_threshold
        self.depth_threshold = depth_threshold
        
    # Initialize values
    def set_values(self):
        self.currEntropy = self.calc_entropy(self.df, 'label')
        temp = self.calc_label_inconsistency(self.df, 'label')
        self.finalRes = temp[1]
        self.percRes = temp[0]
        self.rowCount = len(self.df)

    # Function to find the majority class in a split
    def calc_label_inconsistency(self, dataframe, columnname):
        totrecords = len(dataframe)
        if totrecords == 0:
            return (1)
            
        x1 = dataframe.loc[dataframe[columnname] == 0].count()[columnname] / totrecords
        x2 = dataframe.loc[dataframe[columnname] == 1].count()[columnname] / totrecords
            
        if x1 > x2:
            return([x1, 0])
        else:
            return([x2, 1])

    # Function to calculate entropy
    def calc_entropy(self, dataframe, columnname):
        totrecords = len(dataframe)
        if totrecords == 0:
            return (1)
        
        x1 = dataframe.loc[dataframe[columnname] == 0].count()[columnname] / totrecords
        x2 = dataframe.loc[dataframe[columnname] == 1].count()[columnname] / totrecords
        
        if x1 == 0:
            val1 = 0
        else:
            val1 = -1 * x1 * math.log2(x1)
            
        if x2 == 0:
            val2 = 0
        else:
            val2 = -1 * x2 * math.log2(x2)
            
        entropy = val1 + val2
        
        return entropy
               
    # Find the best predictor on which split needs to take place
    def best_predictor(self):
        df = self.df
        col_list = list(df.columns.values)
        col_list.remove('label')
        
        curr_entropy = self.calc_entropy(df, 'label')
        maxig = float("-inf")
        
        for column in col_list:
            # Convert to a list and sort the values
            a = df[column].values.tolist()
            a.sort()
            
            # Pick 1/10th of the unique records to decide the best classifier
            uniqset = set()
            for i in range(len(a)):
                if i%10==0:
                    uniqset.add(a[i])
            
            for val in uniqset:
                newdf1 = df.loc[df[column]<val]
                newdf2 = df.loc[df[column]>=val]
                
                entropy1 = self.calc_entropy(newdf1, 'label')
                entropy2 = self.calc_entropy(newdf2, 'label')
                
                totrecords = len(df)
                ig = curr_entropy - (len(newdf1)/totrecords) * entropy1 -(len(newdf2)/totrecords) * entropy2
            
                if ig > maxig:
                    maxig = ig
                    splitval = val
                    splitcolumn = column
                    splitentropy1 = entropy1
                    splitentropy2 = entropy2
                    splitdf1 = newdf1
                    splitdf2 = newdf2
            
        self.informationGain = maxig
        self.colsplitvalue = splitval
        self.columnname = splitcolumn
        self.trueEntropy = splitentropy1
        self.falseEntropy = splitentropy2
        self.trueRes = splitdf1
        self.falseRes = splitdf2
        
    # Function to check whether to continue splitting or not
    def proc(self, dataset, currdepth):
        new_node = DecisionTree(self.entropy_threshold, self.rowcount_threshold, self.depth_threshold)
        new_node.df = dataset
        new_node.set_values()
        new_node.depth = currdepth+1
        
        if new_node.currEntropy > new_node.entropy_threshold and new_node.rowCount > new_node.rowcount_threshold and new_node.depth < new_node.depth_threshold:
            new_node.best_predictor()
            return (new_node)
        else:
            if new_node.finalRes == 1:
                return (1)
            elif new_node.finalRes == 0:
                return (0)        
        
    # Function to run decision tree recursively
    def process_decision_tree(self, model):
        model.trueCond = self.proc(model.trueRes, model.depth)
        model.falseCond = self.proc(model.falseRes, model.depth)
        
        if model.trueCond not in set((0,1)):
            model.trueCond = self.process_decision_tree(model.trueCond)
        else:
            pass
    
        if model.falseCond not in set((0,1)):
            model.falseCond = self.process_decision_tree(model.falseCond)
        else:
            pass
        
        return model
        
    # Function to calculate the initial split and call processDecisionTree
    def fit(self, dataset, model):
        model.df = dataset
        model.set_values()
        model.best_predictor()
        model.process_decision_tree(model)

    # Function to make prediction for a single row
    def predict_row(self, model, row):
        col = model.columnname
        val = model.colsplitvalue
        global pred
        if row[col].values[0] < val:
            output = model.trueCond
            if output not in set((0,1)):
                self.predict_row(model.trueCond, row)
            else:
                pred = output
                
        else:
            output = model.falseCond
            if output not in set((0,1)):
                self.predict_row(model.falseCond, row)
            else:
                pred = output
                
        return pred
            
    # Function to make predictions for the entire dataset
    def predict(self, model, df):
        prediction = []
        
        for i in range(len(df)):
            prediction.append(self.predict_row(model, df.iloc[[i]]))
        
        label = {'label': prediction}

        return pd.DataFrame(label)
        
        
    # Function to calculate accuracy
    def accuracy(self, actual, predictions):
        matchcount = 0
        for i in range(len(actual)):
            if actual.iloc[[i]]['label'].values[0] == predictions.iloc[[i]]['label'].values[0]:
                matchcount+=1
         
        return matchcount/len(actual)