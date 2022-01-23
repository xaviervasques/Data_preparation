#!/usr/bin/python3
# categorical_encoders.py
# Xavier Vasques (Last update: 18/01/2022)

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import hashlib

def categorical_encoders(df, y, encoder, feature_to_encode, n_components, hash_method):

    if encoder == "ordinal":
        try:
            df[feature_to_encode] = ordinalencoding(df[feature_to_encode])
        except:
            print("Something went wrong with Ordinal Encoder: Please Check")
        
    if encoder == "one_hot":
        try:
         df = pd.get_dummies(df, prefix="One_Hot", columns=feature_to_encode)
        except:
            print("Something went wrong with One Hot Encoder: Please Check")
        
    if encoder == "label":
        try:
            df[feature_to_encode] = labelencoding(df[feature_to_encode])
        except:
            print("Something went wrong with Label Encoder: Please Check")
            
    if encoder == "helmert":
        try:
            Y = helmertencoding(df[feature_to_encode])
            df = df.drop(feature_to_encode, axis = 1)
            df = pd.concat([df, Y], axis=1)
        except:
            print("Something went wrong with Helmert Encoder: Please Check")
            
    if encoder == "binary":
        try:
            Y = binaryencoding(df[feature_to_encode])
            df = df.drop(feature_to_encode, axis = 1)
            df = pd.concat([df, Y], axis=1)
        except:
            print("Something went wrong with Binary Encoder: Please Check")
            
    if encoder == "frequency":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # grouping by frequency
                frequency = df.groupby(col_name).size()/len(df)
                # mapping values to dataframe
                df.loc[:,"{}_freq_encode".format(col_name)] = df[(col_name)].map(frequency)
                # drop original column
                df = df.drop([col_name], axis = 1)
        except:
            print("Something went wrong with Frequency Encoder: Please Check")
    
    
    if encoder == "mean":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of TargetEncoder
                mean_encoder = ce.TargetEncoder(drop_invariant=True)
                # Assigning numerical value and storing it
                df_encoded = mean_encoder.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('Mean_Encoding_')
                # Concatenate dataframe and drop original column
                df = pd.concat([df, df_encoded], axis=1)
                df = df.drop([col_name], axis = 1)
        except:
            print("Something went wrong with Mean Encoder: Please Check")
                   
    if encoder == "sum":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of SumEncoder
                sum_encoder = ce.SumEncoder(drop_invariant=True)
                # Assigning numerical value and storing it
                df_encoded = sum_encoder.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('Sum_Encoding_')
                # Concatenate dataframe and drop original column
                df = pd.concat([df, df_encoded], axis=1)
                df = df.drop([col_name], axis = 1)
        except:
            print("Something went wrong with Sum Encoder: Please Check")
            
    if encoder == "weightofevidence":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of SumEncoder
                #regularization is mostly to prevent division by zero.
                woe = ce.WOEEncoder(random_state=42, regularization=0)
                # Assigning numerical value and storing it
                df_encoded = woe.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('WoE_Encoding_')
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
                print("Something went wrong with Weight of Evidence Encoder: Please Check")
            

    if encoder == "probabilityratio":
        try:
            df = pd.concat([y, df],axis=1)
            for col_name in feature_to_encode:
                # Calculation of the probability of target being 1
                probability_encoding_1 = df.groupby(col_name)['Target'].mean()
                print(probability_encoding_1)
                # Calculation of the probability of target not being 1
                probability_encoding_0 = 1 - probability_encoding_1
                probability_encoding_0 = np.where(probability_encoding_0 == 0, 0.00001, probability_encoding_0)
                # Probability ratio calculation
                df_encoded = probability_encoding_1 / probability_encoding_0
                # Map the probability ratio into the data
                df.loc[:,'Proba_Ratio_%s'%col_name] = df[col_name].map(df_encoded)
                # Drop feature to let the transformed one
                df = df.drop([col_name], axis = 1)
            df = df.drop(['Target'], axis = 1)
        except:
            print("Something went wrong with Probability Ratio Encoder: Please Check")
    
    if encoder == "hashing":
        #try:
        # Get all features
        for col_name in feature_to_encode:
            # Creating an instance of HashingEncoder
            # n_components contains the number of bits you want in your hash value.
            encoder_purpose = ce.HashingEncoder(n_components=n_components, hash_method=hash_method)
            # Assigning numerical value and storing it
            df_encoded = encoder_purpose.fit_transform(df[col_name])
            # We renanme columns to identify which feature we transformed
            for x in range(n_components):
                df_encoded = df_encoded.rename(columns={"col_%i"%x: "%s_%s_%i"%('Hashing',col_name,x)})
            # Drop original column and concatenate dataframe
            df = df.drop([col_name], axis = 1)
            df = pd.concat([df, df_encoded], axis=1)
        #except:
        #   print("Something went wrong with Hashing Encoder: Please Check")
            
    if encoder == "backwarddifference":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of BackwardDifferenceEncoder
                encoder = ce.BackwardDifferenceEncoder(cols=col_name,drop_invariant=True)
                # Assigning numerical value and storing it
                df_encoded = encoder.fit_transform(df[col_name])
                df_encoded = df_encoded.add_prefix('Backward_Diff_')
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
            print("Something went wrong with Backward Difference Encoder: Please Check")

    if encoder == "leaveoneout":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of BackwardDifferenceEncoder
                encoder = ce.LeaveOneOutEncoder(cols=col_name)
                # Assigning numerical value and storing it
                df_encoded = encoder.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('Leave_One_Out_')
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
            print("Something went wrong with Leave One Out Encoder: Please Check")
            
            
    if encoder == "jamesstein":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of BackwardDifferenceEncoder
                encoder = ce.JamesSteinEncoder(cols=col_name)
                # Assigning numerical value and storing it
                df_encoded = encoder.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('James_Stein_')
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
            print("Something went wrong with James-Stein Encoder: Please Check")
    
    if encoder == "mestimator":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of MEstimateEncoder
                encoder = ce.MEstimateEncoder(cols=col_name)
                # Assigning numerical value and storing it
                df_encoded = encoder.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('M_Estimator_')
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
            print("Something went wrong with M-Estimator Encoder: Please Check")
            
    return(df)
               
def ordinalencoding(X):
    # Creating an instance of Ordinalencoder
    enc = OrdinalEncoder()
    # Assigning numerical value and storing it
    enc.fit(X)
    X = enc.transform(X)
    X = X.add_prefix('Ordinal_Encoding_')
    return X
    
def onehotencoding(X):
    X = pd.get_dummies(X, prefix="One_Hot_Encoding")
    return X
    
def labelencoding(X):
    # Creating an instance of Labelencoder
    enc = LabelEncoder()
    # Assigning numerical value and storing it
    X = X.apply(enc.fit_transform)
    X = X.add_prefix('Label_Encoding_')
    return X
        
def helmertencoding(X):
    # Creating an instance of HelmertEncoder
    enc = ce.HelmertEncoder(drop_invariant=True)
    # Assigning numerical value and storing it
    X = enc.fit_transform(X)
    X = X.add_prefix('Helmert_Encoding_')
    return X
            
def binaryencoding(X):
    # Creating an instance of BinaryEncoder
    enc = ce.BinaryEncoder()
    # Assigning numerical value and storing it
    df_binary = enc.fit_transform(X)
    df_binary = df_binary.add_prefix('Binary_Encoding_')
    return df_binary

        

