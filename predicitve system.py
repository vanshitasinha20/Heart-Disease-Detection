# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.


"""
import numpy as np
import pickle

loaded_model=pickle.load(open('S:/CODING/Heart Disease Detection/trained_model.sav','rb'))

input_data=(71,0,0,112,149,0,1,125,0,1.6,1,0,2)

#change the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as we are predicting for only one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1 )
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):#prediction[0] means the 0th index value i.e 0
  print('The Person does not have a Heart Disease')
else:
  print('The Person has a Heart Disease')