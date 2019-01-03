from ExtractFeatures import  Extract_Features
from scipy.spatial import distance
sample=Extract_Features

row=3
col=3

row_goal=3
col_goal=3

data_current=sample.Extract_Samples(row,col)
data_goal=sample.Extract_Samples(row_goal,col_goal)

print (distance.euclidean(data_goal, data_current))