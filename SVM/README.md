Dataset: Wine quality: http://archive.ics.uci.edu/ml/datasets/Wine+Quality
Expert tasters given wine score, can chemical content be linked to this score? 

Dataset is divided into white and red wines of Portuguese of type Vinho Verde. 11 chemical substances is measured and score is given from 0 to 10 (best). We want to see how score correlates with chemical content. We classify good wines as having score above 6 (7-10) and poor wines as having lower than 5 (0-4). A weakness with the dataset is that there are most medium wines.

Plot shows poor wines in light blue colour.
We use the svm package from sklearn.

Several parameters proves to be useless as classifiers to wine quality: Density, Ph and sulphates.
The Python program tries to classify wines and then creates 4 type of plots.