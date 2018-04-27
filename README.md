# Parkinsons-Voice-Detection
Detect Parkinson's disease using the voice of a patient. 


From the UC Irvine's Dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons//

There was not much data so supervised learning would be the best approach, parkinsons.data.py contains various algorithms and their accuracy using a 75-25 split

KNN produced the best accuracy of about98%, with XGB Boost coming in second w/ about 96% accuracy. 

Future steps are to determine what features actually contribute to the model and continue to fine-tune the models. Also getting more data would help better determine the accuracy of the models.
