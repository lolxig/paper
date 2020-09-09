# oneclassSVM
## citation:
> http://papers.nips.cc/paper/1723-support-vector-method-for-novelty-detection.pdf <br />
> https://scikit-learn.org/stable/modules/outlier_detection.html#outlier-detection

## Novelty and Outlier Detection：

Many applications require being able to decide whether a new observation belongs to the same distribution as existing observations (it is an inlier), or should be considered as different (it is an outlier).
Often, this ability is used to clean real data sets. Two important distinction must be made:

### novelty detection:
The training data is not polluted by outliers, and we are interested in detecting anomalies in new observations.

### outlier detection:
The training data contains outliers, and we need to fit the central mode of the training data, ignoring the deviant observations.
