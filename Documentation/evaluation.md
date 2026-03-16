# 2.5 Model evaluation and testing

The model was evaluated on 87 labeled slices from four different trees. Random slices were selected to reduce the labeling workload. Three of the trees in the test dataset are from Griese et al. (2025) and from a different site in Germany than the training data with two coniferous and one deciduous tree. Selecting trees from geographically distinct sites and utilizing different data collection methods (such as varying measurement devices) ensures spatial cross-validation to avoid inflated performance scores (Kattenborn et al., 2022). One of the labeled trees was from the Matthisle site to evaluate the effect of site and data collection differences.

Two different approaches were used to evaluate the predictions. First, bounding box performance was assessed using standard metrics computed by the Ultralytics package (Precision, Recall, F1, mAP50, and mAP50-95) with an Intersection over Union (IoU) threshold of 0.5. However, because our application uses bounding boxes to assign each point to a class based on a defined hierarchy, we implemented a custom pixel-level evaluation for a more meaningful comparison. Both ground truth and predicted bounding boxes were rendered into pixel masks, where higher-priority classes overwrote lower-priority ones in overlapping regions. Pixels not recognized as data points were explicitly filtered out as background. From the resulting pixel-level confusion matrix, we extracted True Positives (TP), False Positives (FP), and False Negatives (FN) to calculate:

**Precision:**

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$,

**Recall:** 

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$,

**F1 Score:**
 
$$\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$.