---
updated: 2025-04-08T15:35
---
# RESULT ANALYSIS

This chapter describes the evaluation of the skin cancer detection model by discussing the performance metrics used to quantify the accuracy of the classification, and by outlining the process of deploying the model as a web application. Although the original discussion focused on general classification tasks, the same underlying principles apply to detecting multiple skin lesion types. In our context, the ability to quickly and accurately identify potential malignant lesions—especially melanoma—is critical for ensuring early intervention.

---

## PERFORMANCE METRICS

The foundation for improving any machine learning model lies in the iterative process of evaluating its outputs, learning from its errors, and refining the model. For our skin cancer detection model, performance metrics provide critical quantitative feedback that guides further development. The following metrics were examined:

### Accuracy  
Accuracy measures the overall proportion of correct predictions across all classes. When all classes (such as various types of benign lesions and malignant melanoma) are of equal importance in a preliminary evaluation, accuracy offers a broad understanding of model performance. It is computed as:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

* **In Skin Cancer Detection:**  
While accuracy is a useful indicator, caution must be exercised because the cost of misclassifying a malignant lesion (false negative) is much higher than wrongly flagging a benign lesion (false positive). In our experiments, accuracy was measured on a validation set that included a wide spectrum of lesion images.

### Precision  
Precision is defined as the ratio of correctly predicted positive observations (e.g., correctly predicted cancerous lesions) to the total predicted positives. It indicates the accuracy of the model when it flags a lesion as suspicious.
$$\text{Precision} = \frac{TP}{TP + FP}$$

* **Relevance for Skin Cancer:**  
High precision is crucial in medical applications to ensure that once the model marks a lesion as malignant (or a high-risk type), there is a high confidence that it is indeed the case. This helps in minimizing unnecessary anxiety and follow-up procedures due to overdiagnosis.

### Recall  
Recall (or sensitivity) measures the proportion of actual positives that are correctly identified. It is especially important in the context of skin cancer detection, where missing a malignant case (a false negative) could have serious consequences.

$$\text{Recall} = \frac{TP}{TP + FN}$$
* **Clinical Impact:**  
A high recall ensures that the majority of true cancer cases are identified by the model, thereby reducing the likelihood of dangerous oversights. For skin cancer detection, a robust recall is essential to support early diagnosis.

### F1 Score  
The F1 Score is the harmonic mean of precision and recall. This metric gives a balanced measure even if the classes are imbalanced, helping to capture both the rate of false positives and false negatives.
$$\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **Model Optimization:**  
In our iterative training process, the F1 score was a key metric—especially in situations where precision or recall individually might be high while the other is insufficiently low. Because skin cancer detection is a high-stakes domain, the F1 score serves as a more nuanced performance indicator.
