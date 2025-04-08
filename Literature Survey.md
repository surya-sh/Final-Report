---
updated: 2025-04-08T07:39
---
# Literature Survey

Over the past decade, skin cancer detection using AI and ML has become a rapidly advancing area of medical research. Numerous studies have been conducted to apply computer vision techniques to dermatology, specifically for the classification of skin lesions. Early approaches involved traditional ML algorithms like SVMs (Support Vector Machines) and decision trees, which relied heavily on handcrafted features such as texture, shape, and color. However, these methods were limited in performance due to the complexity and variability of skin lesions. With the advent of deep learning, especially Convolutional Neural Networks (CNNs), the ability to automatically learn intricate and abstract features from dermatoscopic images has significantly improved classification accuracy. Landmark studies have demonstrated that CNNs can match or even surpass dermatologists in identifying melanoma. This literature provides a strong foundation for integrating ensemble models, as used in this project, to further improve generalization and robustness across diverse populations.

Below is an extensive literature survey tailored to the paper “An Interpretable Deep Learning Approach for Skin Cancer Categorization.” This survey covers prior works on skin cancer detection, the evolution of deep learning architectures in medical imaging, and the role of explainable AI (XAI) in clinical applications. Each section includes links and citations to help locate the original sources.

## Deep Learning for Skin Cancer Detection

Early studies demonstrated that convolutional neural networks (CNNs) could automatically learn discriminative features from dermoscopic images. For instance:

- **Innani et al. (2022)** proposed a cascaded approach with an encoder–decoder framework for skin lesion analysis. Their method involved image segmentation to focus on the region of interest (ROI) before classification, achieving improved accuracy when the ROI was emphasized.  
  Link to paper details[^1].

- **Fraiwan et al. (2022)** employed transfer learning by fine-tuning 13 deep CNN models on the HAM10000 dataset without extra feature extraction or segmentation. Their work highlighted that DenseNet201, among others, could serve as a baseline for skin cancer detection[^2].

- **Kiran Pai and Giridharan (2019)** applied the VGGNet architecture for imbalanced multiclass classification of skin lesions. Their results underscored the importance of early detection for a higher chance of survival[^3].

- **Huang et al. (2021)** focused on developing lightweight deep learning models deployable on cloud and mobile platforms for skin cancer diagnosis, achieving high accuracy with EfficientNetB4[^4].

Collectively, these studies form a strong foundation for the integration of deep learning into dermatological diagnostics, with continual improvements in both model accuracy and real-world applicability.

---

## Pre-trained CNN Architectures in Medical Imaging

The surveyed paper leverages four state-of-the-art pre-trained CNN models, each with its unique architecture and scaling approach:

- **XceptionNet (Chollet, 2017):**  
  This architecture utilizes depthwise separable convolutions to reduce the number of parameters while maintaining high performance on image classification benchmarks. Its modular design enables more efficient learning and has been successfully applied to various medical imaging tasks[^5].

- **EfficientNetV2S (Tan & Le, 2021):**  
  An evolution of the EfficientNet family, EfficientNetV2S employs compound scaling of depth, width, and resolution to achieve a balanced trade-off between accuracy and model size. Its design is particularly beneficial for resource-constrained environments[^6].

- **InceptionResNetV2 (Szegedy et al., 2017):**  
  Combining the strengths of Inception modules and residual connections, InceptionResNetV2 efficiently captures multi-scale features while mitigating the vanishing gradient problem in very deep networks[^7]. 

- **EfficientNetV2M:**  
  Similar in principle to the “S” variant, the “M” variant is scaled to balance between computational efficiency and model performance. It uses modern training techniques such as Squeeze-and-Excitation blocks and the Swish activation function[^8]. 

These architectures have been widely adopted in medical imaging because their pre-training on large datasets provides a strong starting point for transfer learning on specialized tasks like skin cancer categorization.

---

## Explainable Artificial Intelligence (XAI) in Medical Diagnostics

A critical advancement in this field is the integration of XAI techniques to make deep learning models more transparent:

- **SmoothGrad (Smilkov et al., 2017):**  
  By adding noise to the input and averaging the gradients, SmoothGrad helps reduce visual noise in sensitivity maps, providing clearer insights into which parts of an image influence the model’s decisions.  
  [SmoothGrad paper on arXiv](https://arxiv.org/abs/1706.03825) citeturn0file0

- **Score-CAM and Faster Score-CAM (Wang et al., 2020):**  
  Score-CAM is designed to generate class activation maps that highlight important image regions without relying on gradients. The faster variant improves computational efficiency while retaining interpretability.  
  [Score-CAM publication](https://openaccess.thecvf.com/content_CVPRW_2020/html/w18/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.html) citeturn0file0

The need for XAI in clinical settings cannot be overstated. In healthcare, clinicians require explanations for automated decisions to validate and trust the model outputs. Several studies, including those by Van der Velden et al. (2022) and Rouf Shawon et al. (2022), have demonstrated that incorporating explainability not only aids in verifying diagnoses but also improves the adoption of AI-driven systems in practice.

---

## Performance Metrics and Comparative Analyses

A recurring theme in the literature is the use of performance metrics such as accuracy, precision, recall, and F1-score to evaluate model effectiveness. Comparative studies have shown that while earlier models like MobileNet and DenseNet achieved accuracies in the range of 73–79%, more recent approaches using EfficientNetB4 and XceptionNet have pushed accuracy levels close to or beyond 85% in skin cancer detection tasks. For example:

- **Innani et al. (2022)** reported improvements with ROI-based segmentation in conjunction with CNN models.  
- **Fraiwan et al. (2022)** benchmarked multiple architectures, where DenseNet201 reached 73.5% accuracy.  
- **Huang et al. (2021)** achieved 85.8% accuracy with EfficientNetB4.  
- The surveyed paper reports that XceptionNet achieved an outstanding accuracy of 88.72%, demonstrating the value of combining state-of-the-art CNN architectures with effective XAI techniques.

These comparative insights are crucial for understanding which architectural improvements translate into clinical benefits.

---

## References

1. **Pacheco & Krohling (2020)** – Discusses the impact of patient clinical information on automated skin cancer detection.  
   [Access more details](https://pubmed.ncbi.nlm.nih.gov/31760271/)
2. **Siegel et al. (2023)** – Provides updated cancer statistics, emphasizing the prevalence of skin cancer.  
   [Link to article](https://doi.org/10.3322/caac.21754)
3. **World Health Organization (2023)** – Information on ultraviolet (UV) radiation and skin cancer.  
   [WHO Q&A on UV radiation](https://www.who.int/news-room/questions-and-answers/item/radiation-ultraviolet-(uv)-radiation-and-skin-cancer)
4. **Tabassum et al. (2022)** – Empirical study on pre-trained CNN models for COVID-19 CT scan images, highlighting the relevance of CNNs in medical imaging.  
   [Conference proceeding](https://ieeexplore.ieee.org/document/9795936)
5. **Shawon et al. (2023)** – Uses explainable cost-sensitive deep neural networks for brain tumor detection, underscoring the importance of XAI in medical diagnostics.
6. **Wang et al. (2020)** – Introduces Score-CAM, a method for generating visual explanations in CNNs.  
   [Score-CAM paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.pdf)
7. **Smilkov et al. (2017)** – Proposes SmoothGrad for reducing noise in sensitivity maps.  
   [SmoothGrad paper](https://arxiv.org/abs/1706.03825)
8. **Innani et al. (2022)** – Presents a cascaded approach for skin lesion analysis using an encoder-decoder framework.  
   [ResearchGate publication](https://www.researchgate.net/publication/380199255_An_Interpretable_Deep_Learning_Approach_for_Skin_Cancer_Categorization?enrichId=rgreq-84b4a1fd54afa17647574b40b343bcdd-XXX)
9. **Tschandl et al. (2018)** – Describes the HAM10000 dataset, a benchmark for skin lesion analysis.  
   [HAM10000 dataset details](https://www.nature.com/articles/sdata201818)
10. **Chollet (2017)** – Introduces the Xception architecture.  
    [Xception on arXiv](https://arxiv.org/abs/1610.02357)
11. **Tan & Le (2021)** – Presents EfficientNetV2 architectures.  
    [EfficientNetV2 paper](https://arxiv.org/abs/2104.00298)
12. **Szegedy et al. (2017)** – Details InceptionResNetV2 architecture.  
    [InceptionResNetV2 on arXiv](https://arxiv.org/abs/1602.07261)

[^1]: https://www.researchgate.net/publication/380199255_An_Interpretable_Deep_Learning_Approach_for_Skin_Cancer_Categorization

[^2]: https://doi.org/10.1109/ICCIT60459.2023.10508527

[^3]: https://ieeexplore.ieee.org/document/8815225

[^4]: https://www.nature.com/articles/s41598-021-89239-3

[^5]: https://arxiv.org/abs/1610.02357

[^6]: https://arxiv.org/abs/2104.00298

[^7]:  https://arxiv.org/abs/1602.07261

[^8]: https://arxiv.org/abs/2104.00298
