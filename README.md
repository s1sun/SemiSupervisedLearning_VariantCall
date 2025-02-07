# Enhanced Somatic Variant Detection Using Multi-Detectors and Semi-Supervised Learning Approach
Somatic variant detection relies on multiple variant callers, such as Mutect2, Strelka, and VarScan, each with its own strengths, limitations, and detection algorithms. However, the sensitivity of any single detector typically reaches only about 85%, leaving room for improvement.

## Multi-Detectors Integration Strategy
To improve accuracy, I integrate the results from multiple detectors and classify variants into two groups:

1. High-Confidence Positive Group – Variants that pass all detection criteria across multiple tools, representing high-confidence calls.
2. Unlabeled (Mixed) Group – Variants detected by only a subset of tools. This group contains a mix of true positives and false positives, requiring further classification.

## Semi-Supervised Learning for Variant Classification
To refine the results, I apply a semi-supervised learning approach using the high-confidence group to train a model and classify the mixed group. The workflow includes:

1. Feature Engineering – Designing features based on variant quality metrics, read depth, allele frequency, and other relevant factors.
2. Feature Selection – Applying selection functions to identify the most informative features.
3. Model Training – Using a Random Forest classifier trained on the high-confidence group
4. Prediction & Classification – Applying the trained model to classify the two groups into true positives and false positives.

This approach enhances somatic variant detection accuracy by leveraging multiple detectors and a machine learning-driven refinement process.

Sensitivity raises from 88% before to 92% after.
