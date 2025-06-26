# Heart Disease Classification and Clustering Analysis

## Overview

This project focuses on analyzing heart disease data using various classification and clustering techniques. The primary goal is to compare the accuracy of **Decision Tree** and **Naive Bayes** classifiers, and to evaluate the impact of feature selection on model performance. Additionally, clustering algorithms are applied to explore inherent patterns in the dataset.

## Features

- **Data Preprocessing**: Handling missing values, standardization, and feature-label separation.
- **Classification Models**: Implementation of Decision Tree and Naive Bayes classifiers.
- **Feature Selection**: Reduction of features using Recursive Feature Elimination (RFE).
- **Clustering Algorithms**: KMeans, Hierarchical Clustering, MeanShift, and DBSCAN.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and Silhouette Score.

## Installation

To run this project, ensure you have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. **Data Loading and Preprocessing**:
   - Load the dataset and handle missing values.
   - Standardize the features and split the data into training and testing sets.

2. **Classification**:
   - Train and evaluate Decision Tree and Naive Bayes models.
   - Compare model performance before and after feature reduction.

3. **Clustering**:
   - Apply KMeans, Hierarchical Clustering, MeanShift, and DBSCAN to explore data clusters.
   - Visualize clustering results using PCA for dimensionality reduction.

## Results

- **Classification**: The accuracy of Decision Tree and Naive Bayes models is compared, both with and without feature reduction.
- **Clustering**: Performance metrics such as Silhouette Score, ARI, and Homogeneity are visualized for different clustering algorithms.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or additional features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: Heart Disease Dataset from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Contact

For any inquiries, please contact **Iman Khalil Rahmani** at [imankhtech@gmail.com](mailto:imankhtech@gmail.com).

---

