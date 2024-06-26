# SVM Classifier on Digits Dataset

This project involves using Support Vector Machine (SVM) classifiers to recognize handwritten digits from the digits dataset. We employ both the RBF kernel and the linear kernel to train the models and evaluate their performance.

## Output


https://github.com/sarvesh-2109/SVM/assets/113255836/d055709c-ccf3-424b-b7b4-9ccfb0c58566



## Dataset

The dataset used is the `digits` dataset from the `sklearn.datasets` module, which contains 8x8 pixel images of digits (0-9). The dataset includes:

- `data`: Array of 8x8 images.
- `target`: Array of labels (0-9).

## Project Overview

1. **Data Loading and Preprocessing**: Load the digits dataset and convert it into a Pandas DataFrame for easier manipulation.
2. **Splitting Data**: Split the data into training and testing sets.
3. **Model Training**: Train SVM classifiers with different kernels (RBF and Linear).
4. **Model Evaluation**: Evaluate the performance of each model.

## Dependencies

- pandas
- scikit-learn

## Code Explanation

### 1. Importing Libraries

```python
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
```

### 2. Loading Dataset

```python
digits = load_digits()
dir(digits)

df = pd.DataFrame(digits.data, digits.target)
df.head()

df['target'] = digits.target
df.head()
```

### 3. Splitting Data

```python
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis='columns'), df.target, test_size=0.3)
```

### 4. Using RBF Kernel

```python
rbf_model = SVC(kernel='rbf')
rbf_model.fit(X_train, y_train)

rbf_model.score(X_test, y_test)
```

### 5. Using Linear Kernel

```python
linear_model = SVC(kernel='linear')
linear_model.fit(X_train, y_train)

linear_model.score(X_test, y_test)
```

## Results

- The accuracy of the SVM classifier with the RBF kernel.
- The accuracy of the SVM classifier with the linear kernel.

## Conclusion

This project demonstrates the application of SVM classifiers using different kernels on the digits dataset. The comparison between the RBF and linear kernels provides insights into the performance differences based on the kernel choice.

