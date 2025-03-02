# Neural Network Hyperparameter Optimization

## Overview
This project focuses on optimizing hyperparameters for a neural network model trained on the **Kuzushiji-MNIST (KMNIST)** dataset. The goal is to find the best combination of hyperparameters to achieve the highest accuracy. The project is divided into three parts:

1. **Predefined Hyperparameter Combinations:** We experiment with a set of predefined hyperparameter combinations and identify the best-performing one.
2. **User-Defined Hyperparameters:** The user provides hyperparameter values as input, and the model is trained accordingly.
3. **Exhaustive Grid Search (Pending Execution):** We test all possible hyperparameter combinations using a brute-force approach. Due to its high computational complexity (*O(n⁸)*), full execution is pending as it takes approximately **40 hours** to complete.

---

## Dataset
We use the **KMNIST** dataset, which consists of 70,000 grayscale images of hand-written Japanese characters:
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images
- **Validation Set:** 10% of the training data

Each image is **28x28 pixels** and belongs to one of **10 classes**.

---

## Project Breakdown

### 1️⃣ Predefined Hyperparameter Combinations
In this part, we test 8 predefined hyperparameter sets, varying:
- **Number of Hidden Layers**
- **Number of Neurons per Layer**
- **Activation Functions** (`ReLU`, `Sigmoid`)
- **Optimizers** (`Adam`, `SGD`, `RMSProp`, `Nadam`, `Momentum`, `Nesterov`)
- **Learning Rate**
- **Batch Size**
- **Weight Decay**
- **Epochs**

For each combination:
- We **train the model** on the training set
- We **evaluate validation and test accuracy**
- We **select the best-performing model** based on validation accuracy
- We **train the best model** again using the full training + validation set and evaluate on the test set

#### Output:
- **Best Model Configuration** (Hyperparameters with the highest validation accuracy)
- **Confusion Matrix** for final model evaluation

---

### 2️⃣ User-Defined Hyperparameters
In this part, the user **inputs hyperparameter values**, and the model is trained accordingly.

#### Steps:
- User provides:
  - Number of layers and neurons
  - Learning rate, optimizer, activation function
  - Weight decay, batch size, and epochs
- The model is trained using these parameters
- Validation and test accuracy are reported

#### Output:
- **Model performance based on user-defined parameters**

---

### 3️⃣ Exhaustive Grid Search (Pending Execution)
This method aims to find the absolute best combination by testing **all possible hyperparameter values**:

#### Approach:
- We define a **range of values** for each hyperparameter (2-3 values per parameter)
- All **possible combinations** of these values are tested
- The best combination is identified

#### Challenge:
- **Time Complexity:** *O(n⁸)* (Exponentially increasing combinations)
- **Estimated Execution Time:** ~**40 hours** (Full execution is pending)

---

## Results & Observations
- The best model from predefined combinations achieved **high accuracy** using `Adam` optimizer with `ReLU` activation.
- The user-defined approach allows customization but requires careful tuning.
- The exhaustive grid search is theoretically optimal but practically infeasible due to time constraints.

---

## Future Improvements
- Implement a **more efficient search** method (e.g., **Random Search, Bayesian Optimization**)
- Utilize **GPU acceleration** to reduce execution time
- Optimize the dataset preprocessing for faster training

---

## Installation & Usage
1. Install dependencies:
   ```bash
   pip install tensorflow tensorflow-datasets numpy pandas matplotlib seaborn scikit-learn
   ```
2. Run the script:
   ```bash
   python train_model.py
   ```
3. Follow prompts for **user-defined training** or let the predefined models run.

---

## Author
🚀 Developed by G.Sushrut Reddy

---


