# Fraud Detection using Neural Network

## Overview
This project focuses on detecting fraudulent transactions using machine learning techniques. The goal is to build a model that can accurately classify transactions as either fraudulent or non-fraudulent based on historical data. The project includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Project Description
Fraud detection is a critical task in financial systems to prevent unauthorized transactions and minimize losses. This project uses a dataset of transaction records to build a machine learning model that can identify fraudulent transactions with high accuracy. The model is trained and evaluated using various techniques, including Random Forest, Neural Networks, and hyperparameter tuning.

## Dataset

The data used in this experiment can be found at **https://www.kaggle.com/datasets/ealaxi/paysim1**

The dataset used in this project contains transaction records with the following features:

- **step**: Represents a unit of time (e.g., hour, day).
- **type**: Type of transaction (e.g., CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
- **amount**: The amount of money involved in the transaction.
- **nameOrig**: The name or ID of the customer who initiated the transaction.
- **oldbalanceOrg**: The balance of the origin account before the transaction.
- **newbalanceOrig**: The balance of the origin account after the transaction.
- **nameDest**: The name or ID of the recipient of the transaction.
- **oldbalanceDest**: The balance of the destination account before the transaction.
- **newbalanceDest**: The balance of the destination account after the transaction.
- **isFraud**: A binary indicator (0 or 1) representing whether the transaction is fraudulent.
- **isFlaggedFraud**: A binary indicator (0 or 1) representing whether the transaction was flagged as fraud by the system.

The dataset is highly imbalanced, with a small percentage of transactions being fraudulent.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fraud-detection.git
   cd fraud-detection
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Usage
1. Open the Jupyter notebook (`fraud_detection.ipynb`) in your browser.
2. Run the cells sequentially to:
   - Load and preprocess the data.
   - Perform exploratory data analysis (EDA).
   - Train and evaluate machine learning models.
3. Adjust the code as needed for your specific use case.

## Methodology
### Data Preprocessing
- Handle missing values and encode categorical variables.
- Scale numerical features using `StandardScaler`.
- Balance the dataset using techniques like SMOTE or undersampling.
- For this particular use case, we used undersampling because two reasons.
    1. The minority class had about 8000 rows which is big enough presence.
    2. using SMOTE will generate too much of synthetic data.

### Model Training
- Train a **Neural Network**.
- Use **Grid Search** and **Random Search** for hyperparameter tuning (I used **Manual Search** because it cost less computing resources).
- Adjust the decision threshold to optimize for precision or recall.

### Evaluation
- Evaluate the models using metrics like **precision, recall, F1-score, and ROC-AUC**.
- Analyze the **confusion matrix** to understand the trade-off between False Positives (FP) and False Negatives (FN).

## Results
The final model achieves the following performance metrics:

- **Accuracy**: 98%
- **Precision**: 97% (for fraudulent transactions)
- **Recall**: 99% (for fraudulent transactions)
- **F1-Score**: 98% (for fraudulent transactions)
- **ROC-AUC Score**: 0.997

The confusion matrix shows:

- **True Positives (TP)**: 2444
- **False Positives (FP)**: 88
- **True Negatives (TN)**: 2379
- **False Negatives (FN)**: 17

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your fork.
4. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Thanks to the dataset provider for making the data publicly available.
- Special thanks to the open-source community for providing tools and libraries that made this project possible.

## Contact
For questions or feedback, feel free to reach out:

👤 [**Devang Vamja**](https://devangvamja-portfolio.netlify.app/)  
📧 [devangvamja2000@gmail.com](mailto:devangvamja2000@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/DevangVamja)  
[![For More Repo](https://img.shields.io/badge/GitHub-000?logo=github&logoColor=white)](https://github.com/DevangVamja)