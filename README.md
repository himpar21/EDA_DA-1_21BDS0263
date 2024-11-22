# EDA_DA 21BDS0263
# HDI Dataset Analysis and Visualization

This project involves analyzing and visualizing the Human Development Index (HDI) dataset. It includes cleaning, feature engineering, and visualizing the dataset using Python scripts and a Streamlit web app.

---

## Project Overview

### Files in the Repository
- **`app.py`**: A Streamlit web application for visualizing and comparing original and cleaned datasets through various plots.
- **`test.py`**: A Python script for preprocessing the dataset, including missing value imputation, duplicate handling, feature engineering, encoding, and scaling.
- **Datasets**:
  - **`HDI.csv`**: The original dataset.
  - **`cleaned_HDI.csv`**: The dataset after preprocessing.
  - **`standardized_data.csv`**: The dataset after standardization.
  - **`normalized_data.csv`**: The dataset after normalization.

---

## Installation and Setup

### Prerequisites
Ensure you have Python 3.8+ installed along with `pip`. Clone this repository to your local machine.

### Install Dependencies
Run the following command to install all required packages:
```bash
pip install -r requirements.txt
```

### Create a `requirements.txt` File
Include the following dependencies:
```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Running the Application
1. **Data Preprocessing**:
   Execute `test.py` to preprocess the dataset:
   ```bash
   python test.py
   ```
   This will clean the dataset and generate the following output files:
   - `cleaned_HDI.csv`
   - `standardized_data.csv`
   - `normalized_data.csv`

2. **Launch the Web App**:
   Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   The web app will open in your default browser, providing options for visualizing various aspects of the dataset.

---

## Features
1. **Data Preprocessing**:
   - Handling missing values (mean/mode imputation).
   - Removing duplicate rows.
   - Log and square root transformations for skewed data.
   - Encoding categorical variables using Label Encoding and One-Hot Encoding.
   - Scaling numeric data using StandardScaler and MinMaxScaler.

2. **Data Visualization**:
   - Histograms
   - Bar Plots
   - Scatter Plots
   - Box Plots
   - Violin Plots
   - Heatmaps

3. **Dataset Comparison**:
   Compare original and cleaned datasets for a specific column or across the entire dataset.

4. **Feature Selection**:
   - ANOVA F-test
   - Mutual Information Regression

---

## Dataset
The dataset, `HDI.csv`, contains information on Human Development Index metrics across various countries and years.

---

## Future Enhancements
- Add more visualization options to the Streamlit app.
- Incorporate machine learning models for predictive analysis.
- Extend the app for real-time dataset uploads and processing.

---

## Contributing
Feel free to fork the repository and submit pull requests for new features or bug fixes.

---

## License
This project is licensed under the MIT License.
