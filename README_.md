
# Advanced ECG Analysis and Heart Disease Classification

## Project Overview
This project implements an advanced Electrocardiogram (ECG) analysis system using multiple machine learning models for heart disease classification. It builds upon the research presented in "Comparative Analysis of CNN, RNN, DNN, and XGBoost Models for Automated Heart Disease Diagnosis Using ECG Data" and provides a practical implementation through a Flask web application.
The system utilizes a large-scale ECG database covering over 10,000 patients to train and evaluate various machine learning models for accurate heart disease diagnosis.

## Features

- Multi-model heart disease classification (CNN, RNN, DNN, XGBoost)
- Advanced ECG feature extraction
- Interactive web interface for ECG analysis
- Real-time ECG signal visualization
- Comparative model performance display
- Metadata extraction from ECG header files
- Preprocessing and standardization of ECG data

Dataset link : https://physionet.org/content/ecg-arrhythmia/1.0.0/ 

## Models

- **CNN (Convolutional Neural Network):** Hybrid CNN (H-CNN) with improved architecture
- **RNN (Recurrent Neural Network):** Specialized for temporal ECG data analysis
- **DNN (Deep Neural Network):** Fully connected neural network for ECG classification
- **XGBoost (Extreme Gradient Boosting):** Ensemble learning model for ECG analysis

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ecg-analysis-project.git
    cd ecg-analysis-project
    ```
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download pre-trained models and place them in the `models/` directory:
    - `cnn_model.h5`
    - `rnn_model.h5`
    - `dnn_model.h5`
    - `xgboost_model.json`

## Usage

1. Start the Flask application:
    ```bash
    python app.py
    ```
2. Open a web browser and navigate to `http://localhost:5000`
3. Upload an ECG file (.dat and .hea) and provide patient information
4. View the analysis results and model predictions

## File Structure

- `app.py`: Main Flask application
- `models/`: Directory containing pre-trained models
- `templates/`: HTML templates for the web interface
- `static/`: CSS and JavaScript files
- `uploads/`: Temporary storage for uploaded ECG files

## ECG Feature Extraction
The system extracts advanced features from ECG signals, including:

### Time-domain features:
- RR intervals
- PR interval
- QT interval
- QRS duration
- P wave duration
- T wave duration

### Frequency-domain features:
- Very Low Frequency (VLF) component
- Low Frequency (LF) component
- High Frequency (HF) component

### Statistical features:
- Mean
- Median
- Standard deviation
- Skewness
- Kurtosis

### Amplitude features:
- P wave amplitude
- R wave amplitude
- T wave amplitude
- ST segment amplitude

## Model Architecture

### CNN Model (H-CNN)
The Hybrid CNN (H-CNN) model is an improved version of the traditional CNN architecture. It incorporates:
- Multiple convolutional layers with increasing filter sizes
- Batch normalization for improved stability
- Max pooling layers for downsampling
- Global Average Pooling to reduce overfitting
- Dense layers for final classification

### RNN Model
The RNN model utilizes LSTM (Long Short-Term Memory) units to capture temporal dependencies in ECG signals. It consists of:
- Bidirectional LSTM layers
- Dropout layers for regularization
- Dense layers for classification

### DNN Model
The DNN model is a fully connected neural network designed for ECG classification. It features:
- Multiple dense layers with decreasing neuron counts
- Batch normalization after each dense layer
- Dropout for regularization
- Softmax output layer for multi-class classification

### XGBoost Model
The XGBoost model is an ensemble learning method that uses gradient boosting on decision trees. It is configured with:
- Maximum tree depth of 6
- Learning rate of 0.1
- 100 estimators
- Subsampling rate of 0.8
- Column sampling rate of 0.8

## Performance Comparison
| Model      | Accuracy | Precision | Recall | F1-score | AUC  |
|------------|----------|-----------|--------|----------|------|
| E-CNN      | 96.08%   | 96.00%    | 96.00% | 96.00%   | 0.99 |
| RNN        | 88.76%   | 89.00%    | 89.00% | 88.00%   | 0.99 |
| DNN        | 88.05%   | 88.00%    | 88.00% | 88.00%   | 0.99 |
| XGBoost    | 94.31%   | 94.25%    | 94.31% | 94.22%   | 0.9971 |

## Visualization

- **ECG signal plot:** The application generates a plot of the uploaded ECG signal for visual inspection.
- **AUC and ROC curves:** The system calculates and displays Area Under the Curve (AUC) and Receiver Operating Characteristic (ROC) curves for each model to evaluate their performance.

## Project Management
This project was managed using:
- Jira for task tracking and sprint planning
- GitLab for version control and CI/CD

## Future Work

- Integration of more ECG databases to improve model generalization
- Improvement of model interpretability using techniques like layer-wise relevance propagation
- Development of a mobile application for ECG analysis
- Implementation of continuous learning capabilities to update models with new data
- Enhancement of the Flask application with additional features and improved user interface

## Contributing
Contributions to this project are welcome. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- Zheng, J., et al. (2020). A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. Scientific Data, 7(1), 48.
- Kim, H. Y., & Sunwoo, M. (2024). ECG classification using deep learning: A comparative study.
- The open-source community for providing essential libraries and tools used in this project.
