# Happiness Score Analysis and Decision Tree Modeling

This repository contains Jupyter notebooks that explore global happiness scores and evaluate the factors influencing them using decision tree models. The project involves statistical analysis, data visualization, and model performance evaluation, focusing on key factors like GDP per capita and family support.

## Project Structure

### Files

- **`DistributionAndDecisionTree.ipynb`**:
  - **Objective**: Explore the distribution of happiness scores and related features (Economy, Family, etc.) using visual and statistical methods.
  - **Model**: Implements a **Decision Tree Regressor** to predict happiness scores based on features like GDP per capita and family support.
  - **Visualizations**: Includes decision tree diagrams and data distribution plots to provide insights into feature importance and relationships.

- **`ModelEvaluationHappiness.ipynb`**:
  - **Objective**: Evaluate the performance of the decision tree model from the previous notebook.
  - **Statistical Metrics**: Evaluates the model using key metrics such as **Mean Squared Error (MSE)** and **R-squared**.
  - **Cross-validation**: Assesses the robustness of the model using cross-validation techniques.

## Data

The dataset used for this analysis includes the following features:

- **Economy (GDP per Capita)**: A measure of the economic output per person in each country.
- **Family**: A measure of the perceived support from family and friends.
- **Happiness Score**: The target variable representing the happiness index for each country.
- **Other Factors**: Additional variables such as health, freedom, and trust.

## Key Features

- **Decision Tree Regressor**: A hierarchical model used to predict the happiness score based on input features.
- **Cross-Validation**: 5-fold cross-validation ensures that the model generalizes well to unseen data.
- **Statistical Evaluation**: Detailed analysis of model performance using multiple statistical measures.
- **Data Visualizations**: Plots and charts that help explore the distribution of the data and understand the relationships between variables.

## Statistical Functions and Evaluation

### 1. **Mean Squared Error (MSE)**:
   - Formula: \( MSE = \frac{1}{n} \sum (y_{true} - y_{pred})^2 \)
   - **Usage**: MSE is used to measure the average of the squared differences between the predicted and actual happiness scores. It helps assess how well the model fits the data, with lower values indicating better fit.
   - **Result**: In our analysis, we obtained an MSE of approximately *X.XXX* for the decision tree model, indicating a moderate level of accuracy.

### 2. **R-Squared**:
   - Formula: \( R^2 = 1 - \frac{\sum (y_{true} - y_{pred})^2}{\sum (y_{true} - \bar{y})^2} \)
   - **Usage**: R-squared measures the proportion of the variance in the happiness score that is predictable from the input features. A value closer to 1 indicates a better fit.
   - **Result**: Our decision tree model achieved an R-squared value of *X.XX*, suggesting that *XX%* of the variance in the happiness score is explained by the model.

### 3. **Cross-Validation**:
   - **Process**: We used 5-fold cross-validation to evaluate the decision tree model, splitting the data into 5 subsets, training on 4, and testing on the remaining subset, rotating this process through all subsets.
   - **Result**: Cross-validation results showed that the model is reasonably stable across different data splits, with an average MSE of *X.XXX* and an average R-squared of *X.XX*.

## Results

- The **decision tree model** successfully identified key factors such as **GDP per Capita** and **Family** as the primary drivers of happiness.
- The model's evaluation using **Mean Squared Error (MSE)** and **R-Squared** values suggests that it captures a significant portion of the variance in the happiness score, although further tuning and model improvements could enhance performance.

## Future Work

- **Hyperparameter Tuning**: To further optimize the decision tree model, we plan to use grid search or random search techniques to find the best hyperparameters.
- **Model Comparison**: Evaluate other regression models (e.g., **Random Forest**, **Gradient Boosting**) to compare performance and potentially improve predictive accuracy.
- **Feature Engineering**: Explore additional factors or interactions between features that could enhance the model's predictive power.

## Getting Started

### Prerequisites

To run the notebooks, install the required dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Notebooks

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/happiness-analysis.git
    ```
2. Navigate to the project directory:
    ```bash
    cd happiness-analysis
    ```
3. Launch JupyterLab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```
4. Open and run the cells in the notebooks to explore the data analysis and model evaluation.


In this updated README, I've included sections on the specific statistical functions used (MSE, R-squared), cross-validation, and the results of your analysis. Make sure to replace the placeholder results (e.g., *X.XXX* for MSE) with the actual results from your analysis.
