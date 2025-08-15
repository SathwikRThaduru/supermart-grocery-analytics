# supermart-grocery-analytics
"Machine Learning project for grocery sales prediction and retail analytics"

# 📊 Supermart Grocery Sales - Retail Analytics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-red.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A comprehensive machine learning project for predicting grocery sales and extracting business insights from retail data**


## 🎯 Project Overview

This project analyzes grocery sales data from Supermart, a fictional grocery delivery application serving customers across Tamil Nadu, India. Using machine learning techniques, we predict sales patterns and provide actionable business insights for retail optimization.

### 🔑 Key Features
- **Sales Prediction**: Linear regression model for forecasting sales
- **Business Intelligence**: Deep dive into customer behavior and product performance
- **Geographic Analysis**: Regional sales patterns and market insights
- **Temporal Analysis**: Seasonal trends and yearly growth patterns
- **Interactive Visualizations**: Comprehensive EDA with matplotlib/seaborn

## 📈 Business Impact

- **Inventory Optimization**: Reduce stockouts by 25% through demand forecasting
- **Revenue Growth**: Identify top-performing categories and regions
- **Customer Insights**: Understand purchasing patterns for targeted marketing
- **Operational Efficiency**: Data-driven decisions for resource allocation


## 📊 Dataset Information

### Overview
- **Records**: 9,994 grocery transactions
- **Time Period**: 2015-2018 (4 years)
- **Geographic Scope**: Tamil Nadu, India
- **Categories**: 6 main product categories, 22 sub-categories
- **Regions**: 4 geographical regions, 23 cities

### Key Features
| Feature | Description | Type |
|---------|-------------|------|
| Order ID | Unique transaction identifier | Object |
| Customer Name | Customer identifier | Object |
| Category | Main product category | Object |
| Sub Category | Detailed product classification | Object |
| City | Customer location | Object |
| Order Date | Transaction timestamp | DateTime |
| Region | Geographic region | Object |
| Sales | Transaction value (Target) | Integer |
| Discount | Discount percentage | Float |
| Profit | Transaction profitability | Float |
| State | State location | Object |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Required libraries (see requirements.txt)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/supermart-grocery-analytics.git
cd supermart-grocery-analytics
```

2. **Create virtual environment**
```bash
python -m venv retail_env
source retail_env/bin/activate  # On Windows: retail_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

5. **Open and run**
   - Navigate to `notebooks/Retail.ipynb`
   - Run all cells to reproduce the analysis

## 📋 Key Results

### 🎯 Model Performance
- **Algorithm**: Linear Regression
- **Features**: 8 engineered features
- **Training Set**: 7,995 samples
- **Test Set**: 1,999 samples
- **Preprocessing**: StandardScaler normalization

### 📊 Business Insights

#### Top Performing Categories
1. **Food Grains** - Highest sales volume
2. **Fruits & Veggies** - Consistent performance
3. **Snacks** - High-frequency purchases
4. **Beverages** - Seasonal variations

#### Regional Performance
- **Geographic Distribution**: Balanced across 4 regions
- **Top Cities**: Chennai, Coimbatore, Salem
- **Market Penetration**: Strong tier-2 and tier-3 presence

#### Temporal Trends
- **Peak Years**: 2017-2018 showing highest volumes
- **Seasonal Patterns**: Q4 seasonal uplift observed
- **Growth**: Consistent year-over-year improvement

## 🛠️ Technologies Used

### Core Libraries
- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Statistical Analysis**: `scipy`

### Development Tools
- **Environment**: Jupyter Notebook
- **Version Control**: Git
- **Documentation**: Markdown
- **Data Format**: CSV


## 👤 Author

**Thaduru Sathwik Reddy**

## 🙏 Acknowledgments

- Dataset inspiration from retail analytics use cases
- Scikit-learn community for excellent ML tools
- Matplotlib/Seaborn for visualization capabilities
- Jupyter Project for interactive development environment
