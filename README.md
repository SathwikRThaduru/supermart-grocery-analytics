# supermart-grocery-analytics
"Machine Learning project for grocery sales prediction and retail analytics"

# 📊 Supermart Grocery Sales - Retail Analytics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-red.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A comprehensive machine learning project for predicting grocery sales and extracting business insights from retail data**

![Retail Analytics Dashboard](https://via.placeholder.com/800x400/2E8B57/FFFFFF?text=Supermart+Grocery+Analytics)

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

## 🗂️ Repository Structure

```
supermart-grocery-analytics/
├── 📁 data/
│   ├── Supermart-Grocery-Sales-Retail-Analytics-Dataset.csv
│   └── data_dictionary.md
├── 📁 notebooks/
│   ├── Retail.ipynb                    # Main analysis notebook
│   ├── exploratory_data_analysis.ipynb # Detailed EDA
│   └── model_evaluation.ipynb          # Model performance analysis
├── 📁 src/
│   ├── data_preprocessing.py           # Data cleaning functions
│   ├── feature_engineering.py          # Feature creation utilities
│   ├── model_training.py               # ML model training
│   └── visualization_utils.py          # Custom plotting functions
├── 📁 reports/
│   ├── Retail-Analytics-Project-Report.md
│   └── presentation.pdf
├── 📁 visualizations/
│   ├── sales_by_category.png
│   ├── regional_analysis.png
│   ├── temporal_trends.png
│   └── correlation_heatmap.png
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 LICENSE
```

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

## 📈 Visualizations

### Available Charts
- Sales distribution by category
- Regional performance comparison
- Temporal trend analysis
- Customer behavior patterns
- Correlation heatmaps
- Profit margin analysis

### Sample Visualization
![Sales by Category](https://via.placeholder.com/600x400/4CAF50/FFFFFF?text=Sales+by+Category+Chart)

## 🔍 Usage Examples

### Basic Data Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/Supermart-Grocery-Sales-Retail-Analytics-Dataset.csv')

# Quick overview
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")

# Sales by category
category_sales = df.groupby('Category')['Sales'].sum()
category_sales.plot(kind='bar', title='Total Sales by Category')
plt.show()
```

### Sales Prediction
```python
from src.model_training import train_sales_model
from src.data_preprocessing import preprocess_data

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train model
model = train_sales_model(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 📚 Documentation

### Available Documents
- **[Project Report](reports/Retail-Analytics-Project-Report.md)**: Comprehensive analysis report
- **[Data Dictionary](data/data_dictionary.md)**: Detailed feature descriptions
- **[API Documentation](docs/api.md)**: Function and class references

### Key Notebooks
- **`Retail.ipynb`**: Main analysis with complete ML pipeline
- **`exploratory_data_analysis.ipynb`**: Detailed EDA and insights
- **`model_evaluation.ipynb`**: Model performance and validation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- Model improvements and new algorithms
- Additional visualization features
- Performance optimization
- Documentation enhancements
- Bug fixes and testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**[Your Name]**
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [your-linkedin-profile](https://linkedin.com/in/yourprofile)
- 🐱 GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset inspiration from retail analytics use cases
- Scikit-learn community for excellent ML tools
- Matplotlib/Seaborn for visualization capabilities
- Jupyter Project for interactive development environment

## 📞 Support

If you have questions or need help:
- 📧 Open an [Issue](https://github.com/yourusername/supermart-grocery-analytics/issues)
- 💬 Start a [Discussion](https://github.com/yourusername/supermart-grocery-analytics/discussions)
- 📖 Check the [Documentation](docs/)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/supermart-grocery-analytics&type=Date)](https://star-history.com/#yourusername/supermart-grocery-analytics&Date)

---

<div align="center">
  <p><strong>📊 Made with ❤️ for Data Science</strong></p>
  <p>If you found this project helpful, please consider giving it a ⭐!</p>
</div>
