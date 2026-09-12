# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## Overview

This project uses machine learning to analyze real estate customer data and discover groups of buyers with similar demographic, geographic, financing, acquisition, and satisfaction characteristics.

The project applies data cleaning, exploratory data analysis, feature engineering, categorical encoding, feature scaling, K-Means clustering, cluster evaluation, and investment profiling.

## Objectives

- Segment real estate buyers using unsupervised machine learning.
- Understand buyer demographics and behavior.
- Identify investment-oriented customer groups.
- Analyze financing and acquisition-channel patterns.
- Build business-readable investment profiles.
- Provide a foundation for targeted marketing and personalized recommendations.

## Dataset

The dataset contains:

- `client_id` — unique customer identifier
- `client_type` — Individual / Corporate
- `gender` — buyer gender
- `country` — country of residence
- `region` — geographic region
- `date_of_birth` — used to calculate age
- `acquisition_purpose` — Investment / Personal use
- `loan_applied` — financing indicator
- `referral_channel` — acquisition source
- `satisfaction_score` — customer satisfaction rating

The included dataset is synthetic and intended for educational/project-development use.

## Machine Learning

The primary algorithm is **K-Means Clustering**.

Categorical variables are one-hot encoded and numerical variables are standardized before clustering.

The pipeline evaluates K values from 2 to 8 using the **Silhouette Score** and selects the best-scoring K as a baseline.

## Project Structure

```text
real-estate-buyer-segmentation/
├── data/
│   └── real_estate_buyer_segmentation_dataset.csv
├── notebooks/
├── outputs/
│   └── figures/
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   └── investment_profiling.py
├── app.py
├── run_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Complete Pipeline

From the project root:

```bash
python run_pipeline.py
```

This generates:

- cleaned buyer data
- EDA charts
- cluster evaluation results
- cluster assignments
- PCA cluster visualization
- investment profiles

## Run the Dashboard

```bash
streamlit run app.py
```

## Business Applications

The resulting segments can support:

- targeted marketing
- buyer-specific communication
- investment lead identification
- geographic market analysis
- acquisition-channel optimization
- customer experience analysis
- personalized property recommendations

## Future Scope

- Real-time CRM/customer-data integration
- Property recommendation engine
- Investment scoring
- Real estate price and market-trend data
- Geographic maps
- Advanced clustering methods
- Predictive models
- Real-time dashboard

## Author

**Preeti Patel**

B.Tech CSE – AI/ML  
LNCT University, Bhopal

## License

Educational and academic project.
