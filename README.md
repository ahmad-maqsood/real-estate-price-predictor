# Pakistan Real Estate Price Predictor

A classical machine learning project that predicts property prices in Pakistan using real listing data from Zameen.com and Graana.

## Dataset

**Source:** [Pakistan Real Estate Property Listings Dataset](https://www.kaggle.com/datasets/hassaanmustafavi/pakistan-real-estate-property-listings-dataset) (Kaggle)

- Raw rows: 116,033
- Final rows after cleaning: ~84,977
- Features: property type, area, bedrooms, bathrooms, city, price

## Project Structure

```
real-estate-price-predictor/
│
├── data/
│   ├── property_data.csv               # Raw dataset
│   ├── cleaned_property_data.csv       # After preprocessing
│   └── cleaned_property_data_v2.csv    # After visualization fixes
│
├── model/
│   ├── random_forest_model.pkl         # Generated after running notebook 3
│   └── ohe_encoder.pkl                 # Generated after running notebook 3
│
├── 1_data_analysis_and_preprocessing.ipynb               # Data cleaning and preprocessing
├── 2_data_visualization.ipynb          # EDA and visualizations
├── 3_building_model.ipynb              # Model training and evaluation
├── 4_predict_using_model.py            # Prediction script
└── README.md
```

## Notebooks

### 1. Data Analysis & Preprocessing
- Filtered dataset to "For Sale" properties only
- Cleaned `type` column: fixed duplicates, grouped rare categories (29 → 11 unique values)
- Cleaned `bedroom` and `bath` columns: handled mixed types, dashes, and float strings
- Dropped high cardinality `location` column (3002 unique values), kept `location_city`
- Grouped rare cities into "Other" (242 → 17 unique values)
- Converted `price` from string format (PKR x Crore/Lac/Arab/Thousand) to numeric rupees
- Converted `area` from mixed units (Marla/Kanal/Sqft/Sq.Yd/Ft²/Sqm) to unified Marla
- Removed outliers: prices below PKR 100,000, area above 99th percentile (1600 Marla)

### 2. Data Visualization
- Price distribution showing right skew and effect of log transformation
- Property count and average price by type and city
- Area vs Price scatter plot (raw and log scale)
- Bedroom and bath effect on price
- Correlation heatmap of numerical features

### 3. Model Training
- Applied OHE on categorical columns (type, location_city)
- Applied log transformation on price
- 80/20 train test split
- Compared 3 models using RandomizedSearchCV with 5-fold cross validation:

| Model | R² Score |
|---|---|
| Linear Regression | 0.54 |
| Decision Tree | 0.77 |
| **Random Forest** | **0.79** |

- Best params: `n_estimators=200`, `max_depth=None`

## Setup

### Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Running the Project

1. Run `1_data_analysis_and_preprocessing.ipynb` to clean and preprocess the data
2. Run `2_data_visualization.ipynb` for EDA
3. Run `3_building_model.ipynb` to train and save the model
4. Run `4_predict_using_model.py` to make predictions

> Note: `model/*.pkl` files are not tracked in git. Run notebook 3 to generate them locally.

### Making a Prediction

```bash
python 4_predict_using_model.py
```

```
Total Area in Marla : 10
Total Bedrooms : 4
Total Baths : 3
Property Type (House, Flat, Shop, etc.) : House
City : Lahore
Predicted Price: PKR 33,165,997.69
```

## Results

The Random Forest model achieved **R² = 0.79** on the test set, meaning it explains 79% of the variance in property prices. The remaining 21% is driven by factors not captured in the dataset such as interior condition, exact street location, and seller negotiation.

## Tools & Libraries

- Python, Pandas, NumPy
- Scikit-learn (OHE, RandomizedSearchCV, RandomForestRegressor)
- Matplotlib, Seaborn
- Joblib
