import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('sales-dataset.csv')

# Rename columns to English
df.rename(columns={
    'data': 'date',
    'venda': 'sales',
    'estoque': 'stock',
    'preco': 'price'
}, inplace=True)

# Convert 'date' column to datetime type for easier time-based analysis
df['date'] = pd.to_datetime(df['date'])

#-----------------------------------------------------data cleaning-------------------------------------------------

# Check for missing values in each column
print("Missing values per column:")
print(df.isnull().sum())

# Check for duplicate rows
print(f"Number of duplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates()

# Show basic statistics to understand data spread and spot anomalies
print(df.describe())

# Filter out rows with invalid or negative values if any
df = df[(df['sales'] >= 0) & (df['price'] > 0) & (df['stock'] >= 0)]

# Final data info and preview after cleaning
print(df.info())
print(df.head())


#-----------------------------------------------------data visualization-------------------------------------------------
# Make plots look nicer
sns.set(style='whitegrid')

# 1. Plot sales over time
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['sales'], label='Sales')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# 2. Add date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6

# 3. Plot average sales by month
plt.figure(figsize=(10,5))
monthly_sales = df.groupby('month')['sales'].mean()
sns.barplot(x=monthly_sales.index, y=monthly_sales.values, palette='viridis')
plt.title('Average Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.show()

# 4. Plot average sales by day of week
plt.figure(figsize=(10,5))
dow_sales = df.groupby('day_of_week')['sales'].mean()
sns.barplot(x=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], y=dow_sales.values, palette='magma')
plt.title('Average Sales by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.show()