import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv("googleplaystore.csv")

# Display the first few rows
print(df.head())
print(df.isnull().sum())
df.dropna(subset=['Type', 'Content Rating', 'Current Ver'], inplace=True)
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())

df['Android Ver'] = df['Android Ver'].fillna(df['Android Ver'].mode()[0])

print(df.isnull().sum())
df['Installs'] = df['Installs'].str.replace(',', '').str.replace('+', '').astype(int)
df['Price'] = df['Price'].str.replace('$', '').astype(float)
df['Reviews'] = df['Reviews'].astype(int)
print(df.dtypes)
print(df[['App', 'Category', 'Price']].sort_values(by='Price', ascending=False).head(10))




# Function to convert size values
def convert_size(size):
    if size == "Varies with device":
        return None  # Keep as NaN
    elif "M" in size:
        return float(size.replace("M", ""))
    elif "k" in size:
        return float(size.replace("k", "")) / 1024  # Convert KB to MB
    return None

df['Size'] = df['Size'].apply(convert_size)
df['Size'] = df['Size'].fillna(df['Size'].mean())
  # Fill NaN with average size
print(df.dtypes)
print(df.describe())
#Analyze Installs (Most installed categories)
top_installs = df['Installs'].value_counts().head(10)
print(top_installs)
#Find Top Categories by Ratings
top_categories = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)
print(top_categories)

df.groupby('Type')[['Installs', 'Rating', 'Reviews']].mean()
sns.scatterplot(x=df['Size'], y=df['Rating'])
plt.title("App Size vs. Ratings")
plt.show()



plt.figure(figsize=(8,6))
sns.heatmap(df[['Installs', 'Price', 'Size', 'Rating']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#Check Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Price'], bins=20, kde=True, color='green')
plt.title("Distribution of App Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()



plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'], bins=20, kde=True, color='blue')
plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()




sns.scatterplot(x=df['Reviews'], y=df['Rating'])
plt.xscale('log')  # Scale reviews for better visualization
plt.title('Reviews vs. Ratings')
plt.xlabel('Number of Reviews (log scale)')
plt.ylabel('App Rating')
plt.show()


df.groupby('Category')['Installs'].sum().sort_values(ascending=False).head(10).plot(kind='bar', figsize=(10, 5))
plt.title('Top 10 Categories by Total Installs')
plt.ylabel('Total Installs')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='Rating', data=df)
plt.xticks(rotation=90)
plt.title("Rating Distribution by Category")
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(x=top_categories.index, y=top_categories.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 10 Categories by Average Rating")
plt.show()

df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df['Year'] = df['Last Updated'].dt.year
sns.lineplot(x=df['Year'], y=df['Rating'])
plt.title("Yearly Trend of Ratings")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Selecting features
X = df[['Reviews', 'Size', 'Installs', 'Price']]
y = df['Rating']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Checking accuracy
from sklearn.metrics import mean_absolute_error
print('MAE:', mean_absolute_error(y_test, y_pred))
