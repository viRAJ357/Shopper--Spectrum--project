import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    # Transactions by Country
    top_countries = df['Country'].value_counts().head(10)
    top_countries.plot(kind='bar', title='Top Countries by Transactions')
    plt.show()

    # Top Products
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    top_products.plot(kind='bar', title='Top Selling Products')
    plt.show()

    # Purchase Trend
    df.set_index('InvoiceDate').resample('M')['InvoiceNo'].count().plot()
    plt.title("Monthly Purchase Trend")
    plt.show()

    # Monetary Distribution
    sns.histplot(df['TotalAmount'], bins=50)
    plt.title("Transaction Value Distribution")
    plt.show()
