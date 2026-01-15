import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path, encoding='ISO-8859-1')

    # Remove missing CustomerID
    df.dropna(subset=['CustomerID'], inplace=True)

    # Remove cancelled invoices
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Remove negative or zero values
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Convert date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Total amount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    return df
