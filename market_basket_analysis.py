import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

def load_and_preprocess_data(dataset_path):
    data = pd.read_csv(dataset_path, encoding='latin')
    pd.set_option('display.max_columns', None)

    # General information about the dataset
    print(data.info())
    print(data.shape)
    # print(data.head())

    # Check for missing values
    # miss_val = data.isnull().sum
    # print(miss_val)

    # Remove rows with null values in 'CustomerID' and 'Description'
    data = data.dropna(subset=['CustomerID', 'Description'])

    # Filter rows where 'Quantity' is non-negative and 'UnitPrice' is >0
    data = data.loc[(data.Quantity >= 0) & (data.UnitPrice > 0)]

    # Dataset after cleaning
    print(data.info())
    print(data.shape)
    print(data.describe())

    return data


def mba_apriori(data):
    items = list(data.Description.unique())
    items_per_invoice = data.groupby('InvoiceNo')
    transaction = items_per_invoice.aggregate(lambda x: tuple(x)).reset_index()[['InvoiceNo', 'Description']]

    temp = dict()
    for record in transaction.to_dict('records'):
        invoice_num = record['InvoiceNo']
        items_list = record['Description']

        transaction_dict = dict.fromkeys(items, 0)
        transaction_dict.update({item: 1 for item in items if item in items_list})
        temp.update({invoice_num: transaction_dict})

    encoded_items = [value for key, value in temp.items()]
    transactions_data = pd.DataFrame(encoded_items)
    # transactions_data.to_csv("transactions.csv")

    # Determine frequent items - min_support=0.01 means item appears in at least 1% of transactions
    frequent_items = apriori(transactions_data, min_support=0.01, use_colnames=True)
    print(frequent_items)

    # Find association rules using frequent_items
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)

    # Strong rules only
    rules = rules[(rules['confidence'] > 0.7) & (rules['lift'] > 3)]
    # print(rules)

    rules = pd.DataFrame(rules, columns=['id', 'antecedents', 'consequents', 'antecedent support', 'consequent support',
                                         'support', 'confidence', 'lift', 'leverage', 'conviction'])
    rules = rules[['id', 'antecedents', 'consequents', 'antecedent support', 'consequent support',
                   'support', 'confidence', 'lift']]
    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
    rules.to_csv("association_rules.csv")


def mba_fpgrowth(data):
    items = list(data.Description.unique())
    items_per_invoice = data.groupby('InvoiceNo')
    transaction = items_per_invoice.aggregate(lambda x: tuple(x)).reset_index()[['InvoiceNo', 'Description']]

    temp = dict()
    for record in transaction.to_dict('records'):
        invoice_num = record['InvoiceNo']
        items_list = record['Description']

        transaction_dict = dict.fromkeys(items, 0)
        transaction_dict.update({item: 1 for item in items if item in items_list})
        temp.update({invoice_num: transaction_dict})

    encoded_items = [value for key, value in temp.items()]
    transactions_data = pd.DataFrame(encoded_items)
    # transactions_data.to_csv("transactions.csv")

    # Determine frequent items - min_support=0.01 means item appears in at least 1% of transactions
    frequent_items = fpgrowth(transactions_data, min_support=0.01, use_colnames=True)
    print(frequent_items)

    # Find association rules using frequent_items
    rules = association_rules(frequent_items, metric="confidence", min_threshold=0.05)

    # Strong rules only
    rules = rules[(rules['confidence'] > 0.7) & (rules['lift'] > 3)]

    rules = pd.DataFrame(rules, columns=['id', 'antecedents', 'consequents', 'antecedent support', 'consequent support',
                                         'support', 'confidence', 'lift', 'leverage', 'conviction'])
    rules = rules[['id', 'antecedents', 'consequents', 'antecedent support', 'consequent support',
                   'support', 'confidence', 'lift']]
    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
    rules.to_csv("association_rules2.csv")


def market_basket_analysis(dataset_path):
    # Load&Preprocess data
    data = load_and_preprocess_data(dataset_path)

    # mba_apriori(data)
    mba_fpgrowth(data)



if __name__ == '__main__':
    market_basket_analysis("online_retail.csv")