import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.float_format', lambda x: '%.4f' % x)


def outlier_thresholds(data, column):
    quartile1 = data[column].quantile(0.01)  # Q1, 25th percentile
    quartile3 = data[column].quantile(0.99)  # Q3, 75th percentile
    interquartile_range = quartile3 - quartile1 # IQR
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit


def replace_with_thresholds(data, column):
    low_limit, up_limit = outlier_thresholds(data, column)
    data.loc[(data[column] > up_limit), column] = up_limit


def check_data(data):
    print("General information about the dataset")
    print("-------------------------------------")
    print(data.info())

    print("\nDimensions")
    print("----------")
    print(data.shape)

    print("\nTypes")
    print("-----------------------")
    print(data.dtypes)

    print("\nUnique elements")
    print("---------------------")
    print(data.nunique())

    print("\nMissing values")
    print("--------------------")
    print(data.isnull().sum())

    print("\nDuplicates")
    print("----------")
    print(data.duplicated().sum())

    print("\nDescription")
    print("-----------")
    print(data.describe().T)

    print("\nExample")
    print("-------")
    print(data.head(10))


def preprocess_data(data):
    # In the dataset 'amount' column's type is object, but it should be numerical
    data['amount'] = data['amount'].str.replace(',', '.')
    data['amount'] = data['amount'].astype(float)
    data['quantity'] = data['quantity'].astype(float)

    # print(data.dtypes)
    # print(data.describe().T)

    # Minimum values of 'quantity' and 'amount' columns are negative indicating cancellation of the product
    # Remove all entries where 'invoiceID' starts with 'C' [C->Cancel]
    # Filter only 'quantity' and 'amount' > 0

    cleaned_data = data[(data['quantity'] > 0) & (data['amount'] > 0) & (~data['invoiceID'].str.startswith("C"))].copy()
    cleaned_data.dropna(inplace=True)

    print(cleaned_data.describe().T)
    # Check outliers
    sns.boxplot(
        data=cleaned_data[["quantity", "amount"]],
        orient="h",
        flierprops={
            "marker": "x",
            "markersize": 7
        },
    )
    plt.show()

    # Removing outliers
    # Tukey’s Fences method
    replace_with_thresholds(cleaned_data, 'quantity')
    replace_with_thresholds(cleaned_data, 'amount')

    sns.boxplot(
        data=cleaned_data[["quantity", "amount"]],
        orient="h",
        flierprops={
            "marker": "x",
            "markersize": 7
        },
    )
    plt.show()

    return cleaned_data


def clv_calculation(data):
    print(data.head())
    data['total_income'] = data['quantity'] * data['amount']  # amount represent unit price

    clv_data = data.groupby('customerID').agg({
        'invoiceID': lambda x: x.nunique(),
        'quantity': lambda x: x.sum(),
        'total_income': lambda x: x.sum()
    })

    clv_data.columns = ['total_transactions_num', 'total_quantity', 'total_income']
    print(clv_data.head())

    clv_data['average_order_value'] = clv_data['total_income'] / clv_data['total_transactions_num']
    clv_data['purchase_frequency'] = clv_data['total_transactions_num'] / clv_data.shape[0]

    repeat_rate = clv_data[clv_data["total_transactions_num"] > 1].shape[0] / clv_data.shape[0]
    churn_rate = 1 - repeat_rate

    clv_data["profit_margin"] = clv_data['total_income'] / 3

    clv_data['customer_value'] = clv_data['average_order_value'] * clv_data['purchase_frequency']
    clv_data["customer_lifetime_value"] = (clv_data["customer_value"] / churn_rate) * clv_data["profit_margin"]

    clv_data["class"] = pd.qcut(clv_data["customer_lifetime_value"], 4, labels=["D", "C", "B", "A"])

    clv_data_sorted = clv_data.sort_values(by="customer_lifetime_value", ascending=False)
    print("\n CLV results")
    print("-----------------------------------------------------------------------------------------------------------")
    print(clv_data_sorted.head())

    results = clv_data.groupby("class", observed=False).agg({"count", "sum", "mean"})
    print("\n Statistics")
    print("-----------------------------------------------------------------------------------------------------------")
    print(results)

    # BGD-ND

    data['invoice_date'] = pd.to_datetime(data['invoice_date'])
    most_recent_date = data['invoice_date'].max()
    today_date = most_recent_date + pd.Timedelta(days=2)

    data = data.groupby('customerID').filter(lambda x: x['invoiceID'].nunique() > 1)

    cltv_data2 = data.groupby('customerID').agg(
        R=('invoice_date', lambda x: (x.max() - x.min()).days),
        T=('invoice_date', lambda x: (today_date - x.min()).days),
        F=('invoiceID', 'nunique'),
        M=('total_income', 'sum')
    )

    cltv_data2["R"] = cltv_data2["R"] / 7
    cltv_data2["M"] = cltv_data2["M"] / cltv_data2["F"]
    cltv_data2["T"] = cltv_data2["T"] / 7


    BGF = BetaGeoFitter(penalizer_coef=0.001)
    BGF.fit(cltv_data2['F'], cltv_data2['R'], cltv_data2['T'])

    # Predict expected purchases in 1 month (30 days) and 3 months (90 days)
    cltv_data2["expected_purchase_1_month"] = BGF.predict(4, cltv_data2['F'], cltv_data2['R'], cltv_data2['T'])
    cltv_data2["expected_purchase_3_months"] = BGF.predict(4*3, cltv_data2['F'], cltv_data2['R'], cltv_data2['T'])

    print(cltv_data2.head())
    # plot_period_transactions(BGF)
    # plt.show()

    GGF = GammaGammaFitter(penalizer_coef=0.01)
    GGF.fit(cltv_data2['F'], cltv_data2['M'])

    # GGF predicts the average profit per transaction for each customer, based on observed frequency and monetary values
    cltv_data2["expected_average_profit"] = GGF.conditional_expected_average_profit(cltv_data2['F'], cltv_data2['M'])
    # print(cltv_data2["expected_average_profit"].head())
    # print(cltv_data2.sort_values("expected_average_profit", ascending=False).head(10))

    # 3. Calculation of CLTV with BG-NBD and Gamma-Gamma Model
    cltv_predict = GGF.customer_lifetime_value(BGF,
                                               cltv_data2['F'],
                                               cltv_data2['R'],
                                               cltv_data2['T'],
                                               cltv_data2['M'],
                                               time=3,  # 3 months
                                               freq="W",  # Frequency T (Weekly)
                                               discount_rate=0.01)
    cltv_predict.reset_index()

    cltv_final = cltv_data2.merge(cltv_predict, on="customerID", how="left")
    print(cltv_final.head())
    print(cltv_final.sort_values(by="clv", ascending=False).head(10))



def clv_segmentation():
    data = pd.read_csv("tobacco_sales.csv")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 576)
    check_data(data)
    cleaned_data = preprocess_data(data)
    clv_calculation(cleaned_data)


if __name__ == '__main__':
    clv_segmentation()
