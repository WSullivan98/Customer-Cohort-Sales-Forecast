import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
plt.style.use('ggplot')



# Cohort Analysis  

def transactions_by_cohort(df):
    Cohort_yr = pd.DataFrame(df.groupby('Customer ID')['Year'].min()).reset_index()
    Cohort_yr.columns = ['Customer ID', 'Cohort Yr']
    cohort_transactions= df.merge(Cohort_yr, on='Customer ID').reset_index(drop=True)
    cohort_transactions = cohort_transactions[['Customer ID', 'Cohort Yr', 'Customer', 'Type', 'Date', 'Year',
                            'Month', 'Document Number', 'Description', 'Qty', 'Unit Price',
                            'Sales Total', 'Revenue', 'State']]
    return cohort_transactions

def cohort_size(df):
    cohort_by_customers = pd.DataFrame(df.groupby('Cohort Yr')['Customer ID'].nunique()).reset_index()
    cohort_by_customers.columns = ['Cohort Yr', 'Customer count']
    return cohort_by_customers

def cohort_sales_by_year(df):
    sales_yr_cohort = pd.DataFrame(df.groupby(['Year','Cohort Yr'])['Sales Total'].sum()).reset_index()
    return sales_yr_cohort





# Visuals
## Customer Cohort Chart
def customer_cohort_data(df):
    sales_yr_cohort = cohort_sales_by_year(df)
    sales_yr_cohort = sales_yr_cohort.reset_index()
    sales_yr_cohort = sales_yr_cohort.rename({'Cohort Yr':'Cohort Yr Sales'},axis=1)
    sales_cohort_data = pd.pivot_table(sales_yr_cohort, index='Cohort Yr Sales', values='Sales Total', columns='Year', fill_value=0)
    return sales_cohort_data 


def customer_cohort_chart(df):
    plt.style.use('ggplot')
    annual_sales = df.groupby('Year')['Sales Total'].sum()
    sales_cohort_data = customer_cohort_data(df)
    years = sales_cohort_data.columns.to_list()
    y={}
    for yr in sales_cohort_data.index.to_list():
        y[yr] =sales_cohort_data.loc[yr].to_list()
    
    fig, ax = plt.subplots(figsize=(12,10))
    x = years
    ax.plot(annual_sales, color='blue',linewidth=2)
    ax.stackplot(years, y.values(), labels=y.keys())
    ax.legend(loc='upper left', fontsize=16)
    ax.set_title('Sales by Customer Cohort Year', fontsize=16)
    ax.set_xlabel('Years', fontsize=16)
    ax.set_ylabel('Sales ($000,000)',fontsize=16)
    plt.yticks(np.arange(0, 15000000, 2000000), labels=['0','2m', '4m', '6m', '8m', '10m', '12m', '14m'], fontsize=16)
    plt.xticks(fontsize=16)
    plt.show()
    return plt

## Retention plot

# breakout Cohort Customer count per year

def retention_matrix_data(df):
    df_cohort = df.groupby(['Cohort Yr', 'Year']).agg(n_customers=('Customer ID', 'nunique')).reset_index(drop=False)
    df_cohort['period_number'] = (df_cohort['Year'] - df_cohort['Cohort Yr'])

    cohort_pivot = df_cohort.pivot_table(index = 'Cohort Yr',
                                        columns = 'period_number',
                                        values = 'n_customers')
    print('Cohort Customer count per year \n',cohort_pivot)
    cohort_size = cohort_pivot.iloc[:,0]
    cohort_size.columns = ['Cohort Yr','unique customers per cohort']
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    print('Customer Retention \n',retention_matrix)
    return retention_matrix

def retention_curve(df):
    retention_matrix = retention_matrix_data(df)
    curve = retention_matrix.mean(axis=0).reset_index()
    curve.columns = ['periods', 'retention']
    curve['churn'] = 1-curve['retention']
    return curve

def retention_matrix_plot(df):
    with sns.axes_style("white"):
        fig, ax = plt.subplots(1,2, figsize=(12,8), sharey=True, gridspec_kw={'width_ratios':[1,11]})

        #retention matrix
        retention_matrix = retention_matrix_data(df)
        sns.heatmap(retention_matrix,
                    mask=retention_matrix.isnull(),
                    annot=True,
                    fmt='.0%',
                    cmap='RdYlGn',
                    ax=ax[1])
        ax[1].set_title('Yearly Cohorts: User Retention', fontsize=16)
        ax[1].set(xlabel='# of periods', ylabel='')

        #cohort size
        cohort_by_customers = cohort_size(df)
        white_cmap = mcolors.ListedColormap(['White'])
        sns.heatmap(cohort_by_customers,
                    annot=True,
                    cbar=False,
                    fmt='g',
                    cmap=white_cmap,
                    ax=ax[0])
        fig.tight_layout()
        plt.show()
    return fig


# Need retentions curve here or in transaction 


if __name__ == "__main__":  # used for testing purposes but will transition to OOP
    df = pd.read_csv('../data/processed/sales.csv')
    print(df.head())

    cohort_by_customers = cohort_size(df)
    print('Cohort by customers \n',cohort_by_customers,'\n')

    sales_yr_cohort = cohort_sales_by_year(df)
    print(sales_yr_cohort)

    sales_cohort_data = customer_cohort_data(df)
    print(sales_cohort_data)

    print('\n', sales_cohort_data.index)
    print('\n', sales_cohort_data.columns.to_list())
    print('\n', sales_cohort_data.loc[2017])

    c3 = customer_cohort_chart(df)
    c3.savefig('../data/processed/customer_cohort_chart.png')

    retention_matrix = retention_matrix_data(df)
    retention_plot = retention_matrix_plot(df)
    retention_plot.savefig('../images/cohort_retention.png')

    curve = retention_curve(df)
    print('Retention Curve data \n',curve)
