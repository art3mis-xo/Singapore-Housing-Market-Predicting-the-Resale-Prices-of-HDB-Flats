import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_histogram(series, bins=50, figsize=(8, 5), edgecolor='black', title=None, xlabel=None, save_path=None):
    desc = series.describe()
    plt.figure(figsize=figsize)
    plt.hist(series, bins=bins, edgecolor=edgecolor)
    #plt.title(title if title else f"Distribution of {series.name}", fontsize=20)
    plt.xlabel(xlabel if xlabel else series.name, fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.axvline(desc["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean: {desc['mean']:.0f}")
    plt.axvline(desc["50%"], color="blue", linestyle="--", linewidth=2, label=f"Median: {desc['50%']:.0f}")
    plt.axvline(desc["25%"], color="green", linestyle=":", linewidth=2, label=f"Q1: {desc['25%']:.0f}")
    plt.axvline(desc["75%"], color="green", linestyle=":", linewidth=2, label=f"Q3: {desc['75%']:.0f}")
    plt.text(0.98, 0.95, f"Std: {desc['std']:.0f}", transform=plt.gca().transAxes,
             ha="right", va="top", fontsize=16, bbox=dict(facecolor="white", alpha=0.6))
    plt.legend(fontsize=16)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_scatterplot(x, y, save_path=None):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=x, y=y, alpha=0.5)

    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    coeffs = np.polyfit(x, y, deg=1)
    poly_eqn = np.poly1d(coeffs)
    plt.plot(x, poly_eqn(x), color="red", linewidth=2, label="Linear Fit")

    plt.xlabel(x.name, fontsize=20)
    plt.ylabel(y.name, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.title(f"Scatterplot of {x.name} vs {y.name}", fontsize=20)
    plt.legend(fontsize=16)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def corr_heatmap(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    real_numeric = [
        col for col in num_cols
        if not (df[col].dropna().isin([0,1]).all()
                or str(col)[-1].isdigit())
    ]
    corr = df[real_numeric].corr()
    plt.figure(figsize=(25,20))  # wider and taller figure
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                annot_kws={"size":12})  # slightly larger font
    plt.title("Correlation Heatmap of Numeric Features", fontsize=20)
    plt.show()

def high_corr_pairs(df, threshold=0.7):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    real_numeric = [
        col for col in num_cols
        if not (df[col].dropna().isin([0,1]).all()
                or str(col)[-1].isdigit())
    ]
    corr = df[real_numeric].corr()

    # Get pairs
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))  # upper triangle
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["feature1", "feature2", "correlation"]

    # Filter by threshold
    return corr_pairs[abs(corr_pairs["correlation"]) > threshold]

def plot_box_plots(df, cols_to_plot=None, cols_to_omit=()):
  if cols_to_plot is None:
    cols_to_plot = df.columns.tolist()
  cols_to_plot = set(cols_to_plot) - set(cols_to_omit)

  # Remove columns of dtype object from plotting
  object_dtype_colums = df.select_dtypes(include='object').columns
  cols_to_plot = cols_to_plot - set(object_dtype_colums.tolist())

  num_cols_in_plot = 5
  num_rows_in_plot = math.ceil(len(cols_to_plot) / num_cols_in_plot)

  fig = make_subplots(rows=num_rows_in_plot, cols=num_cols_in_plot)
  fig.update_layout(
    autosize=False,
    width=1200,
    height=num_rows_in_plot * 400,
  )


  for i, col in enumerate(cols_to_plot):
    row_idx = int(i / num_cols_in_plot) + 1
    col_idx = int(i % num_cols_in_plot) + 1
    fig.append_trace(go.Box(y=df[col], name=col),
                      row=row_idx, col=col_idx)

  fig.show()
  

# def plot_monthly_rental_resale_comparison(rental_df, train_df, agg_per_mth='Mean'):
#     rental_df_filtered = rental_df.copy()[['rent_approval_date', 'monthly_rent']]
#     rental_df_filtered['monthly_rent'] = (rental_df_filtered['monthly_rent'] - rental_df_filtered['monthly_rent'].mean()) / rental_df_filtered['monthly_rent'].std()

#     train_df_filtered = train_df.copy()
#     train_df_filtered = train_df_filtered[['MONTH', 'RESALE_PRICE']]
#     train_df_filtered['RESALE_PRICE'] = (train_df_filtered['RESALE_PRICE'] - train_df_filtered['RESALE_PRICE'].mean()) / train_df_filtered['RESALE_PRICE'].std()

#     if agg_per_mth == 'Mean':
#         rental_df_filtered = rental_df_filtered.groupby('rent_approval_date').mean().reset_index()
#         train_df_filtered = train_df_filtered.groupby("MONTH").mean().reset_index()
#     elif agg_per_mth == 'Median':
#         rental_df_filtered = rental_df_filtered.groupby('rent_approval_date').median().reset_index()
#         train_df_filtered = train_df_filtered.groupby("MONTH").median().reset_index()

#     fig = go.Figure(
#         layout=go.Layout(
#             title=go.layout.Title(text="Normalized Mean Monthly Rental vs Resale Price")
#         )
#     )
#     fig.add_trace(go.Scatter(x=rental_df_filtered["rent_approval_date"], y=rental_df_filtered["monthly_rent"], mode='lines', name='MONTHLY_RENT'))
#     fig.add_trace(go.Scatter(x=train_df_filtered["MONTH"], y=train_df_filtered["RESALE_PRICE"], mode='lines', name='RESALE_PRICE'))
#     fig.show()
    
def plot_mean_resale_price_per_sqm_rpi_adj_vs_remaining_lease(train_df, bin_size=1, *args, **kwargs):
    """
    Plots mean resale price per sqm against remaining lease (binned by `bin_size` years).
    """
    # Copy and clean
    df = train_df.copy()[['RESALE_PRICE_PER_SQM_RPI_ADJ', 'REMAINING_LEASE']].dropna()

    # Convert REMAINING_LEASE if it's a string like "82 years 05 months"
    if df['REMAINING_LEASE'].dtype == 'object':
        df['REMAINING_LEASE'] = (
            df['REMAINING_LEASE']
            .str.extract(r'(\d+)')  # Extract numeric years
            .astype(float)
        )

    # Bin lease years
    df['LEASE_BIN'] = (df['REMAINING_LEASE'] // bin_size) * bin_size

    # Compute mean resale price per sqm per bin
    lease_mean = df.groupby('LEASE_BIN', as_index=False)['RESALE_PRICE_PER_SQM_RPI_ADJ'].mean()

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=lease_mean['LEASE_BIN'],
        y=lease_mean['RESALE_PRICE_PER_SQM_RPI_ADJ'],
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=6),
        name=f'Mean Price per SQM (binned by {bin_size} year{"s" if bin_size>1 else ""})'
    ))

    fig.update_layout(
        *args, kwargs
    )

    
    fig.show()
    
    
def plot_mean_resale_price_vs_remaining_lease(train_df, bin_size=1, *args, **kwargs):
    """
    Plots mean resale price against remaining lease (binned by `bin_size` years).
    """
    # Copy and clean
    df = train_df.copy()[['RESALE_PRICE', 'REMAINING_LEASE']].dropna()

    # Convert REMAINING_LEASE if it's in string format like "82 years 05 months"
    if df['REMAINING_LEASE'].dtype == 'object':
        df['REMAINING_LEASE'] = (
            df['REMAINING_LEASE']
            .str.extract(r'(\d+)')  # extract first number (years)
            .astype(float)
        )

    # Bin the lease years (e.g., every 1 year)
    df['LEASE_BIN'] = (df['REMAINING_LEASE'] // bin_size) * bin_size

    # Compute mean resale price per bin
    lease_mean = df.groupby('LEASE_BIN', as_index=False)['RESALE_PRICE'].mean()

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=lease_mean['LEASE_BIN'],
        y=lease_mean['RESALE_PRICE'],
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=6),
        name=f'Mean Resale Price (binned by {bin_size} year{"s" if bin_size>1 else ""})'
    ))

    fig.update_layout(
        *args,
        **kwargs
    )

    fig.show()


def plot_monthly_rental_resale_comparison(rental_df, train_df, agg_per_mth='Mean'):
    # Copy needed columns
    rental_df_filtered = rental_df.copy()[['rent_approval_date', 'monthly_rent']]
    train_df_filtered = train_df.copy()[['MONTH', 'RESALE_PRICE', 'RESALE_QUARTER_RPI']]

    # Ensure datetime format
    rental_df_filtered['rent_approval_date'] = pd.to_datetime(rental_df_filtered['rent_approval_date'])
    train_df_filtered['MONTH'] = pd.to_datetime(train_df_filtered['MONTH'])

    # Filter for 2021 onwards
    rental_df_filtered = rental_df_filtered[rental_df_filtered['rent_approval_date'].dt.year >= 2021]
    train_df_filtered = train_df_filtered[train_df_filtered['MONTH'].dt.year >= 2021]

    # Normalize
    rental_df_filtered['monthly_rent'] = (
        rental_df_filtered['monthly_rent'] - rental_df_filtered['monthly_rent'].mean()
    ) / rental_df_filtered['monthly_rent'].std()
    train_df_filtered['RESALE_PRICE'] = (
        train_df_filtered['RESALE_PRICE'] - train_df_filtered['RESALE_PRICE'].mean()
    ) / train_df_filtered['RESALE_PRICE'].std()
    train_df_filtered['RESALE_QUARTER_RPI'] = (
        train_df_filtered['RESALE_QUARTER_RPI'] - train_df_filtered['RESALE_QUARTER_RPI'].mean()
    ) / train_df_filtered['RESALE_QUARTER_RPI'].std()

    # Aggregate
    if agg_per_mth == 'Mean':
        rental_df_filtered = rental_df_filtered.groupby('rent_approval_date').mean().reset_index()
        train_df_filtered = train_df_filtered.groupby('MONTH').mean().reset_index()
    elif agg_per_mth == 'Median':
        rental_df_filtered = rental_df_filtered.groupby('rent_approval_date').median().reset_index()
        train_df_filtered = train_df_filtered.groupby('MONTH').median().reset_index()

    # Plot
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text="Normalized Mean Monthly Rental vs Resale Price vs RPI (2021 Onwards)")
        )
    )

    fig.add_trace(go.Scatter(
        x=rental_df_filtered["rent_approval_date"],
        y=rental_df_filtered["monthly_rent"],
        mode='lines',
        name='MONTHLY_RENT'
    ))

    fig.add_trace(go.Scatter(
        x=train_df_filtered["MONTH"],
        y=train_df_filtered["RESALE_PRICE"],
        mode='lines',
        name='RESALE_PRICE'
    ))

    fig.add_trace(go.Scatter(
        x=train_df_filtered["MONTH"],
        y=train_df_filtered["RESALE_QUARTER_RPI"],
        mode='lines',
        name='RESALE_QUARTER_RPI',
        line=dict(dash='dot')  # visually distinct
    ))

    fig.show()