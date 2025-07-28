import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking charts
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_data_for_story():
    print("Loading data for data story visualizations...")
    
    agent_df = pd.read_csv('CEASalespersonsPropertyTransactionRecordsresidential.csv')
    agent_df['transaction_date'] = pd.to_datetime(agent_df['transaction_date'], format='%b-%Y')
    agent_df = agent_df[(agent_df['property_type'] == 'HDB') & (agent_df['transaction_type'] == 'RESALE')]
    
    price_df = pd.read_csv('Resale flat prices based on registration date from Jan-2017 onwards.csv')
    price_df['month'] = pd.to_datetime(price_df['month'], format='%Y-%m')
    
    return agent_df, price_df

def create_chart_1_agent_impact(agent_df):
    print("Creating Chart 1: Agent Business Impact...")
    
    monthly_counts = agent_df.groupby(agent_df['transaction_date'].dt.to_period('M')).size().reset_index(name='count')
    monthly_counts['transaction_date'] = monthly_counts['transaction_date'].dt.to_timestamp()
    
    agent_monthly = agent_df.groupby(agent_df['transaction_date'].dt.to_period('M'))['salesperson_name'].nunique().reset_index()
    agent_monthly['transaction_date'] = agent_monthly['transaction_date'].dt.to_timestamp()
    agent_monthly = agent_monthly.rename(columns={'salesperson_name': 'unique_agents'})
    
    monthly_data = monthly_counts.merge(agent_monthly, on='transaction_date')
    monthly_data['transactions_per_agent'] = monthly_data['count'] / monthly_data['unique_agents']
    
    portal_launch = pd.Timestamp('2018-01-01')
    monthly_data['portal_period'] = monthly_data['transaction_date'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch else 'Pre-Portal'
    )
    
    pre_portal = monthly_data[monthly_data['portal_period'] == 'Pre-Portal']
    post_portal = monthly_data[monthly_data['portal_period'] == 'Post-Portal']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.plot(monthly_data['transaction_date'], monthly_data['count'], 
             color='#2E86AB', linewidth=2, alpha=0.7)
    ax1.axvline(x=portal_launch, color='#A23B72', linestyle='--', linewidth=2, label='Portal Launch')
    ax1.fill_between(monthly_data['transaction_date'], monthly_data['count'], 
                     alpha=0.3, color='#2E86AB')
    ax1.set_title('Monthly Agent Transactions', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Transactions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(monthly_data['transaction_date'], monthly_data['unique_agents'], 
             color='#F18F01', linewidth=2, alpha=0.7)
    ax2.axvline(x=portal_launch, color='#A23B72', linestyle='--', linewidth=2, label='Portal Launch')
    ax2.fill_between(monthly_data['transaction_date'], monthly_data['unique_agents'], 
                     alpha=0.3, color='#F18F01')
    ax2.set_title('Monthly Active Agents', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Agents')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(monthly_data['transaction_date'], monthly_data['transactions_per_agent'], 
             color='#C73E1D', linewidth=2, alpha=0.7)
    ax3.axvline(x=portal_launch, color='#A23B72', linestyle='--', linewidth=2, label='Portal Launch')
    ax3.fill_between(monthly_data['transaction_date'], monthly_data['transactions_per_agent'], 
                     alpha=0.3, color='#C73E1D')
    ax3.set_title('Transactions per Agent', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Transactions per Agent')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chart_1_agent_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Chart 1 saved as 'chart_1_agent_impact.png'")

def create_chart_2_price_trends(price_df):
    print("Creating Chart 2: Price Trends...")
    
    monthly_prices = price_df.groupby('month')['resale_price'].agg(['mean', 'median', 'count']).reset_index()
    monthly_prices.columns = ['month', 'avg_price', 'median_price', 'transaction_count']
    
    monthly_volatility = price_df.groupby('month')['resale_price'].agg(['mean', 'std']).reset_index()
    monthly_volatility['cv'] = monthly_volatility['std'] / monthly_volatility['mean']
    
    portal_launch = pd.Timestamp('2018-01-01')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    ax1.plot(monthly_prices['month'], monthly_prices['avg_price'], 
             color='#2E86AB', linewidth=2, alpha=0.7, label='Average Price')
    ax1.plot(monthly_prices['month'], monthly_prices['median_price'], 
             color='#F18F01', linewidth=2, alpha=0.7, label='Median Price')
    ax1.axvline(x=portal_launch, color='#A23B72', linestyle='--', linewidth=2, label='Portal Launch')
    ax1.fill_between(monthly_prices['month'], monthly_prices['avg_price'], 
                     alpha=0.2, color='#2E86AB')
    ax1.set_title('HDB Resale Price Trends', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Price (SGD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(monthly_volatility['month'], monthly_volatility['cv'], 
             color='#C73E1D', linewidth=2, alpha=0.7)
    ax2.axvline(x=portal_launch, color='#A23B72', linestyle='--', linewidth=2, label='Portal Launch')
    ax2.fill_between(monthly_volatility['month'], monthly_volatility['cv'], 
                     alpha=0.3, color='#C73E1D')
    ax2.set_title('Price Volatility (Coefficient of Variation)', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Volatility (CV)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chart_2_price_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Chart 2 saved as 'chart_2_price_trends.png'")

def create_chart_3_regional_performance(price_df):
    print("Creating Chart 3: Regional Performance...")
    
    portal_launch = pd.Timestamp('2018-01-01')
    price_df['portal_period'] = price_df['month'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch else 'Pre-Portal'
    )
    
    top_towns = price_df['town'].value_counts().head(10).index.tolist()
    
    town_changes = []
    for town in top_towns:
        town_data = price_df[price_df['town'] == town]
        pre_data = town_data[town_data['portal_period'] == 'Pre-Portal']
        post_data = town_data[town_data['portal_period'] == 'Post-Portal']
        
        if len(pre_data) > 0 and len(post_data) > 0:
            pre_avg = pre_data['resale_price'].mean()
            post_avg = post_data['resale_price'].mean()
            change_pct = ((post_avg - pre_avg) / pre_avg) * 100
            
            town_changes.append({
                'town': town,
                'change_pct': change_pct,
                'pre_avg': pre_avg,
                'post_avg': post_avg
            })
    
    town_changes_df = pd.DataFrame(town_changes).sort_values('change_pct', ascending=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['#2E86AB' if x > 0 else '#C73E1D' for x in town_changes_df['change_pct']]
    bars = ax.barh(range(len(town_changes_df)), town_changes_df['change_pct'], 
                   color=colors, alpha=0.7)
    
    for i, (bar, change) in enumerate(zip(bars, town_changes_df['change_pct'])):
        ax.text(bar.get_width() + (0.5 if change > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                f'{change:+.1f}%', ha='left' if change > 0 else 'right', va='center', fontweight='bold')
    
    ax.set_yticks(range(len(town_changes_df)))
    ax.set_yticklabels(town_changes_df['town'])
    ax.set_xlabel('Price Change (%)')
    ax.set_title('Price Performance by Town (Top 10 by Volume)', fontweight='bold', fontsize=16)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    positive_patch = mpatches.Patch(color='#2E86AB', label='Price Increase')
    negative_patch = mpatches.Patch(color='#C73E1D', label='Price Decrease')
    ax.legend(handles=[positive_patch, negative_patch])
    
    plt.tight_layout()
    plt.savefig('chart_3_regional_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Chart 3 saved as 'chart_3_regional_performance.png'")

def create_chart_4_flat_type_comparison(price_df):
    print("Creating Chart 4: Flat Type Performance...")
    
    portal_launch = pd.Timestamp('2018-01-01')
    price_df['portal_period'] = price_df['month'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch else 'Pre-Portal'
    )
    
    flat_type_changes = []
    for flat_type in price_df['flat_type'].unique():
        pre_data = price_df[(price_df['flat_type'] == flat_type) & (price_df['portal_period'] == 'Pre-Portal')]
        post_data = price_df[(price_df['flat_type'] == flat_type) & (price_df['portal_period'] == 'Post-Portal')]
        
        if len(pre_data) > 0 and len(post_data) > 0:
            pre_avg = pre_data['resale_price'].mean()
            post_avg = post_data['resale_price'].mean()
            change_pct = ((post_avg - pre_avg) / pre_avg) * 100
            
            flat_type_changes.append({
                'flat_type': flat_type,
                'change_pct': change_pct,
                'pre_avg': pre_avg,
                'post_avg': post_avg
            })
    
    flat_type_changes_df = pd.DataFrame(flat_type_changes).sort_values('change_pct', ascending=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['#2E86AB' if x > 0 else '#C73E1D' for x in flat_type_changes_df['change_pct']]
    bars = ax.barh(range(len(flat_type_changes_df)), flat_type_changes_df['change_pct'], 
                   color=colors, alpha=0.7)
    
    for i, (bar, change) in enumerate(zip(bars, flat_type_changes_df['change_pct'])):
        ax.text(bar.get_width() + (0.5 if change > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                f'{change:+.1f}%', ha='left' if change > 0 else 'right', va='center', fontweight='bold')
    
    ax.set_yticks(range(len(flat_type_changes_df)))
    ax.set_yticklabels(flat_type_changes_df['flat_type'])
    ax.set_xlabel('Price Change (%)')
    ax.set_title('Price Performance by Flat Type', fontweight='bold', fontsize=16)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    positive_patch = mpatches.Patch(color='#2E86AB', label='Price Increase')
    negative_patch = mpatches.Patch(color='#C73E1D', label='Price Decrease')
    ax.legend(handles=[positive_patch, negative_patch])
    
    plt.tight_layout()
    plt.savefig('chart_4_flat_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Chart 4 saved as 'chart_4_flat_type_comparison.png'")

def create_chart_5_covid_comparison(agent_df, price_df):
    print("Creating Chart 5: COVID Comparison...")
    
    covid_start = pd.Timestamp('2020-04-01')
    agent_pre_covid = agent_df[agent_df['transaction_date'] < covid_start]
    price_pre_covid = price_df[price_df['month'] < covid_start]
    
    portal_launch = pd.Timestamp('2018-01-01')
    
    agent_full = agent_df.groupby(agent_df['transaction_date'].dt.to_period('M')).size().reset_index(name='count')
    agent_full['transaction_date'] = agent_full['transaction_date'].dt.to_timestamp()
    agent_full['portal_period'] = agent_full['transaction_date'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch else 'Pre-Portal'
    )
    
    agent_pre = agent_pre_covid.groupby(agent_pre_covid['transaction_date'].dt.to_period('M')).size().reset_index(name='count')
    agent_pre['transaction_date'] = agent_pre['transaction_date'].dt.to_timestamp()
    agent_pre['portal_period'] = agent_pre['transaction_date'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch else 'Pre-Portal'
    )
    
    price_full = price_df.groupby('month')['resale_price'].mean().reset_index()
    price_full['portal_period'] = price_full['month'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch else 'Pre-Portal'
    )
    
    price_pre = price_pre_covid.groupby('month')['resale_price'].mean().reset_index()
    price_pre['portal_period'] = price_pre['month'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch else 'Pre-Portal'
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(agent_full['transaction_date'], agent_full['count'], 
             color='#2E86AB', linewidth=2, alpha=0.7, label='Full Period')
    ax1.plot(agent_pre['transaction_date'], agent_pre['count'], 
             color='#F18F01', linewidth=2, alpha=0.7, label='Pre-COVID')
    ax1.axvline(x=portal_launch, color='#A23B72', linestyle='--', linewidth=2, label='Portal Launch')
    ax1.axvline(x=covid_start, color='#C73E1D', linestyle='--', linewidth=2, label='COVID Start')
    ax1.set_title('Agent Transactions: Pre-COVID vs Full Period', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Monthly Transactions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(price_full['month'], price_full['resale_price'], 
             color='#2E86AB', linewidth=2, alpha=0.7, label='Full Period')
    ax2.plot(price_pre['month'], price_pre['resale_price'], 
             color='#F18F01', linewidth=2, alpha=0.7, label='Pre-COVID')
    ax2.axvline(x=portal_launch, color='#A23B72', linestyle='--', linewidth=2, label='Portal Launch')
    ax2.axvline(x=covid_start, color='#C73E1D', linestyle='--', linewidth=2, label='COVID Start')
    ax2.set_title('Average Prices: Pre-COVID vs Full Period', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Average Price (SGD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chart_5_covid_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Chart 5 saved as 'chart_5_covid_comparison.png'")

def create_chart_6_summary_infographic():
    print("Creating Chart 6: Summary Infographic...")
    
    metrics = ['Agent Transactions', 'Active Agents', 'Agent Productivity', 'Average Price', 'Price Volatility']
    pre_portal = [100, 100, 100, 100, 100]  # Base 100
    post_portal = [126.7, 116.9, 107.6, 118.6, 97.3]  # Percentage changes
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pre_portal, width, label='Pre-Portal (Base 100)', 
                   color='#2E86AB', alpha=0.7)
    bars2 = ax.bar(x + width/2, post_portal, width, label='Post-Portal', 
                   color='#F18F01', alpha=0.7)

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Index (Pre-Portal = 100)')
    ax.set_title('HDB Resale Portal Impact Summary', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.axhline(y=100, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('chart_6_summary_infographic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Chart 6 saved as 'chart_6_summary_infographic.png'")

def main():
    print("Creating Data Story Visualizations")
    print("=" * 50)
    
    agent_df, price_df = load_data_for_story()
    
    create_chart_1_agent_impact(agent_df)
    create_chart_2_price_trends(price_df)
    create_chart_3_regional_performance(price_df)
    create_chart_4_flat_type_comparison(price_df)
    create_chart_5_covid_comparison(agent_df, price_df)
    create_chart_6_summary_infographic()
    
    print("\nAll data story visualizations completed!")
    print("Charts saved as:")
    print("- chart_1_agent_impact.png")
    print("- chart_2_price_trends.png")
    print("- chart_3_regional_performance.png")
    print("- chart_4_flat_type_comparison.png")
    print("- chart_5_covid_comparison.png")
    print("- chart_6_summary_infographic.png")

if __name__ == "__main__":
    main()