import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_agent_data():
    print("Loading agent transaction data...")
    df = pd.read_csv('CEASalespersonsPropertyTransactionRecordsresidential.csv')
    
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%b-%Y')
    
    hdb_resale_df = df[(df['property_type'] == 'HDB') & 
                       (df['transaction_type'] == 'RESALE')]
    
    covid_start = pd.Timestamp('2020-04-01')
    hdb_resale_df = hdb_resale_df[hdb_resale_df['transaction_date'] < covid_start]
    
    portal_launch_date = pd.Timestamp('2018-01-01')
    hdb_resale_df['portal_period'] = hdb_resale_df['transaction_date'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch_date else 'Pre-Portal'
    )
    
    print(f"Total HDB resale transactions (pre-COVID): {len(hdb_resale_df):,}")
    print(f"Date range: {hdb_resale_df['transaction_date'].min()} to {hdb_resale_df['transaction_date'].max()}")
    
    return hdb_resale_df

def load_and_prepare_price_data():
    print("Loading resale flat prices data...")
    df = pd.read_csv('Resale flat prices based on registration date from Jan-2017 onwards.csv')
    
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
    
    covid_start = pd.Timestamp('2020-04-01')
    df = df[df['month'] < covid_start]
    
    portal_launch_date = pd.Timestamp('2018-01-01')
    df['portal_period'] = df['month'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch_date else 'Pre-Portal'
    )
    
    print(f"Total resale transactions (pre-COVID): {len(df):,}")
    print(f"Date range: {df['month'].min()} to {df['month'].max()}")
    
    return df

def analyze_agent_impact_pre_covid(df):
    print("\n=== AGENT IMPACT ANALYSIS (PRE-COVID) ===")
    
    monthly_counts = df.groupby(df['transaction_date'].dt.to_period('M')).size().reset_index(name='count')
    monthly_counts['transaction_date'] = monthly_counts['transaction_date'].dt.to_timestamp()
    monthly_counts = monthly_counts.rename(columns={'transaction_date': 'month'})
    
    portal_launch_date = pd.Timestamp('2018-01-01')
    monthly_counts['portal_period'] = monthly_counts['month'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch_date else 'Pre-Portal'
    )
    
    agent_monthly = df.groupby(df['transaction_date'].dt.to_period('M'))['salesperson_name'].nunique().reset_index()
    agent_monthly['transaction_date'] = agent_monthly['transaction_date'].dt.to_timestamp()
    agent_monthly = agent_monthly.rename(columns={'transaction_date': 'month', 'salesperson_name': 'unique_agents'})
    agent_monthly['portal_period'] = agent_monthly['month'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch_date else 'Pre-Portal'
    )

    pre_portal_data = monthly_counts[monthly_counts['portal_period'] == 'Pre-Portal']
    post_portal_data = monthly_counts[monthly_counts['portal_period'] == 'Post-Portal']
    
    pre_portal_agents = agent_monthly[agent_monthly['portal_period'] == 'Pre-Portal']
    post_portal_agents = agent_monthly[agent_monthly['portal_period'] == 'Post-Portal']
    
    pre_avg_transactions = pre_portal_data['count'].mean()
    post_avg_transactions = post_portal_data['count'].mean()
    
    pre_avg_agents = pre_portal_agents['unique_agents'].mean()
    post_avg_agents = post_portal_agents['unique_agents'].mean()
    
    pre_avg_trans_per_agent = pre_avg_transactions / pre_avg_agents
    post_avg_trans_per_agent = post_avg_transactions / post_avg_agents
    
    transaction_change = ((post_avg_transactions - pre_avg_transactions) / pre_avg_transactions) * 100
    agent_change = ((post_avg_agents - pre_avg_agents) / pre_avg_agents) * 100
    trans_per_agent_change = ((post_avg_trans_per_agent - pre_avg_trans_per_agent) / pre_avg_trans_per_agent) * 100
    
    print(f"\nPRE-PORTAL PERIOD (before Jan 2018):")
    print(f"Average monthly transactions: {pre_avg_transactions:.0f}")
    print(f"Average unique agents per month: {pre_avg_agents:.0f}")
    print(f"Average transactions per agent per month: {pre_avg_trans_per_agent:.2f}")
    
    print(f"\nPOST-PORTAL PERIOD (Jan 2018 - Mar 2020):")
    print(f"Average monthly transactions: {post_avg_transactions:.0f}")
    print(f"Average unique agents per month: {post_avg_agents:.0f}")
    print(f"Average transactions per agent per month: {post_avg_trans_per_agent:.2f}")
    
    print(f"\nAGENT IMPACT (PRE-COVID):")
    print(f"Change in monthly transactions: {transaction_change:+.1f}%")
    print(f"Change in active agents: {agent_change:+.1f}%")
    print(f"Change in transactions per agent: {trans_per_agent_change:+.1f}%")
    
    return {
        'monthly_counts': monthly_counts,
        'agent_monthly': agent_monthly,
        'pre_avg_transactions': pre_avg_transactions,
        'post_avg_transactions': post_avg_transactions,
        'pre_avg_agents': pre_avg_agents,
        'post_avg_agents': post_avg_agents,
        'pre_avg_trans_per_agent': pre_avg_trans_per_agent,
        'post_avg_trans_per_agent': post_avg_trans_per_agent,
        'transaction_change_pct': transaction_change,
        'agent_change_pct': agent_change,
        'trans_per_agent_change_pct': trans_per_agent_change
    }

def analyze_price_impact_pre_covid(df):
    print("\n=== PRICE IMPACT ANALYSIS (PRE-COVID) ===")
    
    monthly_prices = df.groupby('month')['resale_price'].agg(['mean', 'median', 'count']).reset_index()
    monthly_prices.columns = ['month', 'avg_price', 'median_price', 'transaction_count']
    
    portal_launch_date = pd.Timestamp('2018-01-01')
    monthly_prices['portal_period'] = monthly_prices['month'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch_date else 'Pre-Portal'
    )
    
    pre_portal = df[df['portal_period'] == 'Pre-Portal']
    post_portal = df[df['portal_period'] == 'Post-Portal']
    
    pre_stats = {
        'avg_price': pre_portal['resale_price'].mean(),
        'median_price': pre_portal['resale_price'].median(),
        'std_price': pre_portal['resale_price'].std(),
        'count': len(pre_portal)
    }
    
    post_stats = {
        'avg_price': post_portal['resale_price'].mean(),
        'median_price': post_portal['resale_price'].median(),
        'std_price': post_portal['resale_price'].std(),
        'count': len(post_portal)
    }
    
    price_change_pct = ((post_stats['avg_price'] - pre_stats['avg_price']) / pre_stats['avg_price']) * 100
    median_change_pct = ((post_stats['median_price'] - pre_stats['median_price']) / pre_stats['median_price']) * 100
    
    print(f"\nPRE-PORTAL PERIOD (before Jan 2018):")
    print(f"Average price: ${pre_stats['avg_price']:,.0f}")
    print(f"Median price: ${pre_stats['median_price']:,.0f}")
    print(f"Standard deviation: ${pre_stats['std_price']:,.0f}")
    print(f"Transaction count: {pre_stats['count']:,}")
    
    print(f"\nPOST-PORTAL PERIOD (Jan 2018 - Mar 2020):")
    print(f"Average price: ${post_stats['avg_price']:,.0f}")
    print(f"Median price: ${post_stats['median_price']:,.0f}")
    print(f"Standard deviation: ${post_stats['std_price']:,.0f}")
    print(f"Transaction count: {post_stats['count']:,}")
    
    print(f"\nPRICE IMPACT (PRE-COVID):")
    print(f"Average price change: {price_change_pct:+.1f}%")
    print(f"Median price change: {median_change_pct:+.1f}%")
    
    flat_type_changes = []
    for flat_type in df['flat_type'].unique():
        pre_data = df[(df['flat_type'] == flat_type) & (df['portal_period'] == 'Pre-Portal')]
        post_data = df[(df['flat_type'] == flat_type) & (df['portal_period'] == 'Post-Portal')]
        
        if len(pre_data) > 0 and len(post_data) > 0:
            pre_avg = pre_data['resale_price'].mean()
            post_avg = post_data['resale_price'].mean()
            change_pct = ((post_avg - pre_avg) / pre_avg) * 100
            
            flat_type_changes.append({
                'flat_type': flat_type,
                'pre_avg_price': pre_avg,
                'post_avg_price': post_avg,
                'change_pct': change_pct,
                'pre_count': len(pre_data),
                'post_count': len(post_data)
            })
    
    flat_type_changes_df = pd.DataFrame(flat_type_changes)
    print(f"\nPrice changes by flat type (PRE-COVID):")
    print(flat_type_changes_df)
    
    top_towns = df['town'].value_counts().head(15).index.tolist()
    town_changes = []
    for town in top_towns:
        town_data = df[df['town'] == town]
        pre_data = town_data[town_data['portal_period'] == 'Pre-Portal']
        post_data = town_data[town_data['portal_period'] == 'Post-Portal']
        
        if len(pre_data) > 0 and len(post_data) > 0:
            pre_avg = pre_data['resale_price'].mean()
            post_avg = post_data['resale_price'].mean()
            change_pct = ((post_avg - pre_avg) / pre_avg) * 100
            
            town_changes.append({
                'town': town,
                'pre_avg_price': pre_avg,
                'post_avg_price': post_avg,
                'change_pct': change_pct,
                'pre_count': len(pre_data),
                'post_count': len(post_data)
            })
    
    town_changes_df = pd.DataFrame(town_changes).sort_values('change_pct', ascending=False)
    print(f"\nPrice changes by town (PRE-COVID, top 15):")
    print(town_changes_df)
    
    return {
        'monthly_prices': monthly_prices,
        'pre_stats': pre_stats,
        'post_stats': post_stats,
        'price_change_pct': price_change_pct,
        'median_change_pct': median_change_pct,
        'flat_type_changes_df': flat_type_changes_df,
        'town_changes_df': town_changes_df
    }

def analyze_volatility_pre_covid(df):
    print("\n=== PRICE VOLATILITY ANALYSIS (PRE-COVID) ===")
    
    monthly_volatility = df.groupby('month')['resale_price'].agg(['mean', 'std']).reset_index()
    monthly_volatility['cv'] = monthly_volatility['std'] / monthly_volatility['mean']
    
    portal_launch_date = pd.Timestamp('2018-01-01')
    monthly_volatility['portal_period'] = monthly_volatility['month'].apply(
        lambda x: 'Post-Portal' if x >= portal_launch_date else 'Pre-Portal'
    )
    
    pre_volatility = monthly_volatility[monthly_volatility['portal_period'] == 'Pre-Portal']['cv'].mean()
    post_volatility = monthly_volatility[monthly_volatility['portal_period'] == 'Post-Portal']['cv'].mean()
    
    volatility_change = ((post_volatility - pre_volatility) / pre_volatility) * 100
    
    print(f"Pre-portal average volatility (CV): {pre_volatility:.3f}")
    print(f"Post-portal average volatility (CV): {post_volatility:.3f}")
    print(f"Volatility change: {volatility_change:+.1f}%")
    
    return monthly_volatility, pre_volatility, post_volatility, volatility_change

def create_pre_covid_visualizations(agent_results, price_results, volatility_results):
    print("\n=== CREATING PRE-COVID VISUALIZATIONS ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('HDB Resale Portal Impact Analysis (Pre-COVID: Before Jan 2018 vs Jan 2018 - Mar 2020)', 
                 fontsize=16, fontweight='bold')
    
    monthly_counts = agent_results['monthly_counts']
    axes[0, 0].plot(monthly_counts['month'], monthly_counts['count'], marker='o', linewidth=1, markersize=3)
    axes[0, 0].axvline(x=pd.Timestamp('2018-01-01'), color='red', linestyle='--', alpha=0.7, label='Portal Launch')
    axes[0, 0].axvline(x=pd.Timestamp('2020-04-01'), color='orange', linestyle='--', alpha=0.7, label='COVID Start')
    axes[0, 0].set_title('Monthly Agent Transaction Volume')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Transactions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    agent_monthly = agent_results['agent_monthly']
    axes[0, 1].plot(agent_monthly['month'], agent_monthly['unique_agents'], 
                    marker='s', linewidth=1, markersize=3, color='orange')
    axes[0, 1].axvline(x=pd.Timestamp('2018-01-01'), color='red', linestyle='--', alpha=0.7, label='Portal Launch')
    axes[0, 1].axvline(x=pd.Timestamp('2020-04-01'), color='orange', linestyle='--', alpha=0.7, label='COVID Start')
    axes[0, 1].set_title('Monthly Active Agents')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Number of Unique Agents')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    monthly_prices = price_results['monthly_prices']
    axes[0, 2].plot(monthly_prices['month'], monthly_prices['avg_price'], 
                    marker='^', linewidth=1, markersize=3, color='green')
    axes[0, 2].axvline(x=pd.Timestamp('2018-01-01'), color='red', linestyle='--', alpha=0.7, label='Portal Launch')
    axes[0, 2].axvline(x=pd.Timestamp('2020-04-01'), color='orange', linestyle='--', alpha=0.7, label='COVID Start')
    axes[0, 2].set_title('Average Resale Price Trends')
    axes[0, 2].set_xlabel('Year')
    axes[0, 2].set_ylabel('Average Price (SGD)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    flat_type_changes_df = price_results['flat_type_changes_df']
    if len(flat_type_changes_df) > 0:
        flat_types = flat_type_changes_df['flat_type']
        changes = flat_type_changes_df['change_pct']
        colors = ['green' if x > 0 else 'red' for x in changes]
        
        bars = axes[1, 0].bar(range(len(flat_types)), changes, color=colors, alpha=0.7)
        axes[1, 0].set_title('Price Change by Flat Type (Pre-COVID)')
        axes[1, 0].set_xlabel('Flat Type')
        axes[1, 0].set_ylabel('Price Change (%)')
        axes[1, 0].set_xticks(range(len(flat_types)))
        axes[1, 0].set_xticklabels(flat_types, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                           f'{change:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    town_changes_df = price_results['town_changes_df']
    if len(town_changes_df) > 0:
        top_10_towns = town_changes_df.head(10)
        towns = top_10_towns['town']
        changes = top_10_towns['change_pct']
        colors = ['green' if x > 0 else 'red' for x in changes]
        
        bars = axes[1, 1].bar(range(len(towns)), changes, color=colors, alpha=0.7)
        axes[1, 1].set_title('Price Change by Town (Pre-COVID, Top 10)')
        axes[1, 1].set_xlabel('Town')
        axes[1, 1].set_ylabel('Price Change (%)')
        axes[1, 1].set_xticks(range(len(towns)))
        axes[1, 1].set_xticklabels(towns, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                           f'{change:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    monthly_volatility, _, _, _ = volatility_results
    axes[1, 2].plot(monthly_volatility['month'], monthly_volatility['cv'], 
                    marker='d', linewidth=1, markersize=3, color='purple')
    axes[1, 2].axvline(x=pd.Timestamp('2018-01-01'), color='red', linestyle='--', alpha=0.7, label='Portal Launch')
    axes[1, 2].axvline(x=pd.Timestamp('2020-04-01'), color='orange', linestyle='--', alpha=0.7, label='COVID Start')
    axes[1, 2].set_title('Price Volatility Over Time (CV)')
    axes[1, 2].set_xlabel('Year')
    axes[1, 2].set_ylabel('Coefficient of Variation')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pre_covid_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Pre-COVID impact visualizations saved as 'pre_covid_impact_analysis.png'")

def generate_pre_covid_summary(agent_results, price_results, volatility_results):
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY: HDB RESALE PORTAL IMPACt")
    print("=" * 80)
    
    print(f"\nANALYSIS PERIOD:")
    print(f"Pre-Portal: Before January 2018")
    print(f"Post-Portal: January 2018 to March 2020")
    
    print(f"\nAGENT IMPACT:")
    print(f"Transaction Volume Change: {agent_results['transaction_change_pct']:+.1f}%")
    print(f"Active Agents Change: {agent_results['agent_change_pct']:+.1f}%")
    print(f"Agent Productivity Change: {agent_results['trans_per_agent_change_pct']:+.1f}%")
    
    print(f"\nPRICE IMPACT:")
    print(f"Average Price Change: {price_results['price_change_pct']:+.1f}%")
    print(f"Median Price Change: {price_results['median_change_pct']:+.1f}%")
    print(f"Price Volatility Change: {volatility_results[3]:+.1f}%")

def main():
    print("HDB RESALE PORTAL IMPACT ANALYSIS")
    print("=" * 60)

    agent_df = load_and_prepare_agent_data()
    price_df = load_and_prepare_price_data()
    
    agent_results = analyze_agent_impact_pre_covid(agent_df)
    price_results = analyze_price_impact_pre_covid(price_df)
    volatility_results = analyze_volatility_pre_covid(price_df)
    
    create_pre_covid_visualizations(agent_results, price_results, volatility_results)
    
    generate_pre_covid_summary(agent_results, price_results, volatility_results)

if __name__ == "__main__":
    main() 