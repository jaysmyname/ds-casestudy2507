"""
Elasticity Scatter Plots for COE Analysis
Creates scatter plots showing the relationship between quota increases and price changes
for Category A and Category B vehicles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare COE data for analysis"""
    print("📊 Loading and preparing COE data...")
    
    # Load bidding results data
    bidding_df = pd.read_csv('COEBiddingResultsPrices.csv')
    
    # Clean and prepare bidding data
    bidding_df['month'] = pd.to_datetime(bidding_df['month'], format='%Y-%m')
    bidding_df['year'] = bidding_df['month'].dt.year
    bidding_df['month_num'] = bidding_df['month'].dt.month
    
    # Convert numeric columns to proper types
    numeric_columns = ['quota', 'bids_success', 'bids_received', 'premium']
    for col in numeric_columns:
        bidding_df[col] = pd.to_numeric(bidding_df[col], errors='coerce')
    
    # Filter for Categories A and B only
    coe_data = bidding_df[bidding_df['vehicle_class'].isin(['Category A', 'Category B'])].copy()
    
    # Create features
    coe_data['competition_ratio'] = coe_data['bids_received'] / coe_data['quota']
    
    # Remove rows with missing values
    coe_data = coe_data.dropna()
    
    print(f"✅ Data prepared: {len(coe_data)} records")
    return coe_data

def calculate_elasticity(df, category_name):
    """Calculate price elasticity of quota for the given category"""
    print(f"📈 Calculating elasticity for {category_name}...")
    
    category_data = df[df['vehicle_class'] == category_name].copy()
    
    # Calculate elasticity using log-log regression
    category_data['log_premium'] = np.log(category_data['premium'])
    category_data['log_quota'] = np.log(category_data['quota'])
    category_data['log_competition'] = np.log(category_data['competition_ratio'])
    
    # Elasticity model: log(premium) = α + β1*log(quota) + β2*log(competition) + controls
    X_elasticity = category_data[['log_quota', 'log_competition', 'year']]
    y_elasticity = category_data['log_premium']
    
    # Fit elasticity model
    elasticity_model = LinearRegression()
    elasticity_model.fit(X_elasticity, y_elasticity)
    
    # Calculate elasticity
    quota_elasticity = elasticity_model.coef_[0]
    competition_elasticity = elasticity_model.coef_[1]
    
    # Calculate average values
    avg_premium = category_data['premium'].mean()
    avg_quota = category_data['quota'].mean()
    
    return {
        'quota_elasticity': quota_elasticity,
        'competition_elasticity': competition_elasticity,
        'avg_premium': avg_premium,
        'avg_quota': avg_quota,
        'model': elasticity_model
    }

def create_elasticity_scatter_plots(df):
    """Create scatter plots for elasticity analysis"""
    print("📊 Creating elasticity scatter plots...")
    
    # Calculate elasticity for both categories
    elasticity_a = calculate_elasticity(df, 'Category A')
    elasticity_b = calculate_elasticity(df, 'Category B')
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('COE Quota Elasticity Analysis', fontsize=16, fontweight='bold')
    
    # Generate data points for scatter plots
    quota_increases = np.linspace(0, 1000, 100)  # 100 points from 0 to 1000
    
    # Category A
    price_changes_a = []
    for increase in quota_increases:
        if increase == 0:
            price_changes_a.append(0)
        else:
            pct_change_quota = increase / elasticity_a['avg_quota']
            pct_change_price = elasticity_a['quota_elasticity'] * pct_change_quota
            price_changes_a.append(pct_change_price * 100)
    
    # Category B
    price_changes_b = []
    for increase in quota_increases:
        if increase == 0:
            price_changes_b.append(0)
        else:
            pct_change_quota = increase / elasticity_b['avg_quota']
            pct_change_price = elasticity_b['quota_elasticity'] * pct_change_quota
            price_changes_b.append(pct_change_price * 100)
    
    # Plot Category A
    axes[0].scatter(quota_increases, price_changes_a, alpha=0.7, s=30, color='steelblue')
    axes[0].set_title('Category A - Price Change for Quota Increases')
    axes[0].set_xlabel('Quota Increase')
    axes[0].set_ylabel('Price Change (%)')
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line for Category A
    z_a = np.polyfit(quota_increases, price_changes_a, 1)
    p_a = np.poly1d(z_a)
    axes[0].plot(quota_increases, p_a(quota_increases), "r--", alpha=0.8, linewidth=2)
    
    # Add specific data point labels for Category A
    key_points = [100, 200, 500, 1000]
    for point in key_points:
        idx = np.argmin(np.abs(quota_increases - point))
        price_change = price_changes_a[idx]
        axes[0].annotate(f'{price_change:.1f}%', 
                        (quota_increases[idx], price_changes_a[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=9)
    
    # Plot Category B
    axes[1].scatter(quota_increases, price_changes_b, alpha=0.7, s=30, color='steelblue')
    axes[1].set_title('Category B - Price Change for Quota Increases')
    axes[1].set_xlabel('Quota Increase')
    axes[1].set_ylabel('Price Change (%)')
    axes[1].grid(True, alpha=0.3)
    
    # Add trend line for Category B
    z_b = np.polyfit(quota_increases, price_changes_b, 1)
    p_b = np.poly1d(z_b)
    axes[1].plot(quota_increases, p_b(quota_increases), "r--", alpha=0.8, linewidth=2)
    
    # Add specific data point labels for Category B
    for point in key_points:
        idx = np.argmin(np.abs(quota_increases - point))
        price_change = price_changes_b[idx]
        axes[1].annotate(f'{price_change:.1f}%', 
                        (quota_increases[idx], price_changes_b[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=9)
    
    plt.tight_layout()
    plt.savefig('elasticity_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Elasticity scatter plots saved as 'elasticity_scatter_plots.png'")
    
    # Print elasticity results
    print(f"\n📊 Elasticity Results:")
    print(f"Category A - Quota Elasticity: {elasticity_a['quota_elasticity']:.4f}")
    print(f"Category B - Quota Elasticity: {elasticity_b['quota_elasticity']:.4f}")
    print(f"Category A - Average Quota: {elasticity_a['avg_quota']:.0f}")
    print(f"Category B - Average Quota: {elasticity_b['avg_quota']:.0f}")

def main():
    """Main function to run the elasticity analysis"""
    print("🚗 COE Quota Elasticity Analysis")
    print("=" * 50)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create elasticity scatter plots
    create_elasticity_scatter_plots(df)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 