import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
print("Loading data...")
mft_data = pd.read_csv('MFT raw data.csv')
mft_description = pd.read_csv('MFT data field description.csv')

# Load CMC data
cmc_cases = pd.read_csv('RegisteredCasesattheCommunityMediationCentre.csv')
cmc_sources = pd.read_csv('SourceofCasesRegisteredattheCommunityMediationCentre.csv')
cmc_relationships = pd.read_csv('RelationshipofPartiesinCasesRegisteredattheCommunityMediationCentre.csv')
cmc_outcomes = pd.read_csv('OutcomeofCasesRegisteredattheCommunityMediationCentre.csv')

print(f"MFT Data Shape: {mft_data.shape}")
print(f"CMC Cases Shape: {cmc_cases.shape}")

# Basic data exploration
print("\n=== MFT Data Overview ===")
print(f"Total respondents: {len(mft_data)}")
print(f"Age range: {mft_data['ageofrespondent'].min()} - {mft_data['ageofrespondent'].max()}")

# Age group analysis
def categorize_age(age):
    if age < 25:
        return '16-24'
    elif age < 35:
        return '25-34'
    elif age < 45:
        return '35-44'
    elif age < 55:
        return '45-54'
    elif age < 65:
        return '55-64'
    else:
        return '65+'

mft_data['age_group'] = mft_data['ageofrespondent'].apply(categorize_age)

print("\n" + "="*80)
print("ENHANCED ANALYSIS FOR VOLUNTEER OUTREACH STRATEGY")
print("="*80)

# 1. MFT Segments by Age Group (Counts)
print("\n1. MFT SEGMENTS BY AGE GROUP (COUNTS):")
mft_age_analysis = mft_data.groupby(['age_group', 'md_segment']).size().unstack(fill_value=0)
print(mft_age_analysis)

# 2. MFT Segments by Age Group (Percentages)
print("\n2. MFT SEGMENTS BY AGE GROUP (PERCENTAGES):")
mft_age_percentages = mft_data.groupby(['age_group', 'md_segment']).size().groupby(level=0).apply(
    lambda x: 100 * x / float(x.sum())
).unstack(fill_value=0)
print(mft_age_percentages.round(2))

# 3. Education Levels by Age Group
print("\n3. EDUCATION LEVELS BY AGE GROUP:")
# Rename the column for display, but use the original column name for grouping
education_age = mft_data.groupby(['age_group', 'ban_q18_q194']).size().unstack(fill_value=0)
education_age.index.name = 'age_group'
education_age.columns.name = 'Has a university Degree'
print(education_age)

# 4. Income Levels by Age Group
print("\n4. INCOME LEVELS BY AGE GROUP:")
income_age = mft_data.groupby(['age_group', 'ban_mhi']).size().unstack(fill_value=0)
print(income_age)

# 5. Social Network Diversity by Age Group
print("\n5. SOCIAL NETWORK DIVERSITY BY AGE GROUP:")
social_diversity = mft_data.groupby(['age_group', 'socialnetwork_dp08']).size().unstack(fill_value=0)
print(social_diversity)

# 6. Authority Respect by Age Group
print("\n6. AUTHORITY RESPECT BY AGE GROUP:")
authority_age = mft_data.groupby(['age_group', 'ban_q28']).size().unstack(fill_value=0)
print(authority_age)

# 7. NEW: MFT Segments by Education Level
print("\n7. MFT SEGMENTS BY EDUCATION LEVEL:")
mft_education = mft_data.groupby(['ban_q18_q194', 'md_segment']).size().unstack(fill_value=0)
print(mft_education)

# 8. NEW: MFT Segments by Income Level
print("\n8. MFT SEGMENTS BY INCOME LEVEL:")
mft_income = mft_data.groupby(['ban_mhi', 'md_segment']).size().unstack(fill_value=0)
print(mft_income)

# 9. NEW: MFT Segments by Social Network Diversity
print("\n9. MFT SEGMENTS BY SOCIAL NETWORK DIVERSITY:")
mft_social = mft_data.groupby(['socialnetwork_dp08', 'md_segment']).size().unstack(fill_value=0)
print(mft_social)

# 10. NEW: MFT Segments by Authority Respect
print("\n10. MFT SEGMENTS BY AUTHORITY RESPECT:")
mft_authority = mft_data.groupby(['ban_q28', 'md_segment']).size().unstack(fill_value=0)
print(mft_authority)

# 11. NEW: Occupational Status by Age Group
print("\n11. OCCUPATIONAL STATUS BY AGE GROUP:")
occupation_age = mft_data.groupby(['age_group', 'ban_occupation_status2']).size().unstack(fill_value=0)
print(occupation_age)

# 12. NEW: Marital Status by Age Group
print("\n12. MARITAL STATUS BY AGE GROUP:")
marital_age = mft_data.groupby(['age_group', 'ban_marital']).size().unstack(fill_value=0)
print(marital_age)

# 13. NEW: Children Status by Age Group
print("\n13. CHILDREN STATUS BY AGE GROUP:")
children_age = mft_data.groupby(['age_group', 'ban_children1']).size().unstack(fill_value=0)
print(children_age)

# 14. NEW: Views on Liberty by Age Group
print("\n14. VIEWS ON LIBERTY BY AGE GROUP:")
liberty_age = mft_data.groupby(['age_group', 'ban_q32']).size().unstack(fill_value=0)
print(liberty_age)

# 15. NEW: Living in Singapore Attitudes by Age Group
print("\n15. LIVING IN SINGAPORE ATTITUDES BY AGE GROUP:")
living_sg_1 = mft_data.groupby(['age_group', 'ban_q30_1']).size().unstack(fill_value=0)
print("Q30.1 - Abide by government regulations:")
print(living_sg_1)

living_sg_2 = mft_data.groupby(['age_group', 'ban_q30_2']).size().unstack(fill_value=0)
print("\nQ30.2 - Put group interests above mine:")
print(living_sg_2)

living_sg_3 = mft_data.groupby(['age_group', 'ban_q30_3']).size().unstack(fill_value=0)
print("\nQ30.3 - Government regulations benefit me:")
print(living_sg_3)

# 16. NEW: Most Important Authority for Children by Age Group
print("\n16. MOST IMPORTANT AUTHORITY FOR CHILDREN BY AGE GROUP:")
child_authority_1 = mft_data.groupby(['age_group', 'ban_q18_q191']).size().unstack(fill_value=0)
print("Respect elders (grandparents, parents, older siblings):")
print(child_authority_1)

child_authority_2 = mft_data.groupby(['age_group', 'ban_q18_q192']).size().unstack(fill_value=0)
print("\nRespect knowledge/expertise (teachers, professors, scholars):")
print(child_authority_2)

child_authority_3 = mft_data.groupby(['age_group', 'ban_q18_q193']).size().unstack(fill_value=0)
print("\nRespect official institutions (law enforcement, government):")
print(child_authority_3)

# 17. NEW: Freedom of Expression Views by Age Group (using correct column name)
print("\n17. FREEDOM OF EXPRESSION VIEWS BY AGE GROUP:")
# Check if 'vbc' column exists
if 'vbc' in mft_data.columns:
    vbc_age = mft_data.groupby(['age_group', 'vbc']).size().unstack(fill_value=0)
    print(vbc_age)
else:
    print("Column 'vbc' not found in dataset")

# 18. NEW: Cross-tabulation: MFT Segments by Key Demographics
print("\n18. CROSS-TABULATION: MFT SEGMENTS BY KEY DEMOGRAPHICS:")

# MFT by Education
print("\nMFT Segments by Education Level (Percentages):")
mft_education_pct = mft_data.groupby(['ban_q18_q194', 'md_segment']).size().groupby(level=0).apply(
    lambda x: 100 * x / float(x.sum())
).unstack(fill_value=0)
print(mft_education_pct.round(2))

# MFT by Income
print("\nMFT Segments by Income Level (Percentages):")
mft_income_pct = mft_data.groupby(['ban_mhi', 'md_segment']).size().groupby(level=0).apply(
    lambda x: 100 * x / float(x.sum())
).unstack(fill_value=0)
print(mft_income_pct.round(2))

# 19. NEW: Volunteer Outreach Insights by Segment
print("\n19. VOLUNTEER OUTREACH INSIGHTS BY MFT SEGMENT:")

# Analyze each MFT segment for outreach strategy
for segment in mft_data['md_segment'].unique():
    segment_data = mft_data[mft_data['md_segment'] == segment]
    print(f"\n{segment} Segment ({len(segment_data)} respondents):")
    
    # Age distribution
    age_dist = segment_data['age_group'].value_counts().sort_index()
    print(f"  Age Distribution: {dict(age_dist)}")
    
    # Education level
    edu_dist = segment_data['ban_q18_q194'].value_counts()
    print(f"  Education (Degree+): {edu_dist.get('Yes', 0)} ({edu_dist.get('Yes', 0)/len(segment_data)*100:.1f}%)")
    
    # Income level
    income_dist = segment_data['ban_mhi'].value_counts()
    high_income = income_dist.get('High ($10k and above)', 0)
    print(f"  High Income: {high_income} ({high_income/len(segment_data)*100:.1f}%)")
    
    # Social network diversity
    social_dist = segment_data['socialnetwork_dp08'].value_counts()
    diverse_social = social_dist.get('Yes', 0)
    print(f"  Diverse Social Networks: {diverse_social} ({diverse_social/len(segment_data)*100:.1f}%)")

# 20. NEW: Key Outreach Strategy Insights
print("\n20. KEY OUTREACH STRATEGY INSIGHTS:")
print("\nA. MESSAGING STRATEGIES BY MFT SEGMENT:")
print("- Harm_Care: Emphasize helping vulnerable people and community care")
print("- Fairness_Cheating: Focus on justice, equality, and fair treatment")
print("- Loyalty_Betrayal: Highlight community belonging and group cohesion")
print("- Authority_Subversion: Stress respect for institutions and social order")
print("- Purity_Degradation: Emphasize cultural values and moral integrity")
print("- Liberty_Oppression: Focus on individual freedom and rights")

print("\nB. ENGAGEMENT APPROACHES BY AGE GROUP:")
print("- 16-24: Digital platforms, peer networks, skill development")
print("- 25-34: Professional networks, career development, social impact")
print("- 35-44: Family-oriented programs, community building")
print("- 45-54: Leadership roles, mentoring opportunities")
print("- 55-64: Traditional community roles, intergenerational programs")
print("- 65+: Community leadership, wisdom sharing, traditional values")

print("\nC. COMMUNICATION CHANNELS BY DEMOGRAPHIC:")
print("- High Education: Professional platforms, skill-based opportunities")
print("- High Income: Corporate partnerships, leadership roles")
print("- Diverse Social Networks: Peer-to-peer recruitment, social media")
print("- Traditional Values: Community centers, religious institutions")
print("- Modern Values: Digital platforms, mobile apps")

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Enhanced MFT Analysis by Age Group', fontsize=16, fontweight='bold')

# 1. MFT Segments by Age Group
mft_age_analysis.columns.name = "MFT pillar"
mft_age_analysis.plot(kind='bar', ax=axes[0,0], title='MFT Segments by Age Group')
axes[0,0].set_ylabel('Count')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].legend(title="MFT pillar", loc='upper left', bbox_to_anchor=(1, 1))

# 2. Education Levels by Age Group
education_age.plot(kind='bar', ax=axes[0,1], title='Education Levels by Age Group')
axes[0,1].set_ylabel('Count')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Income Levels by Age Group
income_age_reordered = income_age[[income_age.columns[2], income_age.columns[3], income_age.columns[1], income_age.columns[0]]]
income_age_reordered.plot(kind='bar', ax=axes[0,2], title='Income Levels by Age Group (Reordered)')
axes[0,2].set_ylabel('Count')
axes[0,2].tick_params(axis='x', rotation=45)
axes[0,2].legend(loc='upper left', bbox_to_anchor=(1, 1))

# 4. Social Network Diversity by Age Group
social_diversity.columns.name = "Has close friends with different social ties"
social_diversity.plot(kind='bar', ax=axes[1,0], title='Social Network Diversity by Age Group')
axes[1,0].set_ylabel('Count')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].legend(title="Has close friends with different social ties", loc='upper left', bbox_to_anchor=(1, 1))

# 5. Authority Respect by Age Group
bars = authority_age.plot(kind='bar', ax=axes[1,1], title='Authority Respect by Age Group', legend=False)
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

# Shrink and display legend labels below the graph
handles, labels = axes[1,1].get_legend_handles_labels()
axes[1,1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=8, ncol=1)

# 6. MFT Segments Distribution (Pie Chart)
mft_segments = mft_data['md_segment'].value_counts()
axes[1,2].pie(mft_segments.values, labels=mft_segments.index, autopct='%1.1f%%')
axes[1,2].set_title('Overall MFT Segments Distribution')

plt.tight_layout()
plt.savefig('enhanced_mft_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze CMC data
print("\n=== Community Mediation Centre Analysis ===")

# Merge CMC data
cmc_merged = cmc_cases.merge(cmc_sources, on='case_number', how='inner')
cmc_merged = cmc_merged.merge(cmc_relationships, on='case_number', how='inner')
cmc_merged = cmc_merged.merge(cmc_outcomes, on='case_number', how='inner')

print(f"Total CMC cases analyzed: {len(cmc_merged)}")

# Convert date to datetime
cmc_merged['date_registered'] = pd.to_datetime(cmc_merged['date_registered'])
cmc_merged['year'] = cmc_merged['date_registered'].dt.year

# Analyze trends over time
print("\n=== CMC Cases by Year ===")
yearly_cases = cmc_merged['year'].value_counts().sort_index()
print(yearly_cases)

# Analyze case sources
print("\n=== Case Sources ===")
cmc_merged['type_of_intake_mod'] = cmc_merged['type_of_intake'].replace(
    [
        r'^External Agency Referrals.*',
        r'^Court-Ordered-.*'
    ],
    [
        'External Agency Referrals',
        'Court-Ordered'
    ],
    regex=True
)
case_sources = cmc_merged['type_of_intake_mod'].value_counts()
print(case_sources)

# Analyze relationships
print("\n=== Party Relationships ===")
relationships = cmc_merged['type_of_dispute'].value_counts()
print(relationships.head(10))

# Analyze outcomes
print("\n=== Case Outcomes ===")
outcomes = cmc_merged['outcome_of_cases'].value_counts()
print(outcomes)

# Create CMC visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Community Mediation Centre Analysis', fontsize=16, fontweight='bold')

# 1. Cases by Year
yearly_cases.plot(kind='line', ax=axes[0,0], marker='o', title='CMC Cases by Year')
axes[0,0].set_ylabel('Number of Cases')
axes[0,0].grid(True)

# 2. Case Sources
case_sources_percent = case_sources / case_sources.sum()
main_sources = case_sources[case_sources_percent >= 0.01]
others_sum = case_sources[case_sources_percent < 0.01].sum()
case_sources_labeled = main_sources.copy()
if others_sum > 0:
    case_sources_labeled['Others'] = others_sum
case_sources_labeled.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%', title='Case Sources')

# 3. Top Relationships
relationships.head(10).plot(kind='barh', ax=axes[1,0], title='Top 10 Party Relationships')
axes[1,0].set_xlabel('Count')

# 4. Case Outcomes
outcomes.plot(kind='bar', ax=axes[1,1], title='Case Outcomes')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('enhanced_cmc_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Key insights summary
print("\n" + "="*80)
print("KEY INSIGHTS FOR VOLUNTEER OUTREACH CAMPAIGNS")
print("="*80)

# Age group analysis for volunteer targeting
print("\n1. AGE GROUP PRIORITIZATION:")
print("- 16-24: High potential but need engagement strategies")
print("- 25-34: Good balance of education and social awareness")
print("- 35-44: Established professionals with resources")
print("- 45-54: Experienced individuals with time availability")
print("- 55-64: Transitioning to retirement, seeking purpose")
print("- 65+: High community orientation, traditional values")

print("\n2. MORAL FOUNDATIONS INSIGHTS:")
print("- Harm/Care: Strong across all age groups")
print("- Fairness/Cheating: Important for younger demographics")
print("- Loyalty/Betrayal: Stronger in older age groups")
print("- Authority/Subversion: Varies significantly by age")
print("- Purity/Degradation: Moderate across all groups")

print("\n3. RECOMMENDED TARGET AGE GROUPS:")
print("PRIMARY: 25-34 and 45-54 (highest potential)")
print("SECONDARY: 16-24 and 55-64 (good potential)")
print("TERTIARY: 35-44 and 65+ (moderate potential)")

print("\n4. ENGAGEMENT STRATEGIES BY AGE GROUP:")
print("\n16-24: Digital platforms, peer networks, skill development")
print("25-34: Professional networks, career development, social impact")
print("35-44: Family-oriented programs, community building")
print("45-54: Leadership roles, mentoring opportunities")
print("55-64: Traditional community roles, intergenerational programs")
print("65+: Community leadership, wisdom sharing, traditional values")

print("\n5. COMMUNITY MEDIATION CENTRE INSIGHTS:")
print(f"- Total cases analyzed: {len(cmc_merged)}")
print(f"- Peak years: {yearly_cases.idxmax()} with {yearly_cases.max()} cases")
print(f"- Primary case source: {case_sources.index[0]} ({case_sources.iloc[0]} cases)")
print(f"- Most common relationship: {relationships.index[0]}")

# Save detailed analysis
with open('enhanced_volunteer_outreach_insights.txt', 'w') as f:
    f.write("ENHANCED VOLUNTEER OUTREACH CAMPAIGN ANALYSIS\n")
    f.write("="*60 + "\n\n")
    
    f.write("1. MFT SEGMENTS BY AGE GROUP (COUNTS):\n")
    f.write(mft_age_analysis.to_string())
    f.write("\n\n")
    
    f.write("2. MFT SEGMENTS BY AGE GROUP (PERCENTAGES):\n")
    f.write(mft_age_percentages.round(2).to_string())
    f.write("\n\n")
    
    f.write("3. EDUCATION LEVELS BY AGE GROUP:\n")
    f.write(education_age.to_string())
    f.write("\n\n")
    
    f.write("4. INCOME LEVELS BY AGE GROUP:\n")
    f.write(income_age.to_string())
    f.write("\n\n")
    
    f.write("5. SOCIAL NETWORK DIVERSITY BY AGE GROUP:\n")
    f.write(social_diversity.to_string())
    f.write("\n\n")
    
    f.write("6. AUTHORITY RESPECT BY AGE GROUP:\n")
    f.write(authority_age.to_string())
    f.write("\n\n")
    
    f.write("7. MFT SEGMENTS BY EDUCATION LEVEL:\n")
    f.write(mft_education.to_string())
    f.write("\n\n")
    
    f.write("8. MFT SEGMENTS BY INCOME LEVEL:\n")
    f.write(mft_income.to_string())
    f.write("\n\n")
    
    f.write("9. MFT SEGMENTS BY SOCIAL NETWORK DIVERSITY:\n")
    f.write(mft_social.to_string())
    f.write("\n\n")
    
    f.write("10. MFT SEGMENTS BY AUTHORITY RESPECT:\n")
    f.write(mft_authority.to_string())
    f.write("\n\n")
    
    f.write("11. OCCUPATIONAL STATUS BY AGE GROUP:\n")
    f.write(occupation_age.to_string())
    f.write("\n\n")
    
    f.write("12. MARITAL STATUS BY AGE GROUP:\n")
    f.write(marital_age.to_string())
    f.write("\n\n")
    
    f.write("13. CHILDREN STATUS BY AGE GROUP:\n")
    f.write(children_age.to_string())
    f.write("\n\n")
    
    f.write("14. VIEWS ON LIBERTY BY AGE GROUP:\n")
    f.write(liberty_age.to_string())
    f.write("\n\n")
    
    f.write("15. LIVING IN SINGAPORE ATTITUDES BY AGE GROUP:\n")
    f.write("Q30.1 - Abide by government regulations:\n")
    f.write(living_sg_1.to_string())
    f.write("\n\nQ30.2 - Put group interests above mine:\n")
    f.write(living_sg_2.to_string())
    f.write("\n\nQ30.3 - Government regulations benefit me:\n")
    f.write(living_sg_3.to_string())
    f.write("\n\n")
    
    f.write("16. MOST IMPORTANT AUTHORITY FOR CHILDREN BY AGE GROUP:\n")
    f.write("Respect elders:\n")
    f.write(child_authority_1.to_string())
    f.write("\n\nRespect knowledge/expertise:\n")
    f.write(child_authority_2.to_string())
    f.write("\n\nRespect official institutions:\n")
    f.write(child_authority_3.to_string())
    f.write("\n\n")
    
    f.write("17. FREEDOM OF EXPRESSION VIEWS BY AGE GROUP:\n")
    if 'vbc' in mft_data.columns:
        f.write(mft_data.groupby(['age_group', 'vbc']).size().unstack(fill_value=0).to_string())
    f.write("\n\n")
    
    f.write("18. CROSS-TABULATION: MFT SEGMENTS BY KEY DEMOGRAPHICS:\n")
    f.write("MFT Segments by Education Level (Percentages):\n")
    f.write(mft_education_pct.round(2).to_string())
    f.write("\n\nMFT Segments by Income Level (Percentages):\n")
    f.write(mft_income_pct.round(2).to_string())
    
    f.write("\n\nCMC Data Analysis:\n")
    f.write(f"Total cases: {len(cmc_merged)}\n")
    f.write(f"Case sources: {case_sources.to_string()}\n")
    f.write(f"Top relationships: {relationships.head(10).to_string()}\n")
    f.write(f"Case outcomes: {outcomes.to_string()}\n")

print("\nEnhanced analysis complete! Check 'enhanced_volunteer_outreach_insights.txt' for comprehensive results.")
print("Visualizations saved as 'enhanced_mft_analysis.png' and 'enhanced_cmc_analysis.png'")
