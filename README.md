To set up the folders `S1Q1_ResaleFlatPrices` and `S1Q2_COEQuota`, follow these steps for each:

---

### S1Q1_ResaleFlatPrices Setup

1. **Download the required CSV files via the links:**
   - [CEASalespersonsPropertyTransactionRecordsresidential.csv](https://data.gov.sg/datasets/d_ee7e46d3c57f7865790704632b0aef71/view)
   - [Resale flat prices based on registration date from Jan-2017 onwards.csv](https://data.gov.sg/collections/189/view)

2. **Place the downloaded CSV files into the `S1Q1_ResaleFlatPrices` folder.**

3. **Run the Python analysis script:**
   ```bash
   cd S1Q1_ResaleFlatPrices
   python pre_covid_impact_analysis.py
   ```

4. **To generate visualisations:**
   ```bash
   python data_story_visualizations.py
   ```

---

### S1Q2_COEQuota Setup

1. **Download the required CSV files:**
   - [COEBiddingResultsPrices.csv](https://data.gov.sg/datasets/d_69b3380ad7e51aff3a7dcc84eba52b8a/view)
   - [MotorVehicleQuotaQuotaPremiumAndPrevailingQuotaPremiumMonthly.csv](https://data.gov.sg/datasets/d_22094bf608253d36c0c63b52d852dd6e/view)

2. **Place the downloaded CSV files into the `S1Q2_COEQuota` folder.**

3. **Run the Python analysis script:**
   ```bash
   cd S1Q2_COEQuota
   python coe_price_prediction_model.py
   ```

4. **Generate elastic scatter plots:**
   ```bash
   python elasticity_scatter_plots.py
   ```

### S2Q3_VolunteerOutreach Setup

1. **Download the required CSV file:**
   - [MFT raw data.csv](https://data.gov.sg/collections/395/view)  
   - [RegisteredCasesattheCommunityMediationCentre.csv](https://data.gov.sg/datasets/d_64a479fdc6e9487cd70838f769273b59/view)
   - [SourceofCasesRegisteredattheCommunityMediationCentre.csv](https://data.gov.sg/datasets/d_9439ea87f910688c8218c361df531365/view)
   - [RelationshipofPartiesinCasesRegisteredattheCommunityMediationCentre.csv](https://data.gov.sg/datasets/d_896523571502b305e313b249f555c244/view)
   - [OutcomeofCasesRegisteredattheCommunityMediationCentre.csv](https://data.gov.sg/datasets/d_78b6f3b7151f01bab5740ddfe2f1b5f4/view)

2. **Place the downloaded CSV files into the `S2Q3_VolunteerOutreach` folder.**

3. **Run the enhanced analysis script:**
   ```bash
   cd S2Q3_VolunteerOutreach
   python volunteer_outreach_analysis.py
   ```

4. **Review the generated outputs:**
   - The script will print key insights to the console.
   - Detailed analysis will be saved to `enhanced_volunteer_outreach_insights.txt`.
   - Visualizations (if any) will be saved as image files in the same folder.

---

**Note:**  
- Ensure you have installed all required Python packages (see `requirements.txt`).

