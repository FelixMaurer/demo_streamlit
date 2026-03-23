import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr, ttest_ind, f_oneway
import statsmodels.api as sm

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# -----------------------------------------------------------------------------
# 1. Data Generation & Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_or_create_data():
    file_path = 'experiment_data.parquet'
    if not os.path.exists(file_path):
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 300
        tags = np.random.choice(['Baseline', 'Condition_A', 'Condition_B'], n_samples)
        repulsion = np.random.uniform(0.5, 5.0, n_samples)
        
        # M0_M2_Ratio is dependent on Repulsion
        ratio = []
        for t, r in zip(tags, repulsion):
            if t == 'Baseline':
                ratio.append(1.5 * r + 2.0 + np.random.normal(0, 0.5))
            elif t == 'Condition_A':
                ratio.append(2.8 * r + 0.5 + np.random.normal(0, 0.8))
            else:
                ratio.append(0.8 * r + 4.0 + np.random.normal(0, 0.4))
                
        df = pd.DataFrame({
            'Tag': tags, 
            'Repulsion': repulsion, 
            'M0_M2_Ratio': ratio
        })
        # Save as Parquet
        df.to_parquet(file_path, engine='pyarrow')
        return df
    
    return pd.read_parquet(file_path, engine='pyarrow')

df = load_or_create_data()

# -----------------------------------------------------------------------------
# 2. Sidebar & Data Selection
# -----------------------------------------------------------------------------
st.sidebar.header("Filter Data")
available_tags = df['Tag'].unique().tolist()
selected_tags = st.sidebar.multiselect("Select Tags to Analyze", available_tags, default=available_tags)

if not selected_tags:
    st.warning("Please select at least one tag from the sidebar.")
    st.stop()

# Filter dataframe based on selection
filtered_df = df[df['Tag'].isin(selected_tags)]

# -----------------------------------------------------------------------------
# 3. Main Dashboard UI
# -----------------------------------------------------------------------------
st.title("Experimental Analysis: Repulsion vs M0/M2 Ratio")

st.subheader("Raw Data Preview")
st.dataframe(filtered_df.head(10), use_container_width=True)

col1, col2 = st.columns(2)

# --- Boxplotting ---
with col1:
    st.subheader("Distribution by Tag")
    fig_box = px.box(
        filtered_df, 
        x="Tag", 
        y="M0_M2_Ratio", 
        color="Tag",
        title="Boxplot of M0_M2_Ratio"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# --- Significance Testing ---
with col2:
    st.subheader("Significance Testing")
    st.write("Testing for differences in the mean `M0_M2_Ratio` across selected tags.")
    
    if len(selected_tags) == 1:
        st.info("Select at least two tags to perform significance testing.")
    elif len(selected_tags) == 2:
        group1 = filtered_df[filtered_df['Tag'] == selected_tags[0]]['M0_M2_Ratio']
        group2 = filtered_df[filtered_df['Tag'] == selected_tags[1]]['M0_M2_Ratio']
        stat, p_val = ttest_ind(group1, group2)
        st.write(f"**Independent T-Test** between {selected_tags[0]} and {selected_tags[1]}:")
        st.write(f"- T-statistic: {stat:.4f}")
        st.write(f"- $p$-value: {p_val:.4e}")
        if p_val < 0.05:
            st.success("Result is statistically significant ($p < 0.05$).")
        else:
            st.warning("Result is not statistically significant.")
    else:
        # ANOVA for > 2 groups
        groups = [filtered_df[filtered_df['Tag'] == tag]['M0_M2_Ratio'] for tag in selected_tags]
        stat, p_val = f_oneway(*groups)
        st.write("**One-Way ANOVA** across all selected groups:")
        st.write(f"- F-statistic: {stat:.4f}")
        st.write(f"- $p$-value: {p_val:.4e}")
        if p_val < 0.05:
            st.success("Result is statistically significant ($p < 0.05$).")
        else:
            st.warning("Result is not statistically significant.")

st.markdown("---")

# --- Plotting & Pearson Correlation ---
st.subheader("Correlation & Regression Fit")

col3, col4 = st.columns(2)

with col3:
    fig_scatter = px.scatter(
        filtered_df, 
        x="Repulsion", 
        y="M0_M2_Ratio", 
        color="Tag", 
        trendline="ols",
        title="Scatter Plot with OLS Trendlines"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Pearson Correlation
    r_stat, p_val_pearson = pearsonr(filtered_df['Repulsion'], filtered_df['M0_M2_Ratio'])
    st.write(f"**Overall Pearson Correlation ($r$)**: {r_stat:.4f} ($p$-value: {p_val_pearson:.4e})")

# --- Model Fitting ---
with col4:
    st.write("Ordinary Least Squares (OLS) Linear Regression Fit Model:")
    st.latex(r"\text{M0\_M2\_Ratio} = \beta_0 + \beta_1 \cdot \text{Repulsion} + \epsilon")
    
    # Fit model using statsmodels
    X = filtered_df['Repulsion']
    Y = filtered_df['M0_M2_Ratio']
    
    # Add constant for the intercept
    X_with_const = sm.add_constant(X)
    model = sm.OLS(Y, X_with_const).fit()
    
    # Display summary statistics
    st.text(model.summary().as_text())
