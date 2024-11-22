import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
original_file_path = 'HDI.csv'  # Path to the original file
cleaned_file_path = 'cleaned_HDI.csv'  # Path to the cleaned file

# Load datasets
original_data = pd.read_csv(original_file_path)
cleaned_data = pd.read_csv(cleaned_file_path)

# Streamlit App Title
st.title("HDI Dataset: Original vs Cleaned Data Visualization")

# Sidebar Options
st.sidebar.header("Visualization Settings")

# Select Plot Type
plot_type = st.sidebar.selectbox(
    "Select the Plot Type",
    [
        "Histogram",
        "Bar Plot",
        "Scatter Plot",
        "Box Plot",
        "Violin Plot",
        "Heatmap",
    ]
)

# Select Column for Visualization
columns = original_data.columns.intersection(cleaned_data.columns).tolist()
selected_column = st.sidebar.selectbox("Select the Column", columns)

# Select Year (if applicable)
if "year" in original_data.columns and "year" in cleaned_data.columns:
    year = st.sidebar.selectbox("Select Year", original_data["year"].unique())
    original_data = original_data[original_data["year"] == year]
    cleaned_data = cleaned_data[cleaned_data["year"] == year]

# Option to Include Country Names
if "country" in original_data.columns and "country" in cleaned_data.columns:
    include_countries = st.sidebar.checkbox("Include Country Names", value=True)
else:
    include_countries = False

# Plot Comparison: Original vs Cleaned
st.header(f"{plot_type} for {selected_column}: Original vs Cleaned Data")

# Plot Types
if plot_type == "Histogram":
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(original_data[selected_column], kde=True, ax=axes[0], color='blue')
    axes[0].set_title("Original Data")
    sns.histplot(cleaned_data[selected_column], kde=True, ax=axes[1], color='green')
    axes[1].set_title("Cleaned Data")
    st.pyplot(fig)

elif plot_type == "Bar Plot":
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    original_data[selected_column].value_counts().plot(kind="bar", ax=axes[0], color='blue')
    axes[0].set_title("Original Data")
    cleaned_data[selected_column].value_counts().plot(kind="bar", ax=axes[1], color='green')
    axes[1].set_title("Cleaned Data")
    st.pyplot(fig)

elif plot_type == "Scatter Plot":
    second_column = st.sidebar.selectbox(
        "Select the Second Column for Scatter Plot",
        [col for col in columns if col != selected_column]
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(x=original_data[selected_column], y=original_data[second_column], ax=axes[0], color='blue')
    axes[0].set_title("Original Data")
    sns.scatterplot(x=cleaned_data[selected_column], y=cleaned_data[second_column], ax=axes[1], color='green')
    axes[1].set_title("Cleaned Data")
    st.pyplot(fig)

elif plot_type == "Box Plot":
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=original_data[selected_column], ax=axes[0], color='blue')
    axes[0].set_title("Original Data")
    sns.boxplot(data=cleaned_data[selected_column], ax=axes[1], color='green')
    axes[1].set_title("Cleaned Data")
    st.pyplot(fig)

elif plot_type == "Violin Plot":
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.violinplot(data=original_data[selected_column], ax=axes[0], color='blue')
    axes[0].set_title("Original Data")
    sns.violinplot(data=cleaned_data[selected_column], ax=axes[1], color='green')
    axes[1].set_title("Cleaned Data")
    st.pyplot(fig)

elif plot_type == "Heatmap":
    numeric_original = original_data.select_dtypes(include=['float64', 'int64'])
    numeric_cleaned = cleaned_data.select_dtypes(include=['float64', 'int64'])
    if not numeric_original.empty and not numeric_cleaned.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(numeric_original.corr(), annot=True, cmap="coolwarm", ax=axes[0])
        axes[0].set_title("Original Data")
        sns.heatmap(numeric_cleaned.corr(), annot=True, cmap="coolwarm", ax=axes[1])
        axes[1].set_title("Cleaned Data")
        st.pyplot(fig)
    else:
        st.warning("No numeric data available for a heatmap.")

# Display Dataset Preview with Country Names (Optional)
if include_countries and "country" in original_data.columns:
    st.header("Dataset Preview with Country Names")
    st.write("Original Data")
    st.write(original_data[["country", selected_column]])
    st.write("Cleaned Data")
    st.write(cleaned_data[["country", selected_column]])

# Full Dataset Display
st.header("Full Dataset Preview")
st.write("Original Data")
st.write(original_data)
st.write("Cleaned Data")
st.write(cleaned_data)
