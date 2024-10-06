import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import joblib

# Set page config
st.set_page_config(page_title="Comet Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to improve the UI
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .Widget>label {
        color: #fafafa;
        font-weight: 300;
    }
    .stTextInput>div>div>input {
        color: #4f8bf9;
    }
    .stSelectbox>div>div>select {
        color: #4f8bf9;
    }
    .stTab {
        background-color: #31333F;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
    .stTab[data-baseweb="tab"][aria-selected="true"] {
        background-color: #4f8bf9;
    }
    .stPlotlyChart {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #4f8bf9;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    data = pd.read_csv('sbdb_query_results_sample.csv')
    # If 'full_name' column doesn't exist, create it from available columns
    if 'full_name' not in data.columns:
        if 'name' in data.columns:
            data['full_name'] = data['name']
        elif 'pdes' in data.columns:
            data['full_name'] = data['pdes']
        else:
            # If no suitable column is found, create a generic name
            data['full_name'] = [f"Object_{i}" for i in range(len(data))]
    return data

# Interactive scatter plot
def interactive_scatter_plot_magnitude_vs_diameter(data):
    scatter_data = data[['H', 'diameter', 'full_name']].dropna()
    fig = px.scatter(scatter_data, x='H', y='diameter', 
                     title='Absolute Magnitude (H) vs Diameter',
                     labels={'H': 'Absolute Magnitude (H)', 'diameter': 'Diameter (km)'},
                     hover_data=['full_name'])
    fig.update_traces(marker=dict(size=8, color="#4f8bf9", opacity=0.7),
                      hovertemplate='<b>%{customdata[0]}</b><br>H: %{x:.2f}<br>Diameter: %{y:.2f} km')
    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

# Interactive histogram
def interactive_histogram_orbital_eccentricity(data):
    eccentricity_data = data['e'].dropna()
    fig = px.histogram(eccentricity_data, x='e', nbins=50,
                       title='Distribution of Orbital Eccentricity (e)',
                       labels={'e': 'Eccentricity (e)', 'count': 'Frequency'},
                       marginal='box')
    fig.update_traces(marker=dict(color="#4f8bf9", opacity=0.7))
    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

# Comet details
def display_comet_details(data):
    st.header("Comet Details")
    if 'full_name' in data.columns:
        comet_name = st.selectbox("Select a comet", data['full_name'].tolist())
        
        if comet_name:
            comet_data = data[data['full_name'] == comet_name].iloc[0]
            st.subheader(f"Details for {comet_name}")
            col1, col2 = st.columns(2)
            for i, (column, value) in enumerate(comet_data.items()):
                if column != 'full_name':
                    with col1 if i % 2 == 0 else col2:
                        st.metric(label=column, value=value)
    else:
        st.warning("Comet names are not available in the dataset.")

# Search functionality
def search_comets(data):
    st.sidebar.header("Search Comets")
    if 'full_name' in data.columns:
        search_term = st.sidebar.text_input("Enter comet name or designation")
        if search_term:
            filtered_data = data[data['full_name'].str.contains(search_term, case=False, na=False)]
            return filtered_data
    else:
        st.sidebar.warning("Search functionality is not available due to missing comet names.")
    return data

# Comparison tool
def compare_comets(data):
    st.header("Comet Comparison Tool")
    if 'full_name' in data.columns:
        col1, col2 = st.columns(2)
        with col1:
            comet1 = st.selectbox("Select first comet", data['full_name'].tolist(), key='comet1')
        with col2:
            comet2 = st.selectbox("Select second comet", data['full_name'].tolist(), key='comet2')
        
        if comet1 and comet2:
            comet1_data = data[data['full_name'] == comet1].iloc[0]
            comet2_data = data[data['full_name'] == comet2].iloc[0]
            
            for column in data.columns:
                if column != 'full_name':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label=f"{column} ({comet1})", value=comet1_data[column])
                    with col2:
                        st.metric(label=f"{column} ({comet2})", value=comet2_data[column])
    else:
        st.warning("Comet comparison is not available due to missing comet names.")

# Statistical summary
def statistical_summary(data):
    st.header("Statistical Summary")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    summary = data[numeric_columns].describe()
    st.dataframe(summary.style.highlight_max(axis=0))
    
    st.subheader("Key Findings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total number of comets", len(data))
    
    if 'diameter' in data.columns:
        largest_comet = data.loc[data['diameter'].idxmax()]
        with col2:
            st.metric("Largest comet", f"{largest_comet['full_name']}", f"{largest_comet['diameter']:.2f} km")
    
    if 'e' in data.columns:
        most_eccentric_comet = data.loc[data['e'].idxmax()]
        with col3:
            st.metric("Most eccentric orbit", f"{most_eccentric_comet['full_name']}", f"{most_eccentric_comet['e']:.4f}")
    
    if 'q' in data.columns:
        closest_approach_comet = data.loc[data['q'].idxmin()]
        st.metric("Closest approach to the Sun", f"{closest_approach_comet['full_name']}", f"{closest_approach_comet['q']:.4f} AU")

# Add this function to handle pagination
def paginate_dataframe(dataframe, page_size, page_num):
    total_pages = math.ceil(len(dataframe) / page_size)
    page_num = max(1, min(page_num, total_pages))
    start = (page_num - 1) * page_size
    end = start + page_size
    return dataframe.iloc[start:end], total_pages, page_num

# Predict diameter using pre-trained model
def predict_diameter(data):
    st.header("Diameter Prediction")
    
    # Load the pre-trained model
    model = joblib.load('comet_diameter_model.joblib')
    
    features = ['H', 'e', 'a', 'q', 'i', 'om', 'w']
    
    # Allow user to input values for prediction
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f"Enter {feature}")
    
    if st.button("Predict Diameter"):
        prediction = model.predict([list(user_input.values())])
        st.write(f"Predicted Diameter: {prediction[0]:.2f} km")

    # Optional: Display feature importances if available
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importances")
        importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=False)
        st.bar_chart(importances.set_index('feature'))

# Main function
def main():
    st.title("🌠 Comet Analysis Dashboard")
    
    # Load data
    data = load_data()
    
    # Apply search filter
    data = search_comets(data)
    
    # Create tabs
    tabs = st.tabs(["📊 Plots", "📋 Data Table", "🔍 Comet Details", "🌌 Orbital Elements", "🔬 Comparison Tool", "📈 Statistical Summary", "🤖 Diameter Prediction"])
    
    # Plots tab
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            if 'H' in data.columns and 'diameter' in data.columns:
                fig = interactive_scatter_plot_magnitude_vs_diameter(data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Magnitude or diameter data not available")
        
        with col2:
            if 'e' in data.columns:
                fig = interactive_histogram_orbital_eccentricity(data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Eccentricity data not available")
        
        if 'a' in data.columns and 'e' in data.columns:
            fig = px.scatter(data, x='a', y='e', hover_data=['full_name'],
                             labels={'a': 'Semi-major Axis (AU)', 'e': 'Eccentricity'},
                             title='Semi-major Axis vs Eccentricity')
            fig.update_traces(marker=dict(size=8, color="#4f8bf9", opacity=0.7))
            fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    # Data Table tab
    with tabs[1]:
        st.header("Comet Data Table")
        
        # Add pagination controls
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            page_size = st.selectbox("Rows per page", [10, 20, 50, 100], index=1)
        with col2:
            page_num = st.number_input("Page", min_value=1, value=1, step=1)
        
        # Paginate the dataframe
        paginated_data, total_pages, current_page = paginate_dataframe(data, page_size, page_num)
        
        # Display paginated data
        st.dataframe(paginated_data)
        
        # Display pagination info
        st.write(f"Page {current_page} of {total_pages}")
        
        # Add download button for full dataset
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download full dataset as CSV",
            data=csv,
            file_name="comet_data.csv",
            mime="text/csv",
        )
    
    # Comet Details tab
    with tabs[2]:
        display_comet_details(data)
    
    # Orbital Elements tab
    with tabs[3]:
        st.header("Orbital Elements Distribution")
        orbital_elements = ['e', 'a', 'q', 'i', 'om', 'w']
        element_names = ['Eccentricity', 'Semi-major Axis (AU)', 'Perihelion Distance (AU)', 
                         'Inclination (deg)', 'Longitude of Ascending Node (deg)', 'Argument of Perihelion (deg)']
        
        for elem, name in zip(orbital_elements, element_names):
            if elem in data.columns:
                fig = px.histogram(data, x=elem, title=f'Distribution of {name}')
                fig.update_traces(marker=dict(color="#4f8bf9", opacity=0.7))
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
    
    # Comparison Tool tab
    with tabs[4]:
        compare_comets(data)
    
    # Statistical Summary tab
    with tabs[5]:
        statistical_summary(data)
    
    # Diameter Prediction tab
    with tabs[6]:
        predict_diameter(data)

if __name__ == "__main__":
    main()
