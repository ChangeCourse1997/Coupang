import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# Paghe configs
st.set_page_config(
    page_title="Product Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #ff6b6b;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff5252;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the product data"""
    try:
        df = pd.read_csv('product data.csv')
        # Clean the data
        df = df.drop_duplicates()
        
        if 'Price' in df.columns:
            df['Price_Numeric'] = df['Price'].str.replace('$', '').str.replace(',', '').astype(float)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def clean_text(text):
    """Clean product names for clustering"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def cluster_products_jaccard(df, product_column, similarity_threshold=0.5):
    """
    Cluster products using Jaccard similarity with threshold-based clustering
    """
    # Clean product names
    products = [clean_text(p) for p in df[product_column]]
    n_products = len(products)
    
    word_sets = [set(product.split()) for product in products]
    
    # Calculate Jaccard similarity matrix
    similarity_matrix = np.zeros((n_products, n_products))
    
    for i in range(n_products):
        for j in range(i, n_products):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                sim = jaccard_similarity(word_sets[i], word_sets[j])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
    
    clusters = [-1] * n_products  
    cluster_id = 0
    
    for i in range(n_products):
        if clusters[i] == -1:  # For no assignments
            # Start new cluster
            clusters[i] = cluster_id
            
            # Add all similar products to this cluster
            for j in range(i + 1, n_products):
                if clusters[j] == -1 and similarity_matrix[i][j] >= similarity_threshold:
                    clusters[j] = cluster_id
            
            cluster_id += 1
    
    result_df = df.copy()
    result_df['cluster'] = clusters
    
    return result_df, similarity_matrix

def get_best_products_per_cluster(clustered_df):
    """Get the best product from each cluster based on Units Sold and Price"""
    df_final = pd.DataFrame()
    for cluster in clustered_df['cluster'].unique():
        best_product = clustered_df[clustered_df['cluster']==cluster].sort_values(
            ['Units Sold','Price_Numeric'], ascending=[False,True]
        ).iloc[0:1]
        df_final = pd.concat([df_final, best_product])
    return df_final.reset_index(drop=True)

def main():
    st.title("📊 Product Data Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data.")
        return
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Dataset overview
    st.header("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(df))
    with col2:
        st.metric("Unique Products", df['Product Name'].nunique())
    with col3:
        st.metric("Total Units Sold", f"{df['Units Sold'].sum():,}")
    with col4:
        st.metric("Unique Locations", df['Location'].nunique())
    
    # Basic Statistics
    st.header("📊 Statistical Summary")
    
    # Price Statistics
    if 'Price_Numeric' in df.columns:
        st.subheader("💰 Price Statistics")
        
        price_stats = df['Price_Numeric'].describe()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Mean Price", f"${price_stats['mean']:.2f}")
        with col2:
            st.metric("Median Price", f"${price_stats['50%']:.2f}")
        with col3:
            st.metric("Min Price", f"${price_stats['min']:.2f}")
        with col4:
            st.metric("Max Price", f"${price_stats['max']:.2f}")
        with col5:
            st.metric("Std Dev", f"${price_stats['std']:.2f}")
    
    # Units Sold Statistics
    st.subheader("📦 Units Sold Statistics")
    
    units_stats = df['Units Sold'].describe()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Mean Units", f"{units_stats['mean']:.0f}")
    with col2:
        st.metric("Median Units", f"{units_stats['50%']:.0f}")
    with col3:
        st.metric("Min Units", f"{units_stats['min']:.0f}")
    with col4:
        st.metric("Max Units", f"{units_stats['max']:.0f}")
    with col5:
        st.metric("Std Dev", f"{units_stats['std']:.0f}")
    
    # Visualizations
    st.header("📈 Data Visualizations")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        fig_price = px.histogram(
            df, 
            x='Price_Numeric', 
            title='Price Distribution',
            nbins=30,
            color_discrete_sequence=['#ff6b6b']
        )
        fig_price.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Units sold distribution
        fig_units = px.histogram(
            df, 
            x='Units Sold', 
            title='Units Sold Distribution',
            nbins=30,
            color_discrete_sequence=['#4ecdc4']
        )
        fig_units.update_layout(
            xaxis_title="Units Sold",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_units, use_container_width=True)
    
    # Location analysis
    if 'Location' in df.columns:
        location_counts = df['Location'].value_counts().head(10)
        fig_location = px.bar(
            x=location_counts.index,
            y=location_counts.values,
            title='Top Locations by Product Count',
            color_discrete_sequence=['#45b7d1']
        )
        fig_location.update_layout(
            xaxis_title="Location",
            yaxis_title="Number of Products"
        )
        st.plotly_chart(fig_location, use_container_width=True)
    
    # Scatter plot: Price vs Units Sold
    fig_scatter = px.scatter(
        df, 
        x='Price_Numeric', 
        y='Units Sold',
        title='Price vs Units Sold',
        color='Location' if 'Location' in df.columns else None,
        hover_data=['Product Name']
    )
    fig_scatter.update_layout(
        xaxis_title="Price ($)",
        yaxis_title="Units Sold"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Clustering Section
    st.header("🔍 Product Clustering Analysis")
    
    # Clustering parameters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Higher values create fewer, more similar clusters"
        )
        
        cluster_button = st.button("🚀 Run Clustering Analysis", type="primary")
    
    with col2:
        st.info("""
        **Clustering Algorithm:** Jaccard Similarity-based Clustering
        
        This algorithm groups products based on the similarity of words in their names.
        Products with similar descriptions will be grouped together.
        
        **Similarity Threshold:** Controls how similar products need to be to group together.
        - Lower values: More products per cluster
        - Higher values: Fewer, more similar products per cluster
        - Current setting of 0.5 is good for this data
        """)
    
    # Run clustering when button is clicked
    if cluster_button:
        with st.spinner("Running clustering analysis..."):
            try:
                # Perform clustering
                clustered_df, similarity_matrix = cluster_products_jaccard(
                    df, 'Product Name', similarity_threshold=similarity_threshold
                )
                
                # Store results in session state
                st.session_state['clustered_df'] = clustered_df
                st.session_state['similarity_matrix'] = similarity_matrix
                st.session_state['clustering_done'] = True
                
                st.success("✅ Clustering completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during clustering: {e}")
    
    # Display clustering results if available
    if st.session_state.get('clustering_done', False):
        clustered_df = st.session_state['clustered_df']
        
        st.subheader("📊 Clustering Results")
        
        n_clusters = len(clustered_df['cluster'].unique())
        avg_cluster_size = len(clustered_df) / n_clusters
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Clusters", n_clusters)
        with col2:
            st.metric("Average Cluster Size", f"{avg_cluster_size:.1f}")
        with col3:
            st.metric("Similarity Threshold Used", similarity_threshold)
        
        # Cluster size distribution
        cluster_sizes = clustered_df['cluster'].value_counts().sort_index()
        
        fig_cluster_dist = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            title='Cluster Size Distribution',
            color_discrete_sequence=['#ff6b6b']
        )
        fig_cluster_dist.update_layout(
            xaxis_title="Cluster ID",
            yaxis_title="Number of Products"
        )
        st.plotly_chart(fig_cluster_dist, use_container_width=True)
        
        # Best products from each cluster
        st.subheader("🏆 Best Products from Each Cluster")
        st.info("Products are ranked by Units Sold (descending) and then Price (ascending)")
        
        best_products_df = get_best_products_per_cluster(clustered_df)
        
        # Display best products
        st.dataframe(
            best_products_df[['Product Name', 'Price', 'Units Sold', 'Location', 'cluster']],
            use_container_width=True,
            height=400
        )
        
        # Download best products
        csv = best_products_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Best Products CSV",
            data=csv,
            file_name="best_products_per_cluster.csv",
            mime="text/csv"
        )
        
        # Detailed cluster analysis
        st.subheader("🔍 Detailed Cluster Analysis")
        
        selected_cluster = st.selectbox(
            "Select a cluster to view details:",
            options=sorted(clustered_df['cluster'].unique()),
            help="Choose a cluster to see all products in that group"
        )
        
        if selected_cluster is not None:
            cluster_products = clustered_df[clustered_df['cluster'] == selected_cluster]
            
            st.write(f"**Cluster {selected_cluster}** contains {len(cluster_products)} products:")
            
            st.dataframe(
                cluster_products[['Product Name', 'Price', 'Units Sold', 'Location']],
                use_container_width=True
            )
    
    # Raw data view
    st.header("📋 Raw Data")
    
    with st.expander("View Raw Dataset", expanded=False):
        st.dataframe(df, use_container_width=True)
        
        # Download raw data
        csv_raw = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Raw Data CSV",
            data=csv_raw,
            file_name="product_data_raw.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    # Initialize session state
    if 'clustering_done' not in st.session_state:
        st.session_state['clustering_done'] = False
    
    main()