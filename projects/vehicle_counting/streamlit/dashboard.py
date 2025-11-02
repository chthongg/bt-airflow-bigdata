"""
SE363 - Vehicle Counting Dashboard
Real-time visualization với Streamlit
"""
import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ⚠️ set_page_config PHẢI là lệnh Streamlit ĐẦU TIÊN!
st.set_page_config(page_title="Vehicle Counting Dashboard", layout="wide")

# Auto-refresh mỗi 5 giây
st_autorefresh(interval=5000, key="datarefresh")

# === KẾT NỐI DATABASE ===
def get_connection():
    """Create a fresh database connection."""
    return psycopg2.connect(
        host="postgres",
        port=5432,
        database="airflow",
        user="airflow",
        password="airflow"
    )

# === ĐỌC DỮ LIỆU ===
@st.cache_data(ttl=5)
def load_data():
    """Load data from database with fresh connection."""
    conn = get_connection()
    try:
        query = """
        SELECT * FROM vehicle_counts 
        ORDER BY processed_at DESC 
        LIMIT 500
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()  # Always close connection after use

# === HEADER ===
st.title("🚗 Real-Time Vehicle Counting System")
st.markdown("---")

try:
    df = load_data()
    
    if df.empty:
        st.warning("⚠️ No data yet. Waiting for streaming data...")
        st.stop()
    
    # === METRICS ===
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Vehicles", f"{df['total_vehicles'].sum():,}")
    with col2:
        st.metric("Cars 🚗", f"{df['car_count'].sum():,}")
    with col3:
        st.metric("Buses 🚌", f"{df['bus_count'].sum():,}")
    with col4:
        st.metric("Trucks 🚚", f"{df['truck_count'].sum():,}")
    with col5:
        st.metric("Motorbikes 🏍️", f"{df['motorbike_count'].sum():,}")
    
    st.markdown("---")
    
    # === CHARTS ===
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 Vehicle Distribution")
        
        # Pie chart
        vehicle_totals = {
            'Cars': df['car_count'].sum(),
            'Buses': df['bus_count'].sum(),
            'Trucks': df['truck_count'].sum(),
            'Motorbikes': df['motorbike_count'].sum()
        }
        
        fig_pie = px.pie(
            values=list(vehicle_totals.values()),
            names=list(vehicle_totals.keys()),
            title="Vehicle Type Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        st.subheader("📈 Vehicles Over Time")
        
        # Line chart
        df_time = df.sort_values('processed_at')
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=df_time['processed_at'], 
            y=df_time['car_count'],
            name='Cars',
            mode='lines+markers'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_time['processed_at'], 
            y=df_time['bus_count'],
            name='Buses',
            mode='lines+markers'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_time['processed_at'], 
            y=df_time['truck_count'],
            name='Trucks',
            mode='lines+markers'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_time['processed_at'], 
            y=df_time['motorbike_count'],
            name='Motorbikes',
            mode='lines+markers'
        ))
        
        fig_line.update_layout(
            title="Vehicle Count Timeline",
            xaxis_title="Time",
            yaxis_title="Count"
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    
    # === BY VIDEO ===
    st.markdown("---")
    st.subheader("📹 Statistics by Video Source")
    
    video_stats = df.groupby('video_id').agg({
        'total_vehicles': 'sum',
        'car_count': 'sum',
        'bus_count': 'sum',
        'truck_count': 'sum',
        'motorbike_count': 'sum',
        'frame_number': 'count'
    }).rename(columns={'frame_number': 'frames_processed'})
    
    st.dataframe(video_stats, use_container_width=True)
    
    # === BAR CHART BY VIDEO ===
    fig_bar = px.bar(
        video_stats.reset_index(),
        x='video_id',
        y=['car_count', 'bus_count', 'truck_count', 'motorbike_count'],
        title="Vehicle Counts by Video Source",
        barmode='group'
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # === RAW DATA ===
    with st.expander("📋 View Raw Data (Last 100 records)"):
        st.dataframe(df.head(100), use_container_width=True)
    
    # === INFO ===
    st.markdown("---")
    st.info(f"📊 Total records: {len(df)} | Last update: {df['processed_at'].max()}")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)



