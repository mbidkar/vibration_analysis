# Function to calculate RMS
def calculate_rms(data):
    """Calculate Root Mean Square"""
    return np.sqrt(np.mean(np.array(data)**2))

import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import glob
import re
from scipy.signal import find_peaks

# Try to import AI analysis functions
try:
    from ai_for_real import get_time_table, create_summary
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    import_error = str(e)

# Set page config
st.set_page_config(
    page_title="Vibration Sensor Dashboard",
    layout="wide"
)

# Title
st.title("Vibration Sensor Dashboard")
st.markdown("---")

# Function to map HVT IDs to tag names
def get_sensor_tag_name(hvt_id):
    """Map HVT ID to actual tag name"""
    sensor_mapping = {
        'HONHVT100A130CB3': 'Axial Drive End',
        'HONHVT100A1308F4': 'Non-Drive End', 
        'HONHVT100A130940': 'Motor Non-Drive End',
        'HONHVT100A130C88': 'Gearbox 1 High Speed',
        'HONHVT100A130E26': 'Gearbox 1 Low Speed',
        'HONHVT100A130C04': 'Gearbox 2 High Speed',
        'HONHVT100A130C0D': 'Gearbox 2 Low Speed'
    }
    return sensor_mapping.get(hvt_id, hvt_id)

# Function to extract date and time from filename
def extract_datetime_from_filename(filename):
    """Extract date and time from filename format: HONHVT100A130C0D_20241224_153655_HMD_Merged"""
    match = re.search(r'_(\d{8})_(\d{6})_', filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        dt = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        return dt.date(), dt.time()
    return None, None

# Function to load all JSON files
@st.cache_data
def load_sensor_data():
    all_data = []
    base_path = "./Data"
    
    if not os.path.exists(base_path):
        st.error(f"Data folder not found at {base_path}")
        return []
    
    device_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    for device in device_folders:
        device_path = os.path.join(base_path, device)
        json_files = glob.glob(os.path.join(device_path, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['device'] = device
                    data['filename'] = os.path.basename(json_file)
                    
                    # Add tag name based on sensorId
                    data['tag_name'] = get_sensor_tag_name(data['sensorId'])
                    
                    file_date, file_time = extract_datetime_from_filename(data['filename'])
                    if file_date and file_time:
                        data['file_date'] = file_date
                        data['file_time'] = file_time
                        data['period'] = 'AM' if file_time.hour < 12 else 'PM'
                    else:
                        dt = datetime.fromtimestamp(data['timestamp'])
                        data['file_date'] = dt.date()
                        data['file_time'] = dt.time()
                        data['period'] = 'AM' if dt.hour < 12 else 'PM'
                    
                    all_data.append(data)
            except Exception as e:
                st.error(f"Error loading {json_file}: {str(e)}")
    
    return all_data

# Function to calculate FFT
def calculate_fft(data, sample_rate):
    """Calculate FFT and return frequencies and magnitudes"""
    n = len(data)
    fft_vals = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(n, 1/sample_rate)
    
    pos_mask = fft_freq > 0
    fft_freq = fft_freq[pos_mask]
    fft_magnitude = np.abs(fft_vals[pos_mask])
    
    return fft_freq, fft_magnitude

# Function to display AI analysis results
# Function to display AI analysis results
def display_ai_analysis(date_str, am_bool):
    """Display AI fault analysis results"""
    st.header("AI Fault Analysis")
    st.markdown("Machine learning-based fault detection and classification")
    
    # Show loading spinner while running AI analysis
    with st.spinner("Running AI fault analysis for all sensors..."):
        try:
            # Import the internal function to get full results with LLM outputs
            from ai_for_real import run_llm_for_timestamp, process_dataframe, df as feature_df
            from prompts import report_sys_prompt
            
            # Run LLM analysis and get full results including llm_output
            llm_results_df = run_llm_for_timestamp(feature_df, date=date_str, am=am_bool, sys_prompt=report_sys_prompt, n=0)
            
            if llm_results_df.empty:
                st.warning("No AI analysis results available for this date/time.")
                return
            
            # Process the results to get fault detection summary
            results_df = process_dataframe(llm_results_df)
            
            # Add tag names to results
            results_df['tag_name'] = results_df['sensorId'].apply(get_sensor_tag_name)
            
            # Display summary metrics
            st.subheader("Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sensors = len(results_df)
                st.metric("Total Sensors Analyzed", total_sensors)
            
            with col2:
                faults_detected = len(results_df[results_df['fault_status'] == 'DETECTED'])
                st.metric("Faults Detected", faults_detected)
            
            with col3:
                avg_confidence = results_df['confidence_percentage'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.1f}%")
            
            with col4:
                high_confidence = len(results_df[results_df['confidence_percentage'] >= 80])
                st.metric("High Confidence (≥80%)", high_confidence)

            # Executive Summary Section
            st.subheader("Executive Summary")
            st.markdown("System-wide analysis across all sensors")
            
            # Generate executive summary using the full LLM outputs
            with st.spinner("Generating executive summary..."):
                try:
                    if not llm_results_df.empty and 'llm_output' in llm_results_df.columns:
                        summary_text = create_summary(llm_results_df)
                        
                        # Display the summary in an expandable container
                        with st.expander("View Executive Summary", expanded=True):
                            st.markdown(summary_text)
                    else:
                        st.info("No detailed analysis data available for summary generation.")
                        
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
            
            # Display detailed results table
            st.subheader("Detailed Fault Analysis Results")
            
            # Create a styled dataframe
            def style_fault_status(val):
                if val == 'DETECTED':
                    return 'background-color: #ff4b4b; color: white'
                elif val == 'NOT DETECTED':
                    return 'background-color: #21c354; color: white'
                else:
                    return 'background-color: #ffa500; color: white'
            
            def style_confidence(val):
                if val >= 80:
                    return 'background-color: #21c354; color: white'
                elif val >= 50:
                    return 'background-color: #ffa500; color: white'
                else:
                    return 'background-color: #ff4b4b; color: white'
            
            # Reorder columns for better display
            display_df = results_df[['tag_name', 'fault_status', 'primary_type', 'confidence_percentage']].copy()
            display_df.columns = ['Sensor Location', 'Fault Status', 'Primary Fault Type', 'Confidence (%)']
            
            # Apply styling
            styled_df = display_df.style.applymap(
                style_fault_status, subset=['Fault Status']
            ).applymap(
                style_confidence, subset=['Confidence (%)']
            )
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Fault distribution pie chart
            st.subheader("Fault Type Distribution")
            fault_counts = results_df[results_df['fault_status'] == 'DETECTED']['primary_type'].value_counts()
            
            if not fault_counts.empty:
                fig_pie = px.pie(
                    values=fault_counts.values,
                    names=fault_counts.index,
                    title="Distribution of Detected Fault Types",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No faults detected to show distribution.")
            
            # Critical sensors alert
            critical_sensors = results_df[
                (results_df['fault_status'] == 'DETECTED') & 
                (results_df['confidence_percentage'] >= 70)
            ].sort_values('confidence_percentage', ascending=False)
            
            if not critical_sensors.empty:
                st.subheader("⚠️ Critical Sensors Requiring Attention")
                st.markdown("Sensors with detected faults and high confidence:")
                for _, sensor in critical_sensors.iterrows():
                    st.error(f"**{sensor['tag_name']}**: {sensor['primary_type']} (Confidence: {sensor['confidence_percentage']}%)")
            
        except Exception as e:
            st.error(f"Error running AI analysis: {str(e)}")
            st.info("Please ensure you have the required dependencies and credentials for Google Cloud services.")
            
# Load data automatically on startup
if 'data_loaded' not in st.session_state:
    with st.spinner('Loading sensor data...'):
        sensor_data = load_sensor_data()
        if sensor_data:
            st.session_state['data_loaded'] = True
            st.session_state['sensor_data'] = sensor_data
            st.success(f"Loaded {len(sensor_data)} recordings from {len(set([d['device'] for d in sensor_data]))} devices")
        else:
            st.error("No data found in the Data folder.")

# Main dashboard
if 'sensor_data' in st.session_state and len(st.session_state['sensor_data']) > 0:
    sensor_data = st.session_state['sensor_data']
    
    # Sidebar for date and time period selection
    st.sidebar.header("Date and Time Selection")
    
    # Get unique dates
    unique_dates = sorted(list(set([d['file_date'] for d in sensor_data])))
    
    # Date selector
    selected_date = st.sidebar.selectbox(
        "Select Date:",
        unique_dates,
        format_func=lambda x: x.strftime('%m/%d/%Y')
    )
    
    # Filter data by selected date
    date_filtered_data = [d for d in sensor_data if d['file_date'] == selected_date]
    
    # Check if there are AM/PM recordings for this date
    periods_available = sorted(list(set([d['period'] for d in date_filtered_data])))
    
    # If multiple periods available, show period selector
    if len(periods_available) > 1:
        selected_period = st.sidebar.selectbox(
            "Select Time Period:",
            periods_available
        )
        # Further filter by period
        date_filtered_data = [d for d in date_filtered_data if d['period'] == selected_period]
    elif len(periods_available) == 1:
        selected_period = periods_available[0]
    else:
        selected_period = "N/A"
    
    # Main content area
    # Get sensors available for selected date and period - use tag names for display
    available_sensors_data = [(d['tag_name'], d['sensorId']) for d in date_filtered_data]
    available_sensors_data = sorted(list(set(available_sensors_data)))
    available_sensor_names = [tag_name for tag_name, _ in available_sensors_data]
    
    # Create main layout columns
    sensor_col, tabs_col = st.columns([1, 3])
    
    with sensor_col:
        st.header("Sensor Selection")
        
        # Sensor selector dropdown - display tag names
        selected_sensor_name = st.selectbox(
            "Select Sensor:",
            available_sensor_names,
            key="main_sensor_select"
        )
        
        # Get the actual sensor ID from the tag name
        selected_sensor = next(sensor_id for tag_name, sensor_id in available_sensors_data if tag_name == selected_sensor_name)
        
        # Get specific recording for selected date, period, and sensor
        sensor_recording = next((d for d in date_filtered_data if d['sensorId'] == selected_sensor), None)
        
        if sensor_recording:
            # Display selection info
            period_display = f" ({selected_period})" if len(periods_available) > 1 else ""
            st.subheader(f"Selected: {selected_date.strftime('%m/%d/%Y')}{period_display}")
            st.caption(f"HVT ID: {selected_sensor}")
            
            # Basic info metrics
            st.markdown("#### Sensor Metrics")
            st.metric("Sample Rate", f"{sensor_recording['vibSamplefrequency']} Hz")
            st.metric("Temperature", f"{sensor_recording['temperature']}°")
            st.metric("Humidity", f"{sensor_recording['humidity']}%")
            st.metric("Data Points", len(sensor_recording['vibration']['x']))
    
    with tabs_col:
        if sensor_recording:
            # Create tabs for analysis - Add AI Analysis tab if available
            if AI_AVAILABLE:
                tab1, tab2, tab3, tab4 = st.tabs(["FFT Analysis", "RMS Analysis", "Time Series", "AI Analysis"])
            else:
                tab1, tab2, tab3 = st.tabs(["FFT Analysis", "RMS Analysis", "Time Series"])
                if not AI_AVAILABLE:
                    st.sidebar.warning(f"AI Analysis not available: {import_error}")
            
            with tab1:
                st.header("FFT Analysis")
                st.markdown("Frequency domain analysis of vibration data")
                
                # Calculate FFT for each axis
                sample_rate = sensor_recording['vibSamplefrequency']
                freq_x, mag_x = calculate_fft(sensor_recording['vibration']['x'], sample_rate)
                freq_y, mag_y = calculate_fft(sensor_recording['vibration']['y'], sample_rate)
                freq_z, mag_z = calculate_fft(sensor_recording['vibration']['z'], sample_rate)
                
                # Create layout with main plot and buttons
                plot_col, button_col = st.columns([5, 1])
                
                with button_col:
                    st.markdown("#### Select Axes")
                    
                    # Initialize FFT axis selections if not exist
                    if 'fft_x_selected' not in st.session_state:
                        st.session_state['fft_x_selected'] = False
                    if 'fft_y_selected' not in st.session_state:
                        st.session_state['fft_y_selected'] = False
                    if 'fft_z_selected' not in st.session_state:
                        st.session_state['fft_z_selected'] = False
                    
                    # Checkboxes for individual axes
                    fft_x_check = st.checkbox("X-axis", value=st.session_state['fft_x_selected'], key="fft_x_check")
                    fft_y_check = st.checkbox("Y-axis", value=st.session_state['fft_y_selected'], key="fft_y_check")
                    fft_z_check = st.checkbox("Z-axis", value=st.session_state['fft_z_selected'], key="fft_z_check")
                    
                    # Update session state based on checkbox values
                    st.session_state['fft_x_selected'] = fft_x_check
                    st.session_state['fft_y_selected'] = fft_y_check
                    st.session_state['fft_z_selected'] = fft_z_check
                
                with plot_col:
                    # Create the FFT plot based on selected axes
                    fig_fft = go.Figure()
                    
                    selected_axes = []
                    if st.session_state['fft_x_selected']:
                        selected_axes.append('X')
                    if st.session_state['fft_y_selected']:
                        selected_axes.append('Y')
                    if st.session_state['fft_z_selected']:
                        selected_axes.append('Z')
                    
                    # If no axes selected, default to combined view
                    if not selected_axes:
                        selected_axes = ['X', 'Y', 'Z']
                    
                    if 'X' in selected_axes:
                        fig_fft.add_trace(go.Scatter(x=freq_x[:len(freq_x)//2], y=mag_x[:len(mag_x)//2], 
                                                   mode='lines', name='X-axis', line=dict(color='blue', width=2)))
                    if 'Y' in selected_axes:
                        fig_fft.add_trace(go.Scatter(x=freq_y[:len(freq_y)//2], y=mag_y[:len(mag_y)//2], 
                                                   mode='lines', name='Y-axis', line=dict(color='green', width=2)))
                    if 'Z' in selected_axes:
                        fig_fft.add_trace(go.Scatter(x=freq_z[:len(freq_z)//2], y=mag_z[:len(mag_z)//2], 
                                                   mode='lines', name='Z-axis', line=dict(color='red', width=2)))
                    
                    # Set title based on selected axes
                    if len(selected_axes) == 1:
                        title = f"FFT Analysis - {selected_axes[0]}-axis - {selected_sensor_name}"
                    elif len(selected_axes) == 2:
                        title = f"FFT Analysis - {' & '.join(selected_axes)} axes - {selected_sensor_name}"
                    else:
                        title = f"FFT Analysis - Combined View - {selected_sensor_name}"
                    
                    fig_fft.update_layout(
                        title=title,
                        xaxis_title="Frequency (Hz)",
                        yaxis_title="Magnitude",
                        height=600,
                        hovermode='x unified',
                        showlegend=True
                    )
                    st.plotly_chart(fig_fft, use_container_width=True)
                
                # FFT Peak Trend Analysis
                st.subheader("FFT Peak Trend Analysis")
                st.markdown("Tracking maximum peaks around critical frequencies: 633 Hz, 1266 Hz (2×633), and 1899 Hz (3×633)")
                
                # Get historical data for the same sensor
                current_datetime = datetime.combine(selected_date, sensor_recording['file_time'])
                historical_fft_data = sorted([d for d in sensor_data 
                                            if d['sensorId'] == selected_sensor 
                                            and datetime.combine(d['file_date'], d['file_time']) <= current_datetime], 
                                           key=lambda x: (x['file_date'], x['file_time']))
                
                if len(historical_fft_data) > 1:
                    # Define target frequencies and tolerance
                    target_frequencies = [633, 1266, 1899]  # 633 Hz, 2×633, 3×633
                    tolerance_hz = 10.0  # ±10 Hz tolerance
                    
                    # Calculate FFT peaks for all historical data
                    fft_peak_history = []
                    for recording in historical_fft_data:
                        # Calculate FFT for each axis
                        freq_hist_x, mag_hist_x = calculate_fft(recording['vibration']['x'], recording['vibSamplefrequency'])
                        freq_hist_y, mag_hist_y = calculate_fft(recording['vibration']['y'], recording['vibSamplefrequency'])
                        freq_hist_z, mag_hist_z = calculate_fft(recording['vibration']['z'], recording['vibSamplefrequency'])
                        
                        # Find peak magnitudes around target frequencies for each axis
                        peaks_x = {}
                        peaks_y = {}
                        peaks_z = {}
                        
                        for target_freq in target_frequencies:
                            # Find peaks for X-axis
                            mask_x = np.abs(freq_hist_x - target_freq) <= tolerance_hz
                            if np.any(mask_x):
                                peaks_x[f'{target_freq}Hz'] = np.max(mag_hist_x[mask_x])
                            else:
                                peaks_x[f'{target_freq}Hz'] = 0
                            
                            # Find peaks for Y-axis
                            mask_y = np.abs(freq_hist_y - target_freq) <= tolerance_hz
                            if np.any(mask_y):
                                peaks_y[f'{target_freq}Hz'] = np.max(mag_hist_y[mask_y])
                            else:
                                peaks_y[f'{target_freq}Hz'] = 0
                            
                            # Find peaks for Z-axis
                            mask_z = np.abs(freq_hist_z - target_freq) <= tolerance_hz
                            if np.any(mask_z):
                                peaks_z[f'{target_freq}Hz'] = np.max(mag_hist_z[mask_z])
                            else:
                                peaks_z[f'{target_freq}Hz'] = 0
                        
                        fft_peak_history.append({
                            'datetime': datetime.combine(recording['file_date'], recording['file_time']),
                            'date_str': f"{recording['file_date'].strftime('%m/%d')} {recording['period']}",
                            'is_current': (recording['file_date'] == selected_date and recording['period'] == selected_period),
                            'x_peaks': peaks_x,
                            'y_peaks': peaks_y,
                            'z_peaks': peaks_z
                        })
                    
                    # Create peak trend plots
                    fft_df = pd.DataFrame(fft_peak_history).sort_values('datetime')
                    
                    # Create tabs for each target frequency
                    freq_tab1, freq_tab2, freq_tab3 = st.tabs(["633 Hz", "1266 Hz (2×633)", "1899 Hz (3×633)"])
                    
                    with freq_tab1:
                        fig_633 = go.Figure()
                        
                        # Get selected axes for peak tracking
                        selected_axes = []
                        if st.session_state['fft_x_selected']:
                            selected_axes.append('X')
                        if st.session_state['fft_y_selected']:
                            selected_axes.append('Y')
                        if st.session_state['fft_z_selected']:
                            selected_axes.append('Z')
                        
                        if not selected_axes:
                            selected_axes = ['X', 'Y', 'Z']
                        
                        if 'X' in selected_axes:
                            x_values = [row['x_peaks']['633Hz'] for row in fft_peak_history]
                            fig_633.add_trace(go.Scatter(
                                x=fft_df['date_str'], y=x_values,
                                mode='lines+markers', name='X-axis Peak',
                                line=dict(color='blue', width=2), marker=dict(size=6)
                            ))
                            # Highlight current point
                            current_point = fft_df[fft_df['is_current']]
                            if not current_point.empty:
                                current_idx = current_point.index[0]
                                fig_633.add_trace(go.Scatter(
                                    x=[current_point['date_str'].iloc[0]], y=[fft_peak_history[current_idx]['x_peaks']['633Hz']],
                                    mode='markers', name='X Current',
                                    marker=dict(size=15, color='blue', symbol='star'),
                                    showlegend=False
                                ))
                        
                        if 'Y' in selected_axes:
                            y_values = [row['y_peaks']['633Hz'] for row in fft_peak_history]
                            fig_633.add_trace(go.Scatter(
                                x=fft_df['date_str'], y=y_values,
                                mode='lines+markers', name='Y-axis Peak',
                                line=dict(color='green', width=2), marker=dict(size=6)
                            ))
                            current_point = fft_df[fft_df['is_current']]
                            if not current_point.empty:
                                current_idx = current_point.index[0]
                                fig_633.add_trace(go.Scatter(
                                    x=[current_point['date_str'].iloc[0]], y=[fft_peak_history[current_idx]['y_peaks']['633Hz']],
                                    mode='markers', name='Y Current',
                                    marker=dict(size=15, color='green', symbol='star'),
                                    showlegend=False
                                ))
                        
                        if 'Z' in selected_axes:
                            z_values = [row['z_peaks']['633Hz'] for row in fft_peak_history]
                            fig_633.add_trace(go.Scatter(
                                x=fft_df['date_str'], y=z_values,
                                mode='lines+markers', name='Z-axis Peak',
                                line=dict(color='red', width=2), marker=dict(size=6)
                            ))
                            current_point = fft_df[fft_df['is_current']]
                            if not current_point.empty:
                                current_idx = current_point.index[0]
                                fig_633.add_trace(go.Scatter(
                                    x=[current_point['date_str'].iloc[0]], y=[fft_peak_history[current_idx]['z_peaks']['633Hz']],
                                    mode='markers', name='Z Current',
                                    marker=dict(size=15, color='red', symbol='star'),
                                    showlegend=False
                                ))
                        
                        fig_633.update_layout(
                            title=f"FFT Peak Trend at 633 Hz (±10 Hz) - {selected_sensor_name}",
                            xaxis_title="Date/Time",
                            yaxis_title="Peak Magnitude",
                            height=400,
                            hovermode='x unified',
                            showlegend=True
                        )
                        st.plotly_chart(fig_633, use_container_width=True)
                    
                    with freq_tab2:
                        fig_1266 = go.Figure()
                        
                        if 'X' in selected_axes:
                            x_values = [row['x_peaks']['1266Hz'] for row in fft_peak_history]
                            fig_1266.add_trace(go.Scatter(
                                x=fft_df['date_str'], y=x_values,
                                mode='lines+markers', name='X-axis Peak',
                                line=dict(color='blue', width=2), marker=dict(size=6)
                            ))
                            current_point = fft_df[fft_df['is_current']]
                            if not current_point.empty:
                                current_idx = current_point.index[0]
                                fig_1266.add_trace(go.Scatter(
                                    x=[current_point['date_str'].iloc[0]], y=[fft_peak_history[current_idx]['x_peaks']['1266Hz']],
                                    mode='markers', name='X Current',
                                    marker=dict(size=15, color='blue', symbol='star'),
                                    showlegend=False
                                ))
                        
                        if 'Y' in selected_axes:
                            y_values = [row['y_peaks']['1266Hz'] for row in fft_peak_history]
                            fig_1266.add_trace(go.Scatter(
                                x=fft_df['date_str'], y=y_values,
                                mode='lines+markers', name='Y-axis Peak',
                                line=dict(color='green', width=2), marker=dict(size=6)
                            ))
                            current_point = fft_df[fft_df['is_current']]
                            if not current_point.empty:
                                current_idx = current_point.index[0]
                                fig_1266.add_trace(go.Scatter(
                                    x=[current_point['date_str'].iloc[0]], y=[fft_peak_history[current_idx]['y_peaks']['1266Hz']],
                                    mode='markers', name='Y Current',
                                    marker=dict(size=15, color='green', symbol='star'),
                                    showlegend=False
                                ))
                        
                        if 'Z' in selected_axes:
                            z_values = [row['z_peaks']['1266Hz'] for row in fft_peak_history]
                            fig_1266.add_trace(go.Scatter(
                                x=fft_df['date_str'], y=z_values,
                                mode='lines+markers', name='Z-axis Peak',
                                line=dict(color='red', width=2), marker=dict(size=6)
                            ))
                            current_point = fft_df[fft_df['is_current']]
                            if not current_point.empty:
                                current_idx = current_point.index[0]
                                fig_1266.add_trace(go.Scatter(
                                    x=[current_point['date_str'].iloc[0]], y=[fft_peak_history[current_idx]['z_peaks']['1266Hz']],
                                    mode='markers', name='Z Current',
                                    marker=dict(size=15, color='red', symbol='star'),
                                    showlegend=False
                                ))
                        
                        fig_1266.update_layout(
                            title=f"FFT Peak Trend at 1266 Hz (2×633 Hz, ±10 Hz) - {selected_sensor_name}",
                            xaxis_title="Date/Time",
                            yaxis_title="Peak Magnitude",
                            height=400,
                            hovermode='x unified',
                            showlegend=True
                        )
                        st.plotly_chart(fig_1266, use_container_width=True)
                    
                    with freq_tab3:
                        fig_1899 = go.Figure()
                        
                        if 'X' in selected_axes:
                            x_values = [row['x_peaks']['1899Hz'] for row in fft_peak_history]
                            fig_1899.add_trace(go.Scatter(
                                x=fft_df['date_str'], y=x_values,
                                mode='lines+markers', name='X-axis Peak',
                                line=dict(color='blue', width=2), marker=dict(size=6)
                            ))
                            current_point = fft_df[fft_df['is_current']]
                            if not current_point.empty:
                                current_idx = current_point.index[0]
                                fig_1899.add_trace(go.Scatter(
                                    x=[current_point['date_str'].iloc[0]], y=[fft_peak_history[current_idx]['x_peaks']['1899Hz']],
                                    mode='markers', name='X Current',
                                    marker=dict(size=15, color='blue', symbol='star'),
                                    showlegend=False
                                ))
                        
                        if 'Y' in selected_axes:
                            y_values = [row['y_peaks']['1899Hz'] for row in fft_peak_history]
                            fig_1899.add_trace(go.Scatter(
                                x=fft_df['date_str'], y=y_values,
                                mode='lines+markers', name='Y-axis Peak',
                                line=dict(color='green', width=2), marker=dict(size=6)
                            ))
                            current_point = fft_df[fft_df['is_current']]
                            if not current_point.empty:
                                current_idx = current_point.index[0]
                                fig_1899.add_trace(go.Scatter(
                                    x=[current_point['date_str'].iloc[0]], y=[fft_peak_history[current_idx]['y_peaks']['1899Hz']],
                                    mode='markers', name='Y Current',
                                    marker=dict(size=15, color='green', symbol='star'),
                                    showlegend=False
                                ))
                        
                        if 'Z' in selected_axes:
                            z_values = [row['z_peaks']['1899Hz'] for row in fft_peak_history]
                            fig_1899.add_trace(go.Scatter(
                                x=fft_df['date_str'], y=z_values,
                                mode='lines+markers', name='Z-axis Peak',
                                line=dict(color='red', width=2), marker=dict(size=6)
                            ))
                            current_point = fft_df[fft_df['is_current']]
                            if not current_point.empty:
                                current_idx = current_point.index[0]
                                fig_1899.add_trace(go.Scatter(
                                    x=[current_point['date_str'].iloc[0]], y=[fft_peak_history[current_idx]['z_peaks']['1899Hz']],
                                    mode='markers', name='Z Current',
                                    marker=dict(size=15, color='red', symbol='star'),
                                    showlegend=False
                                ))
                        
                        fig_1899.update_layout(
                            title=f"FFT Peak Trend at 1899 Hz (3×633 Hz, ±10 Hz) - {selected_sensor_name}",
                            xaxis_title="Date/Time",
                            yaxis_title="Peak Magnitude",
                            height=400,
                            hovermode='x unified',
                            showlegend=True
                        )
                        st.plotly_chart(fig_1899, use_container_width=True)
                
                else:
                    st.info("Need at least 2 recordings to show FFT peak trends over time.")
            
            with tab2:
                st.header("RMS (Root Mean Square) Analysis")
                st.markdown("RMS values indicate the overall vibration energy level")
                
                # Calculate RMS for current recording
                rms_x = calculate_rms(sensor_recording['vibration']['x'])
                rms_y = calculate_rms(sensor_recording['vibration']['y'])
                rms_z = calculate_rms(sensor_recording['vibration']['z'])
                rms_total = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)
                
                # Display current RMS metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("X-axis RMS", f"{rms_x:.4f}")
                with col2:
                    st.metric("Y-axis RMS", f"{rms_y:.4f}")
                with col3:
                    st.metric("Z-axis RMS", f"{rms_z:.4f}")
                with col4:
                    st.metric("Total RMS", f"{rms_total:.4f}")
                
                # Get historical data for the same sensor
                current_datetime = datetime.combine(selected_date, sensor_recording['file_time'])
                historical_data = sorted([d for d in sensor_data 
                                        if d['sensorId'] == selected_sensor 
                                        and datetime.combine(d['file_date'], d['file_time']) <= current_datetime], 
                                       key=lambda x: (x['file_date'], x['file_time']))
                
                if len(historical_data) > 1:
                    # Calculate RMS for all historical data
                    rms_history = []
                    for recording in historical_data:
                        rms_hist_x = calculate_rms(recording['vibration']['x'])
                        rms_hist_y = calculate_rms(recording['vibration']['y'])
                        rms_hist_z = calculate_rms(recording['vibration']['z'])
                        rms_hist_total = np.sqrt(rms_hist_x**2 + rms_hist_y**2 + rms_hist_z**2)
                        
                        rms_history.append({
                            'datetime': datetime.combine(recording['file_date'], recording['file_time']),
                            'date_str': f"{recording['file_date'].strftime('%m/%d')} {recording['period']}",
                            'rms_x': rms_hist_x,
                            'rms_y': rms_hist_y,
                            'rms_z': rms_hist_z,
                            'rms_total': rms_hist_total,
                            'is_current': (recording['file_date'] == selected_date and recording['period'] == selected_period)
                        })
                    
                    # Create layout with plot and buttons
                    st.subheader("RMS Historical Trend")
                    plot_col, button_col = st.columns([5, 1])
                    
                    with button_col:
                        st.markdown("#### Select Axes")
                        
                        # Initialize RMS axis selections if not exist
                        if 'rms_x_selected' not in st.session_state:
                            st.session_state['rms_x_selected'] = False
                        if 'rms_y_selected' not in st.session_state:
                            st.session_state['rms_y_selected'] = False
                        if 'rms_z_selected' not in st.session_state:
                            st.session_state['rms_z_selected'] = False
                        
                        # Checkboxes for individual axes
                        rms_x_check = st.checkbox("X-axis", value=st.session_state['rms_x_selected'], key="rms_x_check")
                        rms_y_check = st.checkbox("Y-axis", value=st.session_state['rms_y_selected'], key="rms_y_check")
                        rms_z_check = st.checkbox("Z-axis", value=st.session_state['rms_z_selected'], key="rms_z_check")
                        
                        # Update session state based on checkbox values
                        st.session_state['rms_x_selected'] = rms_x_check
                        st.session_state['rms_y_selected'] = rms_y_check
                        st.session_state['rms_z_selected'] = rms_z_check
                    
                    with plot_col:
                        fig_rms = go.Figure()
                        
                        # Sort by datetime for proper plotting
                        rms_df = pd.DataFrame(rms_history).sort_values('datetime')
                        
                        selected_axes = []
                        if st.session_state['rms_x_selected']:
                            selected_axes.append('X')
                        if st.session_state['rms_y_selected']:
                            selected_axes.append('Y')
                        if st.session_state['rms_z_selected']:
                            selected_axes.append('Z')
                        
                        # If no axes selected, default to combined view
                        if not selected_axes:
                            selected_axes = ['X', 'Y', 'Z']
                        
                        if 'X' in selected_axes:
                            fig_rms.add_trace(go.Scatter(
                                x=rms_df['date_str'], y=rms_df['rms_x'],
                                mode='lines+markers', name='X-axis',
                                line=dict(color='blue', width=2), marker=dict(size=6)
                            ))
                            # Highlight current point
                            current_point = rms_df[rms_df['is_current']]
                            if not current_point.empty:
                                fig_rms.add_trace(go.Scatter(
                                    x=current_point['date_str'], y=current_point['rms_x'],
                                    mode='markers', name='X Current',
                                    marker=dict(size=15, color='blue', symbol='star'),
                                    showlegend=False
                                ))
                        
                        if 'Y' in selected_axes:
                            fig_rms.add_trace(go.Scatter(
                                x=rms_df['date_str'], y=rms_df['rms_y'],
                                mode='lines+markers', name='Y-axis',
                                line=dict(color='green', width=2), marker=dict(size=6)
                            ))
                            current_point = rms_df[rms_df['is_current']]
                            if not current_point.empty:
                                fig_rms.add_trace(go.Scatter(
                                    x=current_point['date_str'], y=current_point['rms_y'],
                                    mode='markers', name='Y Current',
                                    marker=dict(size=15, color='green', symbol='star'),
                                    showlegend=False
                                ))
                        
                        if 'Z' in selected_axes:
                            fig_rms.add_trace(go.Scatter(
                                x=rms_df['date_str'], y=rms_df['rms_z'],
                                mode='lines+markers', name='Z-axis',
                                line=dict(color='red', width=2), marker=dict(size=6)
                            ))
                            current_point = rms_df[rms_df['is_current']]
                            if not current_point.empty:
                                fig_rms.add_trace(go.Scatter(
                                    x=current_point['date_str'], y=current_point['rms_z'],
                                    mode='markers', name='Z Current',
                                    marker=dict(size=15, color='red', symbol='star'),
                                    showlegend=False
                                ))
                        
                        # Always show Total RMS when combined or all axes selected
                        if len(selected_axes) == 3:
                            fig_rms.add_trace(go.Scatter(
                                x=rms_df['date_str'], y=rms_df['rms_total'],
                                mode='lines+markers', name='Total',
                                line=dict(color='purple', width=3), marker=dict(size=8)
                            ))
                            current_point = rms_df[rms_df['is_current']]
                            if not current_point.empty:
                                fig_rms.add_trace(go.Scatter(
                                    x=current_point['date_str'], y=current_point['rms_total'],
                                    mode='markers', name='Total Current',
                                    marker=dict(size=15, color='purple', symbol='star'),
                                    showlegend=False
                                ))
                        
                        # Set title based on selected axes
                        if len(selected_axes) == 1:
                            title = f"RMS Historical Trend - {selected_axes[0]}-axis - {selected_sensor_name}"
                        elif len(selected_axes) == 2:
                            title = f"RMS Historical Trend - {' & '.join(selected_axes)} axes - {selected_sensor_name}"
                        else:
                            title = f"RMS Historical Trend - All Axes - {selected_sensor_name}"
                        
                        fig_rms.update_layout(
                            title=title,
                            xaxis_title="Date/Time",
                            yaxis_title="RMS Value",
                            height=500,
                            hovermode='x unified',
                            showlegend=True
                        )
                        st.plotly_chart(fig_rms, use_container_width=True)
                
                else:
                    # Only one recording available, show bar chart
                    st.subheader("RMS Values")
                    rms_data = pd.DataFrame({
                        'Axis': ['X', 'Y', 'Z', 'Total'],
                        'RMS Value': [rms_x, rms_y, rms_z, rms_total]
                    })
                    
                    fig_rms = px.bar(rms_data, x='Axis', y='RMS Value', 
                                    title=f"RMS Values by Axis - {selected_sensor_name}",
                                    color='Axis',
                                    color_discrete_sequence=['blue', 'green', 'red', 'purple'])
                    fig_rms.update_layout(height=400)
                    st.plotly_chart(fig_rms, use_container_width=True)
                
                # Calculate RMS over windows
                st.subheader("RMS Over Time Windows")
                window_size = st.slider("Window Size (samples)", 100, 1000, 500)
                
                # Calculate windowed RMS
                x_data = np.array(sensor_recording['vibration']['x'])
                y_data = np.array(sensor_recording['vibration']['y'])
                z_data = np.array(sensor_recording['vibration']['z'])
                
                num_windows = len(x_data) // window_size
                window_times = []
                window_rms_x = []
                window_rms_y = []
                window_rms_z = []
                
                for i in range(num_windows):
                    start_idx = i * window_size
                    end_idx = (i + 1) * window_size
                    
                    window_times.append(i * window_size / sample_rate)
                    window_rms_x.append(calculate_rms(x_data[start_idx:end_idx]))
                    window_rms_y.append(calculate_rms(y_data[start_idx:end_idx]))
                    window_rms_z.append(calculate_rms(z_data[start_idx:end_idx]))
                
                # Plot windowed RMS
                fig_windowed = go.Figure()
                fig_windowed.add_trace(go.Scatter(x=window_times, y=window_rms_x, 
                                                mode='lines+markers', name='X-axis RMS'))
                fig_windowed.add_trace(go.Scatter(x=window_times, y=window_rms_y, 
                                                mode='lines+markers', name='Y-axis RMS'))
                fig_windowed.add_trace(go.Scatter(x=window_times, y=window_rms_z, 
                                                mode='lines+markers', name='Z-axis RMS'))
                fig_windowed.update_layout(
                    title=f"RMS Evolution (Window Size: {window_size} samples) - {selected_sensor_name}",
                    xaxis_title="Time (seconds)",
                    yaxis_title="RMS Value",
                    height=500
                )
                st.plotly_chart(fig_windowed, use_container_width=True)
            
            with tab3:
                st.header("Time Series Data")
                
                # Create time array
                time_array = np.arange(len(sensor_recording['vibration']['x'])) / sample_rate
                
                # Create layout with plot and buttons
                plot_col, button_col = st.columns([5, 1])
                
                with button_col:
                    st.markdown("#### Select Axes")
                    
                    # Initialize Time Series axis selections if not exist
                    if 'ts_x_selected' not in st.session_state:
                        st.session_state['ts_x_selected'] = False
                    if 'ts_y_selected' not in st.session_state:
                        st.session_state['ts_y_selected'] = False
                    if 'ts_z_selected' not in st.session_state:
                        st.session_state['ts_z_selected'] = False
                    
                    # Checkboxes for individual axes
                    ts_x_check = st.checkbox("X-axis", value=st.session_state['ts_x_selected'], key="ts_x_check")
                    ts_y_check = st.checkbox("Y-axis", value=st.session_state['ts_y_selected'], key="ts_y_check")
                    ts_z_check = st.checkbox("Z-axis", value=st.session_state['ts_z_selected'], key="ts_z_check")
                    
                    # Update session state based on checkbox values
                    st.session_state['ts_x_selected'] = ts_x_check
                    st.session_state['ts_y_selected'] = ts_y_check
                    st.session_state['ts_z_selected'] = ts_z_check
                
                with plot_col:
                    # Plot time series based on selected axes
                    fig_time = go.Figure()
                    
                    selected_axes = []
                    if st.session_state['ts_x_selected']:
                        selected_axes.append('X')
                    if st.session_state['ts_y_selected']:
                        selected_axes.append('Y')
                    if st.session_state['ts_z_selected']:
                        selected_axes.append('Z')
                    
                    # If no axes selected, default to combined view
                    if not selected_axes:
                        selected_axes = ['X', 'Y', 'Z']
                    
                    if 'X' in selected_axes:
                        fig_time.add_trace(go.Scatter(x=time_array, y=sensor_recording['vibration']['x'], 
                                                    name='X-axis', line=dict(color='blue', width=1)))
                    if 'Y' in selected_axes:
                        fig_time.add_trace(go.Scatter(x=time_array, y=sensor_recording['vibration']['y'], 
                                                    name='Y-axis', line=dict(color='green', width=1)))
                    if 'Z' in selected_axes:
                        fig_time.add_trace(go.Scatter(x=time_array, y=sensor_recording['vibration']['z'], 
                                                    name='Z-axis', line=dict(color='red', width=1)))
                    
                    # Set title based on selected axes
                    if len(selected_axes) == 1:
                        title = f"Vibration Time Series - {selected_axes[0]}-axis - {selected_sensor_name}"
                    elif len(selected_axes) == 2:
                        title = f"Vibration Time Series - {' & '.join(selected_axes)} axes - {selected_sensor_name}"
                    else:
                        title = f"Vibration Time Series - All Axes - {selected_sensor_name}"
                    
                    fig_time.update_layout(
                        title=title,
                        xaxis_title="Time (seconds)",
                        yaxis_title="Acceleration",
                        height=600,
                        hovermode='x unified',
                        showlegend=True if len(selected_axes) > 1 else False
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                
                # Statistics
                st.subheader("Signal Statistics")
                
                # Get selected axes for statistics
                selected_axes = []
                if st.session_state.get('ts_x_selected', False):
                    selected_axes.append('X')
                if st.session_state.get('ts_y_selected', False):
                    selected_axes.append('Y')
                if st.session_state.get('ts_z_selected', False):
                    selected_axes.append('Z')
                
                # If no axes selected, default to all
                if not selected_axes:
                    selected_axes = ['X', 'Y', 'Z']
                
                # Show statistics based on selected axes
                if len(selected_axes) == 1:
                    axis = selected_axes[0].lower()
                    data = sensor_recording['vibration'][axis]
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Mean", f"{np.mean(data):.4f}")
                    with col2:
                        st.metric("Std Dev", f"{np.std(data):.4f}")
                    with col3:
                        st.metric("Min", f"{np.min(data):.4f}")
                    with col4:
                        st.metric("Max", f"{np.max(data):.4f}")
                    with col5:
                        st.metric("Peak-to-Peak", f"{np.max(data) - np.min(data):.4f}")
                else:
                    # Show statistics in table format for multiple axes
                    stats_data = {'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Peak-to-Peak']}
                    
                    if 'X' in selected_axes:
                        stats_data['X-axis'] = [
                            np.mean(sensor_recording['vibration']['x']),
                            np.std(sensor_recording['vibration']['x']),
                            np.min(sensor_recording['vibration']['x']),
                            np.max(sensor_recording['vibration']['x']),
                            np.max(sensor_recording['vibration']['x']) - np.min(sensor_recording['vibration']['x'])
                        ]
                    if 'Y' in selected_axes:
                        stats_data['Y-axis'] = [
                            np.mean(sensor_recording['vibration']['y']),
                            np.std(sensor_recording['vibration']['y']),
                            np.min(sensor_recording['vibration']['y']),
                            np.max(sensor_recording['vibration']['y']),
                            np.max(sensor_recording['vibration']['y']) - np.min(sensor_recording['vibration']['y'])
                        ]
                    if 'Z' in selected_axes:
                        stats_data['Z-axis'] = [
                            np.mean(sensor_recording['vibration']['z']),
                            np.std(sensor_recording['vibration']['z']),
                            np.min(sensor_recording['vibration']['z']),
                            np.max(sensor_recording['vibration']['z']),
                            np.max(sensor_recording['vibration']['z']) - np.min(sensor_recording['vibration']['z'])
                        ]
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df.round(4))
            
            # AI Analysis Tab (only if AI is available)
            if AI_AVAILABLE:
                with tab4:
                    # Convert date to string format and determine AM/PM boolean
                    date_str = selected_date.strftime('%Y-%m-%d')
                    am_bool = True if selected_period == 'AM' else False
                    
                    # Display AI analysis results
                    display_ai_analysis(date_str, am_bool)
        else:
            st.warning("No sensor data available for the selected date and time period.")

else:
    st.info("Looking for sensor data in the /Data/ folder...")

# Footer
st.markdown("---")
st.markdown("Vibration Sensor Dashboard with AI Analysis")
