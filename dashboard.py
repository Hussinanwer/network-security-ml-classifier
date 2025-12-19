"""
Streamlit dashboard for network traffic classification.
Provides an interactive web interface for model predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from preprocessing import NetworkTrafficPreprocessor

# Page configuration
st.set_page_config(
    page_title="Network Traffic Classifier",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class labels (matching the notebook exactly!)
CLASS_LABELS = {
    0: "Normal",
    1: "vsftpd Backdoor",
    2: "SSH Brute Force"
}

CLASS_COLORS = {
    0: "#2ecc71",  # Green for Normal
    1: "#e74c3c",  # Red for vsftpd Backdoor (MALICIOUS!)
    2: "#f39c12"   # Orange for SSH Brute Force (MALICIOUS!)
}

CLASS_DESCRIPTIONS = {
    0: "Normal network traffic - no threats detected",
    1: "vsftpd Backdoor Attack - MALICIOUS! Exploits vsftpd vulnerability (CVE-2011-2523)",
    2: "SSH Brute Force Attack - MALICIOUS! Automated password guessing attempts"
}


@st.cache_resource
def load_models():
    """Load trained models, preprocessor, and feature selector."""
    models_dir = Path("models")

    try:
        # Load preprocessor
        with open(models_dir / "preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        # Load feature selector
        with open(models_dir / "feature_selector.pkl", "rb") as f:
            feature_selector = pickle.load(f)

        # Load Random Forest model (primary model)
        with open(models_dir / "rf_model.pkl", "rb") as f:
            rf_model = pickle.load(f)

        # Load metadata
        with open(models_dir / "model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        return preprocessor, feature_selector, rf_model, metadata

    except FileNotFoundError as e:
        st.error(f"Model files not found! Please run train.py first. Missing: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


def create_feature_input_form():
    """Create form for manual feature input."""
    st.subheader("Enter Network Traffic Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Connection Information**")
        src_ip = st.text_input("Source IP", value="192.168.113.129")
        dst_ip = st.text_input("Destination IP", value="192.168.113.130")
        src_port = st.number_input("Source Port", min_value=0, max_value=65535, value=44017)
        dst_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=22)
        protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"], index=0)
        duration = st.number_input("Duration (seconds)", min_value=0.0, value=2.103768, format="%.6f")

    with col2:
        st.markdown("**Packet Statistics**")
        total_packets = st.number_input("Total Packets", min_value=0, value=26)
        total_bytes = st.number_input("Total Bytes", min_value=0, value=5451)
        min_packet_size = st.number_input("Min Packet Size", min_value=0, value=66)
        max_packet_size = st.number_input("Max Packet Size", min_value=0, value=1602)
        avg_packet_size = st.number_input("Avg Packet Size", min_value=0.0, value=209.65)
        std_packet_size = st.number_input("Std Packet Size", min_value=0.0, value=340.88)

    with col3:
        st.markdown("**TCP Flags**")
        syn_count = st.number_input("SYN Count", min_value=0, value=2)
        ack_count = st.number_input("ACK Count", min_value=0, value=25)
        fin_count = st.number_input("FIN Count", min_value=0, value=2)
        rst_count = st.number_input("RST Count", min_value=0, value=0)
        psh_count = st.number_input("PSH Count", min_value=0, value=8)
        urg_count = st.number_input("URG Count", min_value=0, value=0)

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Flow Metrics**")
        packets_per_second = st.number_input("Packets/Second", min_value=0.0, value=12.35)
        bytes_per_second = st.number_input("Bytes/Second", min_value=0.0, value=2591.23)
        bytes_per_packet = st.number_input("Bytes/Packet", min_value=0.0, value=209.65)
        forward_packets = st.number_input("Forward Packets", min_value=0, value=13)
        backward_packets = st.number_input("Backward Packets", min_value=0, value=13)
        forward_bytes = st.number_input("Forward Bytes", min_value=0, value=2907)
        backward_bytes = st.number_input("Backward Bytes", min_value=0, value=2544)
        forward_backward_ratio = st.number_input("Forward/Backward Ratio", min_value=0.0, value=1.14)

    with col5:
        st.markdown("**Inter-Arrival Time (IAT)**")
        avg_iat = st.number_input("Average IAT", min_value=0.0, value=0.084151, format="%.6f")
        std_iat = st.number_input("Std IAT", min_value=0.0, value=0.385122, format="%.6f")
        min_iat = st.number_input("Min IAT", min_value=0.0, value=0.00000095, format="%.8f")
        max_iat = st.number_input("Max IAT", min_value=0.0, value=1.970212, format="%.6f")

        st.markdown("**Port Indicators**")
        syn_ack_ratio = st.number_input("SYN/ACK Ratio", min_value=0.0, value=0.08)
        is_port_22 = st.selectbox("Is Port 22 (SSH)?", [0, 1], index=1)
        is_port_6200 = st.selectbox("Is Port 6200?", [0, 1], index=0)
        is_ftp_port = st.selectbox("Is FTP Port?", [0, 1], index=0)
        is_ftp_data_port = st.selectbox("Is FTP Data Port?", [0, 1], index=0)

    # Compile features into dictionary
    features = {
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': src_port,
        'dst_port': dst_port,
        'protocol': protocol,
        'duration': duration,
        'total_packets': total_packets,
        'total_bytes': total_bytes,
        'min_packet_size': min_packet_size,
        'max_packet_size': max_packet_size,
        'avg_packet_size': avg_packet_size,
        'std_packet_size': std_packet_size,
        'syn_count': syn_count,
        'ack_count': ack_count,
        'fin_count': fin_count,
        'rst_count': rst_count,
        'psh_count': psh_count,
        'urg_count': urg_count,
        'packets_per_second': packets_per_second,
        'bytes_per_second': bytes_per_second,
        'bytes_per_packet': bytes_per_packet,
        'forward_packets': forward_packets,
        'backward_packets': backward_packets,
        'forward_bytes': forward_bytes,
        'backward_bytes': backward_bytes,
        'forward_backward_ratio': forward_backward_ratio,
        'avg_iat': avg_iat,
        'std_iat': std_iat,
        'min_iat': min_iat,
        'max_iat': max_iat,
        'syn_ack_ratio': syn_ack_ratio,
        'is_port_22': is_port_22,
        'is_port_6200': is_port_6200,
        'is_ftp_port': is_ftp_port,
        'is_ftp_data_port': is_ftp_data_port
    }

    return features


def display_prediction_result(prediction, probabilities):
    """Display prediction results with visualization."""
    st.markdown("---")
    st.subheader("Prediction Result")

    # Main prediction display
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        st.metric(
            label="Predicted Class",
            value=f"Class {prediction}",
            delta=CLASS_LABELS[prediction]
        )

    with col2:
        confidence = probabilities[prediction] * 100
        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

    with col3:
        st.info(CLASS_DESCRIPTIONS[prediction])

    # Probability visualization
    st.markdown("### Class Probabilities")

    # Create probability bar chart
    prob_df = pd.DataFrame({
        'Class': [f"Class {i}: {CLASS_LABELS[i]}" for i in range(3)],
        'Probability': [probabilities[i] * 100 for i in range(3)],
        'Color': [CLASS_COLORS[i] for i in range(3)]
    })

    fig = px.bar(
        prob_df,
        x='Probability',
        y='Class',
        orientation='h',
        color='Color',
        color_discrete_map="identity",
        text='Probability'
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        xaxis_title="Probability (%)",
        yaxis_title="",
        showlegend=False,
        height=250
    )

    st.plotly_chart(fig, use_container_width=True)


def display_batch_results(predictions, probabilities, original_df=None):
    """Display batch prediction results."""
    st.markdown("---")
    st.subheader("Batch Prediction Results")

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", len(predictions))

    with col2:
        class_0_count = sum(predictions == 0)
        st.metric(CLASS_LABELS[0], class_0_count)

    with col3:
        class_1_count = sum(predictions == 1)
        st.metric(CLASS_LABELS[1], class_1_count)

    with col4:
        class_2_count = sum(predictions == 2)
        st.metric(CLASS_LABELS[2], class_2_count)

    # Class distribution pie chart
    st.markdown("### Class Distribution")

    class_counts = pd.DataFrame({
        'Class': [CLASS_LABELS[i] for i in range(3)],
        'Count': [sum(predictions == i) for i in range(3)]
    })

    fig = px.pie(
        class_counts,
        values='Count',
        names='Class',
        color='Class',
        color_discrete_map={
            CLASS_LABELS[0]: CLASS_COLORS[0],
            CLASS_LABELS[1]: CLASS_COLORS[1],
            CLASS_LABELS[2]: CLASS_COLORS[2]
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed results table
    st.markdown("### Detailed Results")

    results_df = pd.DataFrame({
        'Index': range(len(predictions)),
        'Predicted Class': predictions,
        'Prediction Label': [CLASS_LABELS[p] for p in predictions],
        'Confidence (%)': [probabilities[i][predictions[i]] * 100 for i in range(len(predictions))],
        'Class 0 Prob (%)': [probabilities[i][0] * 100 for i in range(len(predictions))],
        'Class 1 Prob (%)': [probabilities[i][1] * 100 for i in range(len(predictions))],
        'Class 2 Prob (%)': [probabilities[i][2] * 100 for i in range(len(predictions))]
    })

    # Add color coding
    def color_prediction(row):
        color = CLASS_COLORS[row['Predicted Class']]
        return [f'background-color: {color}; color: white' if col == 'Prediction Label' else '' for col in row.index]

    st.dataframe(
        results_df.style.apply(color_prediction, axis=1).format({
            'Confidence (%)': '{:.2f}',
            'Class 0 Prob (%)': '{:.2f}',
            'Class 1 Prob (%)': '{:.2f}',
            'Class 2 Prob (%)': '{:.2f}'
        }),
        use_container_width=True,
        height=400
    )

    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )


def main():
    """Main dashboard application."""

    # Header
    st.title("🔒 Network Traffic Classification Dashboard")
    st.markdown("Classify network traffic patterns using machine learning")

    # Load models
    with st.spinner("Loading models..."):
        preprocessor, feature_selector, rf_model, metadata = load_models()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Mode",
        ["Single Prediction", "Batch Prediction", "Model Info"]
    )

    if page == "Single Prediction":
        st.header("Single Prediction Mode")
        st.markdown("Enter network traffic features manually for classification")

        features = create_feature_input_form()

        if st.button("Predict", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    # Convert to DataFrame
                    df = pd.DataFrame([features])

                    # Preprocess
                    X_processed = preprocessor.transform(df)

                    # Apply feature selection
                    X_selected = feature_selector.transform(X_processed)

                    # Predict
                    prediction = rf_model.predict(X_selected)[0]
                    probabilities = rf_model.predict_proba(X_selected)[0]

                    # Display results
                    display_prediction_result(prediction, probabilities)

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    elif page == "Batch Prediction":
        st.header("Batch Prediction Mode")
        st.markdown("Upload a CSV file with network traffic data for bulk classification")

        # Required columns (all original features except 'label')
        required_columns = [
            'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'duration',
            'total_packets', 'total_bytes', 'min_packet_size', 'max_packet_size',
            'avg_packet_size', 'std_packet_size', 'syn_count', 'ack_count',
            'fin_count', 'rst_count', 'psh_count', 'urg_count',
            'packets_per_second', 'bytes_per_second', 'bytes_per_packet',
            'forward_packets', 'backward_packets', 'forward_bytes', 'backward_bytes',
            'forward_backward_ratio', 'avg_iat', 'std_iat', 'min_iat', 'max_iat',
            'syn_ack_ratio', 'is_port_22', 'is_port_6200', 'is_ftp_port', 'is_ftp_data_port'
        ]

        # Create template CSV
        st.markdown("### Download CSV Template")
        st.info("Your CSV must include all 35 features listed below (without the 'label' column)")

        template_df = pd.DataFrame(columns=required_columns)
        # Add one example row from the form defaults
        template_df.loc[0] = {
            'src_ip': '192.168.113.129',
            'dst_ip': '192.168.113.130',
            'src_port': 44017,
            'dst_port': 22,
            'protocol': 'TCP',
            'duration': 2.103768,
            'total_packets': 26,
            'total_bytes': 5451,
            'min_packet_size': 66,
            'max_packet_size': 1602,
            'avg_packet_size': 209.65,
            'std_packet_size': 340.88,
            'syn_count': 2,
            'ack_count': 25,
            'fin_count': 2,
            'rst_count': 0,
            'psh_count': 8,
            'urg_count': 0,
            'packets_per_second': 12.35,
            'bytes_per_second': 2591.23,
            'bytes_per_packet': 209.65,
            'forward_packets': 13,
            'backward_packets': 13,
            'forward_bytes': 2907,
            'backward_bytes': 2544,
            'forward_backward_ratio': 1.14,
            'avg_iat': 0.084151,
            'std_iat': 0.385122,
            'min_iat': 0.00000095,
            'max_iat': 1.970212,
            'syn_ack_ratio': 0.08,
            'is_port_22': 1,
            'is_port_6200': 0,
            'is_ftp_port': 0,
            'is_ftp_data_port': 0
        }

        template_csv = template_df.to_csv(index=False)
        st.download_button(
            label="Download CSV Template",
            data=template_csv,
            file_name="network_traffic_template.csv",
            mime="text/csv",
            help="Download this template and fill in your data"
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)

                # Check for required columns
                missing_columns = set(required_columns) - set(df.columns)
                extra_columns = set(df.columns) - set(required_columns) - {'label'}  # 'label' is optional

                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(sorted(missing_columns))}")
                    st.info("Please download the CSV template above and ensure your file includes all required columns.")
                    st.stop()

                if extra_columns:
                    st.warning(f"⚠️ Extra columns will be ignored: {', '.join(sorted(extra_columns))}")

                st.success(f"✅ Loaded {len(df)} samples with all required columns")
                st.markdown("**Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)

                if st.button("Predict All", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        # Remove label column if present
                        if 'label' in df.columns:
                            df = df.drop('label', axis=1)

                        # Preprocess
                        X_processed = preprocessor.transform(df)

                        # Apply feature selection
                        X_selected = feature_selector.transform(X_processed)

                        # Predict
                        predictions = rf_model.predict(X_selected)
                        probabilities = rf_model.predict_proba(X_selected)

                        # Display results
                        display_batch_results(predictions, probabilities, df)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                import traceback
                with st.expander("Show detailed error"):
                    st.code(traceback.format_exc())

    else:  # Model Info
        st.header("Model Information")

        rf_metrics = metadata['random_forest']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Model Performance")
            st.metric("Test Accuracy", f"{rf_metrics['test_accuracy']:.4f}")
            st.metric("Precision", f"{rf_metrics['precision']:.4f}")
            st.metric("Recall", f"{rf_metrics['recall']:.4f}")
            st.metric("F1 Score", f"{rf_metrics['f1_score']:.4f}")

        with col2:
            st.markdown("### Confusion Matrix")
            cm = np.array(rf_metrics['confusion_matrix'])

            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=[CLASS_LABELS[i] for i in range(3)],
                y=[CLASS_LABELS[i] for i in range(3)],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues'
            ))

            fig.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Final Features Used")
        st.info(f"The model uses {len(metadata['final_features'])} features after preprocessing")

        features_df = pd.DataFrame({
            'Feature': metadata['final_features']
        })
        st.dataframe(features_df, use_container_width=True)

        st.markdown("### Removed Features")
        removed_features = metadata['removed_features'].get('removed', [])
        if removed_features:
            st.write(f"**Removed Features:** {', '.join(removed_features)}")
            st.info("""
            Features removed (matching notebook preprocessing):
            - Zero variance: urg_count, is_ftp_data_port (only 1 unique value)
            - Leaky indicator: is_port_6200 (perfect attack indicator)
            - Encoded categoricals: src_ip, dst_ip (removed after encoding for generalization)
            """)
        else:
            st.write("**Removed Features:** None")


if __name__ == "__main__":
    main()
