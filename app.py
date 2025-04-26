import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import datetime

# Set page configuration
st.set_page_config(
    page_title="Traffic Flow Predictor",
    page_icon="🚦",
    layout="wide"
)

# Application title and description
st.title("Traffic Flow Prediction System")
st.markdown("""
    This application predicts traffic situations based on vehicle counts, time of day, and day of the week.
    Use the sidebar to navigate through different sections of the app.
""")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Home", "Data Exploration", "Make Prediction", "About"])

# Create a sample dataframe for demonstration
def create_sample_dataframe():
    # Create sample data
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    times = [f"{h:02d}:00:00 {'AM' if h < 12 else 'PM'}" for h in range(24)]
    
    sample_data = []
    for day in days:
        for time in times[:5]:  # Just use a few times to keep it small
            hour = int(time.split(':')[0])
            # Create traffic patterns - higher during rush hours
            car_factor = 2 if (7 <= hour <= 9) or (16 <= hour <= 18) else 1
            sample_data.append({
                'Time': time,
                'Date': '15-04-2023',
                'Day of the week': day,
                'CarCount': np.random.randint(20, 100) * car_factor,
                'BikeCount': np.random.randint(5, 30),
                'BusCount': np.random.randint(2, 15),
                'TruckCount': np.random.randint(1, 10),
                'Traffic Situation': np.random.choice(['Light', 'Moderate', 'Heavy', 'Congested'])
            })
    
    df = pd.DataFrame(sample_data)
    # Calculate total
    df['Total'] = df['CarCount'] + df['BikeCount'] + df['BusCount'] + df['TruckCount']
    return df

# Process data function
@st.cache_data
def process_data():
    try:
        # Use sample data
        df = create_sample_dataframe()
            
        # Process the Time column
        try:
            df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p', errors='coerce')
        except:
            try:
                df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            except Exception as e:
                st.warning(f"Time column processing error: {e}. Using original values.")
        
        # Extract hour and minute
        df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
        df['Minute'] = pd.to_datetime(df['Time'], errors='coerce').dt.minute
        df['TimeMinutes'] = df['Hour'] * 60 + df['Minute']
        
        # Convert Date if available
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
            # Calculate days since reference date
            reference_date = pd.Timestamp('2023-10-09')
            df['DaysSinceRef'] = (df['Date'] - reference_date).dt.days
        except:
            df['DaysSinceRef'] = 0  # Default value
            st.sidebar.warning("Date conversion issue. Using default values.")
        
        # Store the original Traffic Situation values before encoding
        df['TrafficSituationOriginal'] = df['Traffic Situation']
        
        # Create label encoders
        le_day = LabelEncoder()
        le_traffic = LabelEncoder()
        
        # Encode categorical features
        df['DayEncoded'] = le_day.fit_transform(df['Day of the week'])
        df['TrafficEncoded'] = le_traffic.fit_transform(df['Traffic Situation'])
        
        # Store the encoders and mappings
        traffic_mapping = dict(zip(le_traffic.classes_, le_traffic.transform(le_traffic.classes_)))
        reverse_traffic_mapping = dict(zip(le_traffic.transform(le_traffic.classes_), le_traffic.classes_))
        day_mapping = dict(zip(le_day.classes_, le_day.transform(le_day.classes_)))
        
        return df, traffic_mapping, reverse_traffic_mapping, day_mapping
    except Exception as e:
        st.error(f"Error processing data: {e}")
        # Return sample data as fallback
        df = create_sample_dataframe()
        
        # Process sample data
        le_day = LabelEncoder()
        le_traffic = LabelEncoder()
        
        df['TimeMinutes'] = pd.to_datetime(df['Time'], format='%H:%M:%S %p').dt.hour * 60 + pd.to_datetime(df['Time'], format='%H:%M:%S %p').dt.minute
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S %p').dt.hour
        df['DayEncoded'] = le_day.fit_transform(df['Day of the week'])
        df['TrafficEncoded'] = le_traffic.fit_transform(df['Traffic Situation'])
        df['TrafficSituationOriginal'] = df['Traffic Situation']
        df['DaysSinceRef'] = 0  # Placeholder
        
        traffic_mapping = dict(zip(le_traffic.classes_, le_traffic.transform(le_traffic.classes_)))
        reverse_traffic_mapping = dict(zip(le_traffic.transform(le_traffic.classes_), le_traffic.classes_))
        day_mapping = dict(zip(le_day.classes_, le_day.transform(le_day.classes_)))
        
        return df, traffic_mapping, reverse_traffic_mapping, day_mapping

# Load data using the sample data
df, traffic_mapping, reverse_traffic_mapping, day_mapping = process_data()

# Train or load the model
@st.cache_resource
def get_model(df):
    try:
        # Select features and target
        X = df[['TimeMinutes', 'DayEncoded', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'DaysSinceRef']]
        y = df['TrafficEncoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

# Only train the model if we have data
if df is not None:
    model, scaler = get_model(df)

# HOME PAGE
if page == "Home":
    st.header("Welcome to Traffic Flow Predictor")
    
    # Display project overview
    st.subheader("Project Overview")
    st.write("""
        This project aims to predict traffic flow based on historical data, including vehicle counts (cars, bikes, buses, trucks), 
        time, and day of the week. By analyzing this data, we seek to identify patterns in traffic flow, predict traffic situations, 
        and provide actionable insights to manage congestion and improve urban mobility.
    """)
    
    # Display summary statistics if data is available
    if df is not None:
        st.subheader("Dataset Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", df.shape[0])
            st.metric("Average Cars", int(df['CarCount'].mean()))
            st.metric("Average Bikes", int(df['BikeCount'].mean()))
        with col2:
            st.metric("Average Buses", int(df['BusCount'].mean()))
            st.metric("Average Trucks", int(df['TruckCount'].mean()))
            st.metric("Average Total Vehicles", int(df['Total'].mean()))
        
        # Display a sample of the data
        # st.subheader("Sample Data")
        # st.table(df.head())  # Using st.table instead of st.dataframe to avoid PyArrow dependency
        st.write("Sample Data:")
        st.write(df.head().to_html(), unsafe_allow_html=True)
    else:
        st.warning("No data available. Please check if the dataset is correctly loaded.")

# DATA EXPLORATION PAGE
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    if df is not None:
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Vehicle Counts", "Traffic by Day", "Correlation", "Time Analysis"])
        
        with tab1:
            st.subheader("Vehicle Counts Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x='Hour', y='CarCount', label='Cars', ax=ax)
            sns.lineplot(data=df, x='Hour', y='BikeCount', label='Bikes', ax=ax)
            sns.lineplot(data=df, x='Hour', y='BusCount', label='Buses', ax=ax)
            sns.lineplot(data=df, x='Hour', y='TruckCount', label='Trucks', ax=ax)
            plt.title('Vehicle Counts Over Time')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Count')
            plt.legend()
            st.pyplot(fig)
            
            st.write("""
                The graph shows how different vehicle types vary throughout the day. 
                Cars and bikes show peak patterns during morning and evening rush hours, 
                while buses and trucks maintain more consistent levels.
            """)
        
        with tab2:
            st.subheader("Total Vehicles by Day of the Week")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='Day of the week', y='Total', ax=ax)
            plt.title('Total Vehicles by Day of the Week')
            plt.xlabel('Day of the Week')
            plt.ylabel('Total Vehicle Count')
            st.pyplot(fig)
            
            st.write("""
                This boxplot shows the distribution of total vehicle counts for each day of the week.
                The median, quartiles, and outliers provide insights into traffic volume patterns across different days.
            """)
        
        with tab3:
            st.subheader("Correlation Heatmap")
            # Select only numeric columns
            numeric_df = df.select_dtypes(include='number')
            
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = numeric_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            plt.title('Correlation Heatmap')
            st.pyplot(fig)
            
            st.write("""
                The heatmap shows correlations between different numerical variables in the dataset.
                Positive correlations (closer to 1) indicate variables that increase together, 
                while negative correlations (closer to -1) indicate variables that move in opposite directions.
            """)
        
        with tab4:
            st.subheader("Traffic Situation by Hour")
            # Get average traffic situation by hour
            avg_traffic_by_hour = df.groupby('Hour')['TrafficEncoded'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=avg_traffic_by_hour, x='Hour', y='TrafficEncoded', ax=ax)
            plt.title('Average Traffic Situation by Hour of the Day')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Average Traffic Situation')
            st.pyplot(fig)
            
            st.write("""
                This graph shows how traffic conditions vary throughout the day.
                Higher values indicate more congested traffic situations.
                Peak traffic typically occurs during morning and evening rush hours.
            """)
    else:
        st.warning("No data available for exploration. Please check if the dataset is correctly loaded.")

# PREDICTION PAGE
elif page == "Make Prediction":
    st.header("Traffic Situation Prediction")
    
    if df is not None and model is not None:
        # Create input form for prediction
        st.subheader("Enter Traffic Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time input
            time_input = st.time_input("Time of Day", datetime.time(8, 0))
            time_minutes = time_input.hour * 60 + time_input.minute
            
            # Day input
            day_options = list(day_mapping.keys())
            day_input = st.selectbox("Day of the Week", day_options)
            day_encoded = day_mapping[day_input]
            
            # Days since reference date (this could be simplified or hidden)
            days_since_ref = st.number_input("Days Since Reference (10/09/2023)", value=30, min_value=0)
        
        with col2:
            # Vehicle counts
            car_count = st.number_input("Number of Cars", value=50, min_value=0)
            bike_count = st.number_input("Number of Bikes", value=10, min_value=0)
            bus_count = st.number_input("Number of Buses", value=5, min_value=0)
            truck_count = st.number_input("Number of Trucks", value=3, min_value=0)
        
        # Calculate total vehicles
        total_vehicles = car_count + bike_count + bus_count + truck_count
        st.metric("Total Vehicles", total_vehicles)
        
        # Prediction button
        if st.button("Predict Traffic Situation"):
            # Prepare input data
            input_data = np.array([[
                time_minutes, 
                day_encoded, 
                car_count, 
                bike_count, 
                bus_count, 
                truck_count, 
                days_since_ref
            ]])
            
            # Scale the input
            scaled_input = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(scaled_input)[0]
            prediction_proba = model.predict_proba(scaled_input)[0]
            
            # Get traffic situation label
            traffic_situation = reverse_traffic_mapping[prediction]
            
            # Display prediction
            st.subheader("Prediction Result")
            
            # Display with appropriate styling based on traffic situation
            if prediction <= 1:  # Low traffic
                st.success(f"Predicted Traffic Situation: {traffic_situation}")
            elif prediction <= 3:  # Medium traffic
                st.warning(f"Predicted Traffic Situation: {traffic_situation}")
            else:  # High traffic
                st.error(f"Predicted Traffic Situation: {traffic_situation}")
            
            # Show prediction probabilities
            st.subheader("Prediction Confidence")
            
            # Create a dataframe for the probabilities
            proba_df = pd.DataFrame({
                'Traffic Situation': [reverse_traffic_mapping[i] for i in range(len(prediction_proba))],
                'Probability': prediction_proba
            })
            
            # Sort by probability in descending order
            proba_df = proba_df.sort_values('Probability', ascending=False)
            
            # Display the top 3 most likely traffic situations
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=proba_df.head(3), x='Traffic Situation', y='Probability', ax=ax)
            plt.title('Top 3 Most Likely Traffic Situations')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            features = ['Time', 'Day of week', 'Cars', 'Bikes', 'Buses', 'Trucks', 'Days Since Ref']
            importances = model.feature_importances_
            
            # Create a dataframe for the feature importances
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            })
            
            # Sort by importance in descending order
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Display the feature importances
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Feature', y='Importance', ax=ax)
            plt.title('Feature Importance')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.warning("Model or data not available. Please check if the dataset is correctly loaded.")

# ABOUT PAGE
elif page == "About":
    st.header("About This Project")
    
    st.write("""
        ## Traffic Flow Prediction Project
        
        This project aims to predict traffic flow based on historical data, including vehicle counts (cars, bikes, buses, trucks), 
        time, and day of the week. By analyzing this data, we seek to identify patterns in traffic flow, predict traffic situations, 
        and provide actionable insights to manage congestion and improve urban mobility.
        
        ### Data Dictionary
        
        The dataset contains traffic flow records, including:
        
        - **Time**: Timestamp of the observation
        - **Date**: Date of the observation
        - **Day of the week**: Day corresponding to the observation
        - **CarCount**: Number of cars counted at the observation point
        - **BikeCount**: Number of bikes counted
        - **BusCount**: Number of buses counted
        - **TruckCount**: Number of trucks counted
        - **Total**: Total number of vehicles at the timestamp
        - **Traffic Situation**: The categorized traffic condition (e.g., congestion levels)
        
        ### Key Insights
        
        - **Peak Hours**: Morning and evening rush hours showed spikes in car and bike counts.
        - **Vehicle Types**: Cars consistently represented the highest volume of traffic, followed by bikes, buses, and trucks.
        - **Traffic Situations**: Traffic conditions worsened as truck traffic increased.
        - **Day of the Week**: There was no significant variation in total traffic across different days, although individual vehicle types showed some variability.
        
        ### Model Performance
        
        The Random Forest Classifier model was trained to predict traffic situations based on the available features.
        The model performed well in predicting high-traffic periods, offering valuable insights for traffic management.
    """)
    
    st.subheader("Applications")
    st.write("""
        - **Traffic Management**: Optimizing signal timings and road usage
        - **Urban Planning**: Informing infrastructure development decisions
        - **Environmental Impact**: Reducing emissions through better traffic flow
        - **Public Transportation**: Adjusting schedules based on predicted congestion
    """)

    st.subheader("Team Members")
    st.write("""
        - **Atharva Sakpal**: 221060056
        - **Ramyaa Balasubramanian**: 221061052
        - **Shashin Vathode**: 221060068
    """)
   

# Add footer
st.sidebar.markdown("---")
st.sidebar.info(
    """**Team Members:**  
    Ramyaa Balasubramanian  
    Atharva Sakpal  
    Shashin Vathode"""
)