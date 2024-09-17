# Importing necessary libraries for the app
import os  # OS library for handling file operations

# Streamlit for creating the web app
import streamlit as st

# Pandas for data manipulation
import pandas as pd

# Plotly for data visualization
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Joblib for loading machine learning models
import joblib

# SciPy for statistical functions
from scipy.stats import randint
from scipy import stats




# --- Web Page Preparation ---
st.set_page_config(page_title="Rent Estimator", layout="centered")



# Load the model
try:
    rfr_model = joblib.load('randomregressor_rent.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'randomregressor_rent.pkl' is present in the directory.")
    st.stop()



# --- Data Loading ---

# Define file path
csv_file_path = r"mudah-apartment-kl-selangor.csv"

# Check if the file exists
if not os.path.isfile(csv_file_path):
    raise FileNotFoundError(f"The file at {csv_file_path} does not exist.")

# Load dataset
df = pd.read_csv(csv_file_path)




# Get feature importance from the model
if hasattr(rfr_model, 'feature_importances_'):
    feature_importances = rfr_model.feature_importances_
else:
    st.error("The model does not have feature importances. Ensure it is a model that supports feature importance.")
    st.stop()



# Define feature names
feature_names = [
    'location', 'property_type', 'rooms', 'parking',
    'bathroom', 'size', 'furnished', 'region'
]

# Create a DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)




# --- Data Cleaning ---

# dropping some columns
df = df.drop(columns=[
    'completion_year',
    'prop_name', 
    'facilities', 
    'additional_facilities'
])

# Convert 'rooms' from object to integer, handling errors with 'coerce' to convert invalid parsing to NaN
df['rooms'] = pd.to_numeric(df['rooms'], downcast='integer', errors='coerce')

# Remove rows where 'rooms', 'bathroom', or 'furnished' have NaN values
df = df.dropna(subset=["rooms", "bathroom", "furnished"])

# Replace NaN values in 'parking' column with the mean value of 'parking'
df['parking'] = df['parking'].fillna(df['parking'].mean())

# Define a function to calculate Z-scores and remove outliers for multiple columns, while also filtering by size
def remove_outliers(df, column, threshold=3):
    # Calculate Z-scores for the specified column
    df['z_score'] = stats.zscore(df[column])
    
    # Mark rows as 'Outlier' or 'Not Outlier' based on the threshold
    df['outlier'] = df['z_score'].apply(lambda x: 'Outlier' if abs(x) > threshold else 'Not Outlier')
    
    # Filter rows where the specified column is not an outlier
    df_cleaned = df[df['outlier'] == 'Not Outlier']
    
    # Remove rows where 'size' is greater than 5,000
    df_cleaned = df_cleaned[df_cleaned['size'] <= 5000]
    
    return df_cleaned

# Apply the function to remove outliers based on 'monthly_rent' and size constraint
df_cleaned = remove_outliers(df, 'monthly_rent')

# Remove rows where 'size' is greater than 5,000
df_cleaned = df_cleaned[df_cleaned['size'] <= 5000]

# Function for User Input
def get_user_inputs():
    # unit Information
    st.sidebar.markdown("""
        <h3 style="color: #0CABA8;">Unit Information</h3>
        """, unsafe_allow_html=True)
    rooms = st.sidebar.slider("Number of Rooms", min_value=1, max_value=10, value=3)
    parking = st.sidebar.slider("Parking Availability", min_value=0, max_value=5, value=1)
    bathroom = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=5, value=1)
    size = st.sidebar.slider("Size (sqft)", min_value=200, max_value=3000, step=10, value=800)
    furnished = st.sidebar.radio("Is the property furnished?", ('Fully Furnished', 'Not Furnished', 'Partially Furnished'))
    property_type = st.sidebar.selectbox('Select Property Type', [
        'Condominium', 'Flat', 'Apartment', 'Service Residence', 'Others',
        'Townhouse Condo', 'Houses', 'Studio', 'Duplex'])
    
    # Location
    st.sidebar.markdown("""
        <h3 style="color: #0CABA8;">Location</h3>
        """, unsafe_allow_html=True)
    region = st.sidebar.selectbox("Select Region", ['Kuala Lumpur', 'Selangor'])
    location = st.sidebar.selectbox(
        'Select Area', [
        'Cheras', 'Petaling Jaya', 'Puncak Alam', 'Bandar Sunway', 'Kajang', 
        'Kota Damansara', 'Seri Kembangan', 'Damansara Damai', 'Bandar Sri Damansara', 
        'Damansara Perdana', 'Batu Caves', 'Puchong', 'Shah Alam', 'Rawang', 
        'Kuala Selangor', 'Semenyih', 'Sepang', 'Cyberjaya', 'Klang', 'Bukit Jalil', 
        'Setapak', 'OUG', 'Desa Petaling', 'Kepong', 'Beranang', 'Bandar Saujana Putra', 
        'Kuala Langat', 'Bandar Sungai Long', 'Setia Alam', '360', 'Serendah', 
        'Bangi', 'Wangsa Maju', 'Titiwangsa', 'Kuchai Lama', 'Sungai Besi', 
        'Mid Valley City', 'Serdang', 'Ampang', 'Selayang', 'Dengkil', 'Gombak', 
        'Balakong', 'Port Klang', 'Bukit Subang', 'Banting', 'Kapar', 'Pandan Indah', 
        'Jenjarom', 'Keramat', 'Old Klang Road', 'Salak Selatan', 'Sri Damansara', 
        'Sungai Buloh', 'Hulu Langat', '43', 'Putra Heights', 'Puchong South', 
        'Sentul', 'Jalan Kuching', 'Pandan Jaya', 'Jalan Ipoh', 'Bandar Tasik Selatan', 
        'Ampang Hilir', 'KLCC', 'Jinjang', 'Subang Bestari', 'Salak Tinggi', 
        'Bandar Kinrara', 'Subang Jaya', 'Puncak Jalil', '389', 'Bandar Damai Perdana', 
        'Taman Desa', 'Bukit Beruntung', 'Bandar Utama', 'Others', 'KL City', 
        'Solaris Dutamas', 'Bangsar South', 'Sri Petaling', 'Pandan Perdana', 
        'Bukit Bintang', 'Bandar Mahkota Cheras', 'Damansara Heights', 'Alam Impian', 
        'Segambut', 'Setiawangsa', 'Seputeh', 'Mont Kiara', 'Kota Kemuning', 
        'USJ', 'Ara Damansara', 'Damansara Jaya', 'I-City', 'Desa Pandan', 
        'Taman Melawati', 'Pantai', 'Bandar Menjalara', 'Sungai Penchala', 
        'Bangsar', '369', 'Kelana Jaya', 'Taman Tun Dr Ismail', 'Sri Hartamas', 
        'Brickfields', 'Saujana Utama', 'Bandar Bukit Raja', 'Glenmarie', 
        '517', 'Jalan Sultan Ismail', 'Damansara', 'Pulau Indah (Pulau Lumut)', 
        'Bandar Botanic', 'Mutiara Damansara', '639', 'Pudu', 'City Centre', 
        'Ulu Klang', 'Bandar Bukit Tinggi', 'Telok Panglima Garang', 
        'KL Sentral', 'KL Eco City', 'Bukit Tunku'
    ])

    return rooms, parking, bathroom, size, furnished, property_type, region, location


# --- Setting separate columns for App and Documentation ---

tab1, tab2 = st.tabs(["App", "Documentation"])

with tab1:
    st.title("🏠 Rental Estimator App")
    st.markdown("""Our Rent Estimator uses advanced machine learning to predict how much rent you might expect to pay for a property based on several important factors.  
                Adjust the inputs on the sidebar to predict rental prices. 
                """)

    # Get inputs from the user
    rooms, parking, bathroom, size, furnished, property_type, region, location = get_user_inputs()

    # User input DataFrame
    user_input = pd.DataFrame({
        'location': [location],
        'property_type': [property_type],
        'rooms': [rooms],
        'parking': [parking],
        'bathroom': [bathroom],
        'size': [size],
        'furnished': [furnished],
        'region': [region]
    })

    # Mappings
    property_type_mapping = {
        'Condominium': 0, 'Apartment': 1, 'Service Residence': 2, 'Studio': 3,
        'Flat': 4, 'Duplex': 5, 'Others': 6, 'Townhouse Condo': 7, 
        'Condo / Services residence / Penthouse / Townhouse': 8, 'Houses': 9
    }

    location_mapping = {
        'Cheras': 1, 'Petaling Jaya': 2, 'Puncak Alam': 3, 'Bandar Sunway': 4,
        'Kajang': 5, 'Kota Damansara': 6, 'Seri Kembangan': 7, 'Damansara Damai': 8,
        'Bandar Sri Damansara': 9, 'Damansara Perdana': 10, 'Batu Caves': 11,
        'Puchong': 12, 'Shah Alam': 13, 'Rawang': 14, 'Kuala Selangor': 15,
        'Semenyih': 16, 'Sepang': 17, 'Cyberjaya': 18, 'Klang': 19, 'Bukit Jalil': 20,
        'Setapak': 21, 'OUG': 22, 'Desa Petaling': 23, 'Kepong': 24, 'Beranang': 25,
        'Bandar Saujana Putra': 26, 'Kuala Langat': 27, 'Bandar Sungai Long': 28,
        'Setia Alam': 29, '360': 30, 'Serendah': 31, 'Bangi': 32, 'Wangsa Maju': 33,
        'Titiwangsa': 34, 'Kuchai Lama': 35, 'Sungai Besi': 36, 'Mid Valley City': 37,
        'Serdang': 38, 'Ampang': 39, 'Selayang': 40, 'Dengkil': 41, 'Gombak': 42,
        'Balakong': 43, 'Port Klang': 44, 'Bukit Subang': 45, 'Banting': 46,
        'Kapar': 47, 'Pandan Indah': 48, 'Jenjarom': 49, 'Keramat': 50,
        'Old Klang Road': 51, 'Salak Selatan': 52, 'Sri Damansara': 53,
        'Sungai Buloh': 54, 'Hulu Langat': 55, '43': 56, 'Putra Heights': 57,
        'Puchong South': 58, 'Sentul': 59, 'Jalan Kuching': 60, 'Pandan Jaya': 61,
        'Jalan Ipoh': 62, 'Bandar Tasik Selatan': 63, 'Ampang Hilir': 64,
        'KLCC': 65, 'Jinjang': 66, 'Subang Bestari': 67, 'Salak Tinggi': 68,
        'Bandar Kinrara': 69, 'Subang Jaya': 70, 'Puncak Jalil': 71, '389': 72,
        'Bandar Damai Perdana': 73, 'Taman Desa': 74, 'Bukit Beruntung': 75,
        'Bandar Utama': 76, 'Others': 77, 'KL City': 78, 'Solaris Dutamas': 79,
        'Bangsar South': 80, 'Sri Petaling': 81, 'Pandan Perdana': 82,
        'Bukit Bintang': 83, 'Bandar Mahkota Cheras': 84, 'Damansara Heights': 85,
        'Alam Impian': 86, 'Segambut': 87, 'Setiawangsa': 88, 'Seputeh': 89,
        'Mont Kiara': 90, 'Kota Kemuning': 91, 'USJ': 92, 'Ara Damansara': 93,
        'Damansara Jaya': 94, 'I-City': 95, 'Desa Pandan': 96, 'Taman Melawati': 97,
        'Pantai': 98, 'Bandar Menjalara': 99, 'Sungai Penchala': 100, 'Bangsar': 101,
        '369': 102, 'Kelana Jaya': 103, 'Taman Tun Dr Ismail': 104, 'Sri Hartamas': 105,
        'Brickfields': 106, 'Saujana Utama': 107, 'Bandar Bukit Raja': 108, 'Glenmarie': 109,
        '517': 110, 'Jalan Sultan Ismail': 111, 'Damansara': 112, 'Pulau Indah (Pulau Lumut)': 113,
        'Bandar Botanic': 114, 'Mutiara Damansara': 115, '639': 116, 'Pudu': 117,
        'City Centre': 118, 'Ulu Klang': 119, 'Bandar Bukit Tinggi': 120,
        'Telok Panglima Garang': 121, 'KL Sentral': 122, 'KL Eco City': 123, 'Bukit Tunku': 124
    }
    region_mapping = {'Kuala Lumpur': 0, 'Selangor': 1}
    furnished_mapping = {'Not Furnished': 0, 'Partially Furnished': 1, 'Fully Furnished': 2}

    # Map categorical data to numerical
    user_input['property_type'] = user_input['property_type'].map(property_type_mapping)
    user_input['location'] = user_input['location'].map(location_mapping)
    user_input['region'] = user_input['region'].map(region_mapping)
    user_input['furnished'] = user_input['furnished'].map(furnished_mapping)

    input_df = user_input[['location', 'property_type', 'rooms', 'parking', 'bathroom', 'size', 'furnished', 'region']]

    # Predict and display results
    prediction = rfr_model.predict(input_df)[0]
    st.write(f"**Predicted Rent:** RM {prediction:,.2f}")

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)

    
    

    # --- Table of Contents ---
with tab2:
    # Set up the Table of Contents
    st.markdown("""
    <style>
        .toc {
            font-family: Arial, sans-serif;
            font-size: 18px;
            line-height: 1.6;
            padding: 10px;
        }
        .toc a {
            text-decoration: none;
            color: #FF6347; /* Tomato color for links */
        }
        .toc a:hover {
            text-decoration: underline;
        }
        .toc h1 {
            font-size: 24px;
            margin-bottom: 10px;
        }
    </style>

    <div class="toc">
        <h1>Table of Contents</h1>
        <ul>
            <li><a href="#introduction">Problem Statement</a></li>
            <li><a href="#data-overview">Data Overview</a></li>
            <li><a href="#eda">EDA</a></li>
            <li><a href="#modeling">Modeling</a></li>
            <li><a href="#conc">Conclusion</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)









    # --- Problem Statement ---

    st.markdown("""
                <a name="introduction"></a>
                
                <h2 style="color: #0CABA8;">Problem Statement</h2>

                The goal of this tool is to estimate the rental price of properties using a predictive model. 
                By analyzing various property attributes like location, number of rooms, size, and furnishings, 
                this tool provides an estimate of what the rent for a property might be. 
                This can help potential renters and landlords make informed decisions.
                
                """, unsafe_allow_html=True)

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)









    # --- Data Overview ---

    st.markdown("""
                <a name="data-overview"></a>

                <h2 style="color: #0CABA8;">Data Overview</h2>

                """, unsafe_allow_html=True)
    st.write("""
            The dataset is scraped from mudah.my and can be found in [Kaggle](https://www.kaggle.com/datasets/ariewijaya/rent-pricing-kuala-lumpur-malaysi). 
            Here we provide an overview of the data. The dataset includes features and analyzes how these factors impact rental.
            """)
    st.write("This table displays the dataset in the first 5 rows")
    st.write(df.head())  # Display the first few rows of the dataset

    st.write("This table displays summary statistics of the dataset.")
    st.write(df.describe())  # Display summary statistics

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)








    # --- EDA ---

    st.write("""
            <a name="eda"></a>

            <h2 style="color: #0CABA8;">EDA</h2>

            In this section, we perform Exploratory Data Analysis (EDA) to gain insights into the dataset. We start by examining the distribution of key features and visualizing relationships between them.

            """, unsafe_allow_html=True)


    ### Plot:Boxplots of Continuous Variables ###
    st.write("#### Boxplots of Continuous Variables")
    fig = px.box(df, y=['monthly_rent','rooms', 'parking', 'bathroom', 'size'])
    st.plotly_chart(fig)   
    st.write("""
            The distribution of the 'size' feature appears to be right-skewed and contains outliers. These outliers can negatively impact the performance of the model. 
            To address this, we will use a Z-score threshold of 3 to identify outliers and remove them from the dataset. As for size, any observations with size larger than 5000 square feet are removed as well.
            """)
    
    st.write("#### Boxplots of Continuous Variables after removing variables")
    # Apply the function to remove outliers for the relevant columns
    df_cleaned = remove_outliers(df, ['monthly_rent'])
    fig = px.box(df_cleaned, y=['monthly_rent','rooms', 'parking', 'bathroom', 'size'])
    st.plotly_chart(fig)   
    st.write("""
            The distribution of the 'size' feature appears to be right-skewed and contains outliers. These outliers can negatively impact the performance of the model. 
            To address this, we will use a Z-score threshold of 3 to identify outliers and remove them from the dataset.
            """)
    
    ### Plot:Distribution of Property Sizes after removing outliers ### 
    st.write("#### Distribution of Property Sizes after removing outliers")
    fig_size_dist = px.histogram(
        df_cleaned, x='size',
        labels={'size': 'Size (sqft)'}
    )
    st.plotly_chart(fig_size_dist)

    ### Plot:Distribution of Property Sizes after removing outliers ###     
    st.write("#### Size vs. Number of Rooms")
    fig = px.scatter(df_cleaned, x='size', y='monthly_rent')
    st.plotly_chart(fig)
    st.write("""
            This scatter plot is used to examine the relationship between the rental price (rent) and the size of properties.
            The chart shows a positive correlation, which is expected because larger properties generally command higher rents.
            """)
    
    ### Plot:Distribution of Property Sizes after removing outliers ###   
    st.write("#### Correlation Heatmap for Continuous Features")
    # Specify the columns for which to compute the correlation matrix
    selected_columns = ['size', 'rooms', 'parking', 'bathroom']

    # Compute correlation matrix for the selected columns
    corr = df_cleaned[selected_columns].corr()

    # Create a heatmap using Plotly
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig)
    st.write("""
            The correlation heatmap provides insights into the relationships between variables. Here's a summary of the key points:

            - Size vs Bathroom (0.44) and Size vs Rooms (0.38): There is a moderate positive correlation, meaning as the size of a property increases, the number of rooms and bathrooms tends to increase as well, though not strongly.

            - Rooms vs Bathroom (0.68): This is the highest correlation in the dataset, indicating a strong positive relationship. Properties with more rooms tend to have more bathrooms.

            - Rooms vs Parking (0.28) and Size vs Parking (0.20): These show weak positive correlations, suggesting a slight tendency for larger properties and those with more rooms to have more parking spaces.

            - Parking vs Bathroom (0.33): There is a weak-to-moderate positive correlation, implying that properties with more bathrooms tend to have more parking spaces, but the relationship isn't particularly strong.

            Overall, size is moderately related to rooms and bathrooms, and rooms are strongly related to bathrooms, but parking is only weakly correlated with other features.
            """)

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)









    # --- Modelling ---

    st.markdown("""
                <a name="modeling"></a>

                <h2 style="color: #0CABA8;">Modelling</h2>

                **Model Choice**

                We use several classification algorithms to compare performance, including:
                - **Random Forest**
                - **Gradient Boosting**
                - **Linear Regression**
                

                The model performance comparison shows:

                - RandomForestRegressor: Best performance with the lowest RMSE (306.82) and highest R² (0.78), indicating strong predictive accuracy.
                - GradientBoostingRegressor: Moderate performance with an RMSE of 391.78 and R² of 0.64, showing decent predictive ability.
                - LinearRegression: Worst performance with the highest RMSE (466.96) and lowest R² (0.49), indicating weaker accuracy compared to the other models.
                

                """, unsafe_allow_html=True)
    
    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.write("""
            **Feature Importance**
             
            For this app, we will use the Random Forest model. The feature importance analysis reveals which features have the most influence on predicting the target variable. 
            Features with higher importance scores contribute more significantly to the model's predictions, guiding us to focus on the key drivers of the outcome.

            By understanding these important features, we can not only improve the model's performance but also gain valuable insights into the factors that most affect the target. 
            This information can be used to optimize decision-making and refine future models.
            """)
    
    # Create a DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)


    fig = px.bar(
        importance_df,
        y='Importance',
        x='Feature',
        title='Feature Importance',
        orientation='v',
        labels={'Importance': 'Feature Importance', 'Feature': 'Feature'},
        color='Importance'
    )

    st.plotly_chart(fig)
    
    st.write("""
            The feature importance analysis of the Random Forest model highlights the following key drivers for predicting the target variable:

            - Size (0.3548): The most influential feature, indicating that property size has the strongest impact on the model's predictions.
            - Furnished (0.2497): The second most important feature, showing a significant effect on predictions.
            - Location (0.1770): Also plays a notable role, contributing meaningfully to the model.
            - Property Type (0.0578), Region (0.0489), and Parking (0.0480): These features have moderate importance but are less impactful compared to size, furnished, and location.
            - Rooms (0.0407) and Bathroom (0.0232): These features have the least influence on the model's predictions.
            
            Overall, size and whether the property is furnished are the dominant factors, while other features contribute to a lesser extent.
            """)
    
    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)

    






    # --- Conclusion ---

    st.markdown("""
                <a name="conc"></a>

                <h2 style="color: #0CABA8;">Conclusion</h2>

                In this section, we summarize the key findings from the modeling analysis.

                ### Key Findings:
                - The Random Forest Regressor provided accurate predictions of rental prices based on the input features.
                - Key factors influencing rent prices include location, size, and furnishing status.
                
                ### Recommendations:
                - For Renters: Use this tool to get an estimate of how much you might expect to pay for a property based on its features.
                - For Landlords: Assess competitive rental pricing based on similar properties in your area.
            
                ### Future Work:
                - Feature Enhancement: Explore adding more features, such as proximity to amenities or public transport, to further refine the predictions.
                - Model Updates: Continuously update the model with new data to improve its accuracy and relevance.

                Overall, the developed model serves as a valuable tool for predicting customer attrition and can help the bank proactively manage customer relationships.
                """, unsafe_allow_html=True)

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)

