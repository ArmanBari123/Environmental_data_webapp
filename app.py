
import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns

# This function creates the Home page
def home():
    st.title("Climate change, Extreme events and Migration: A machine learning approach")
    
    # Load and display the image
    image = "Regions.jpg"
    st.image(image, caption="Regions Map", use_column_width=True)

# This function creates the Historical Data Analysis page
def historical_data_analysis():
    st.title("Historical Data Analysis")

    # Load and display the data
    data = pd.read_excel("Cleaned Natural disasters.xlsx")

    # Plot the frequency bar chart of the incident_type column
    st.title("Incident Type Frequency Bar Chart:")
    plt.figure(figsize=(10, 6))
    sns.countplot(x="incident_type", data=data)
    plt.xticks(rotation=45)
    plt.xlabel("Incident Type")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    st.title("Natural Disasters Frequency by State")
    
    # Read the data from the CSV file
    disaster_freq = data.groupby(['State_full', 'incident_type']).size().unstack()
    
    # Sort the values by descending order of frequency
    sorted_disaster_freq = disaster_freq.sum(axis=1).sort_values(ascending=False)
    disaster_freq = disaster_freq.loc[sorted_disaster_freq.index]
    
    # Slider to select the top N states for visualization
    top_n_states = st.slider("Select the number of states to display:", min_value=5, max_value=len(disaster_freq), value=10)
    
    # Take only the top N states and rotate the x-axis labels for better readability
    plt.figure(figsize=(12, 6))
    disaster_freq.head(top_n_states).plot(kind='bar', stacked=True, cmap='Paired', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("State", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(f"Top {top_n_states} States with Highest Natural Disasters Frequency", fontsize=16)
    plt.legend(title="Incident Type", title_fontsize=12, fontsize=10)
    st.pyplot(plt)

    def plot_yearly_data(file_name, title, ylabel, selected_states):
        st.title(title)
        
        # Read the data from the provided dataset
        df_data = pd.read_excel(file_name)
        st.write("Displaying the data:")
        #st.dataframe(df_data)
        
        # List of all available states
        all_states = df_data.columns.tolist()[1:]  # Exclude 'Year' column
        
        # Multi-select widget to choose states
        selected_states = st.multiselect("Select States:", all_states, default=selected_states, key=file_name)  # Use filename as the key
        
        # Filter the data for the selected states
        selected_states_data = df_data[['Year'] + selected_states]
        
        # Plot the line chart for the selected states (if at least one state is selected)
        if selected_states_data.shape[1] > 1:  # Ensure there is at least one state selected
            plt.figure(figsize=(10, 6))
            for state in selected_states_data.columns[1:]:
                plt.plot(selected_states_data['Year'], selected_states_data[state], marker='o', label=state)
        
            plt.xlabel("Year", fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.title(title, fontsize=16)
            plt.legend(title="State", title_fontsize=12, fontsize=10)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(plt)
        else:
            st.write("Please select one or more states to see the chart.")

    def main():
        # List of all available states
        all_states = ['Alaska', 'Alabama', 'Arkansas', 'Arizona', 'California', 'Colorado', 'Connecticut', 'District of Columbia', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', 'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts', 'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri', 'Mississippi', 'Montana', 'North Carolina', 'North Dakota', 'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', 'Nevada', 'New York', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Virginia', 'Vermont', 'Washington', 'Wisconsin', 'West Virginia', 'Wyoming']
        
        # Create a single filter of states with default value as an empty list
        selected_states = st.multiselect("Select States:", all_states, default=[])
        
        # Create a separate filter for "9. CPI_cleaned.xlsx" dataset
        selected_cpi_states = st.multiselect("Select States for CPI (Cleaned) dataset:", all_states, default=[])
        
        # Plot individual graphs for each file
        plot_yearly_data("2. Inflows.xlsx", "Yearly Inflows by State", "Inflow", selected_states)
        plot_yearly_data("3. Outflows.xlsx", "Yearly Outflows by State", "Outflow", selected_states)
        plot_yearly_data("4. Netflows.xlsx", "Yearly Netflows by State", "Netflow", selected_states)
        plot_yearly_data("5. GDP_per_capita.xlsx", "Yearly GDP per Capita by State", "GDP per Capita", selected_states)
        plot_yearly_data("6. Household_income.xlsx", "Yearly Household Income by State", "Household Income", selected_states)
        plot_yearly_data("7. Crime.xlsx", "Yearly Crime by State", "Crime", selected_states)
        plot_yearly_data("8. Personal_income.xlsx", "Yearly Personal Income by State", "Personal Income", selected_states)
        
        # Exclude "9. CPI_cleaned.xlsx" dataset from the main filter
        if "9. CPI_cleaned.xlsx" not in selected_states:
            plot_yearly_data("9. CPI_cleaned.xlsx", "Yearly CPI (Cleaned) by State", "CPI", selected_cpi_states)
        
        plot_yearly_data("10. Education.xlsx", "Yearly Education by State", "Education", selected_states)
        plot_yearly_data("11. Health.xlsx", "Yearly Health by State", "Health", selected_states)
        plot_yearly_data("12. HDI.xlsx", "Yearly HDI by State", "HDI", selected_states)
        plot_yearly_data("13. Population.xlsx", "Yearly Population by State", "Population", selected_states)
        plot_yearly_data("14. Employment.xlsx", "Yearly Employment by State", "Employment", selected_states)
        # Add other datasets here...

    if __name__ == "__main__":
        main()
     

    def main():
        import matplotlib.dates as mdates
        st.title("Natural Disaster Data Analysis")

        # Read the data from the provided dataset
        df = pd.read_excel("Natural_disasters_all.xlsx")

        # Extract the list of states from column names
        all_states = list(set(col.split("_")[0] for col in df.columns if col != 'Date' and not col.startswith('_')))

        # Extract the list of features from column names
        all_features = list(set(col for col in df.columns if col != 'Date'))

        # Multi-select widget to choose states
        selected_states = st.multiselect("Select States:", all_states)

        # Multi-select widget to choose features
        selected_features = st.multiselect("Select Features:", all_features)

        if selected_states and selected_features:
            # Filter the data for the selected states and features
            selected_data = df[['Date'] + [col for col in df.columns if col.split("_")[0] in selected_states and col in selected_features]]

            # Convert the date strings to datetime objects for plotting
            selected_data['Date'] = pd.to_datetime(selected_data['Date'])

            # Plot the line chart
            plt.figure(figsize=(12, 6))
            for col in selected_data.columns[1:]:
                plt.plot_date(selected_data['Date'], selected_data[col], linestyle='-', marker='o', label=col)

            plt.xlabel("Date", fontsize=14)
            plt.ylabel("values", fontsize=14)
            plt.title("Natural Disaster Data Analysis", fontsize=16)
            plt.legend(title="State & Feature", title_fontsize=12, fontsize=10)
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.xticks(rotation=45, ha='right')
            plt.gca().autoscale(enable=True, axis='x', tight=True)
            st.pyplot(plt)
        else:
            st.write("Please select one or more states and features to see the chart.")

    if __name__ == "__main__":
        main()


    def main():

        import matplotlib.dates as mdates

        st.title("Weather Data Analysis")

        # Read the data from the provided dataset
        df = pd.read_excel("Weather_df_all.xlsx")

        # Extract the list of states from column names
        all_states = list(set(col.split("_")[0] for col in df.columns if col != 'Date' and not col.startswith('_')))

        # Extract the list of features from column names
        all_features = list(set(col for col in df.columns if col != 'Date'))

        # Multi-select widget to choose states
        selected_states = st.multiselect("Select States:", all_states)

        # Multi-select widget to choose features
        selected_features = st.multiselect("Select Features:", all_features)

        if selected_states and selected_features:
            # Filter the data for the selected states and features
            selected_data = df[['Date'] + [col for col in df.columns if col.split("_")[0] in selected_states and col in selected_features]]

            # Convert the date strings to numeric values for plotting
            selected_data['Date'] = pd.to_datetime(selected_data['Date'])
            selected_data['Date'] = selected_data['Date'].apply(mdates.date2num)

            # Plot the line chart
            plt.figure(figsize=(12, 6))
            for col in selected_data.columns[1:]:
                plt.plot_date(selected_data['Date'], selected_data[col], linestyle='-', marker='o', label=col)

            plt.xlabel("Date", fontsize=14)
            plt.ylabel("Value", fontsize=14)
            plt.title("Weather Data Analysis", fontsize=16)
            plt.legend(title="State & Feature", title_fontsize=12, fontsize=10)
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.xticks(rotation=45, ha='right')
            plt.gca().autoscale(enable=True, axis='x', tight=True)
            st.pyplot(plt)
        else:
            st.write("Please select one or more states and features to see the chart.")

    if __name__ == "__main__":
        main()






        

        # Rest of the historical data analysis charts go here...

# This function creates the Forecasting page
def forecasting():

    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt

    def plot_actual_vs_forecast(actual_file, forecast_file):
        st.title("Actual vs. Forecasted Data for California Net Flows")
        
        # Read the actual data from the provided dataset
        df_actual = pd.read_excel(actual_file)
        df_actual['Date'] = pd.to_datetime(df_actual['Date'])
        
        # Read the forecasted data from the provided dataset
        df_forecast = pd.read_excel(forecast_file)
        df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
        
        # Plot the actual vs. forecasted data
        plt.figure(figsize=(10, 6))
        plt.plot(df_actual['Date'], df_actual['D_net_flow_California'], color='black', marker='o', label='Actual')
        plt.plot(df_forecast['Date'], df_forecast['Prediction'], color='blue', marker='o', label='Forecast')
        
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Net Flows", fontsize=14)
        plt.title("Actual vs. Forecasted California Net Flows", fontsize=16)
        plt.legend(fontsize=12)
        plt.xticks(rotation=45)
        
        st.pyplot(plt)

    def main():
        # Read files and plot data
        actual_file = "California_neflows.xlsx"
        forecast_file = "California_neflows_predictions.xlsx"
        plot_actual_vs_forecast(actual_file, forecast_file)

    if __name__ == "__main__":
        main()







    # graph encoder variable importance 

    def main():
        st.title("Encoder Variable Importance")
        
        # Read the data from the provided dataset
        df_encoder = pd.read_excel("California encoder features_exclude_demographic.xlsx")
        
        # Sort the data by the "Value" column in descending order
        df_encoder_sorted = df_encoder.sort_values(by='Value', ascending=False)
        
        # Slider to choose the number of top encoder features to display
        num_features = st.slider("Select the number of top features to display:", min_value=1, max_value=len(df_encoder_sorted), value=10)
        
        # Select the top encoder features based on the slider value
        top_features = df_encoder_sorted.head(num_features)
        
        # Plot the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(top_features['Encoder features'], top_features['Value'])
        plt.xlabel("Encoder Features", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.title(f"Top {num_features} Encoder Variable Importance", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(plt)

    if __name__ == "__main__":
        main()




    # gaph decoder variable importance 

    def main():
        st.title("Decoder Variable Importance")
        
        # Read the data from the provided dataset
        df_decoder = pd.read_excel("California decoder features_exclude_demographic.xlsx")
        
        # Sort the data by the "Value" column in descending order
        df_decoder_sorted = df_decoder.sort_values(by='Value', ascending=False)
        
        # Slider to choose the number of top decoder features to display
        num_features = st.slider("Select the number of top features to display:", min_value=1, max_value=len(df_decoder_sorted), value=10)
        
        # Select the top decoder features based on the slider value
        top_features = df_decoder_sorted.head(num_features)
        
        # Plot the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(top_features['decoder features'], top_features['Value'])
        plt.xlabel("decoder Features", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.title(f"Top {num_features} Decoder Variable Importance", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(plt)

    if __name__ == "__main__":
        main()



    import streamlit as st
    import pandas as pd

    def main():
        st.title("California Feature Summary")
        
        # Read the data from the provided dataset
        df_feature_summary = pd.read_excel("California Feature summary_exclude_demographic.xlsx")
        
        # Display the data
        st.write("Displaying the data:")
        st.dataframe(df_feature_summary)

    if __name__ == "__main__":
        main()


        

def create_sidebar():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "Historical Data Analysis", "Forecasting"))

    if page == "Home":
        home()
    elif page == "Historical Data Analysis":
        historical_data_analysis()
    elif page == "Forecasting":
        forecasting()

def main():
    create_sidebar()

if __name__ == "__main__":
    main()
