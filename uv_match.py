import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the two CSV files
# Adjust file paths as needed
outliers_file = '1525580010_sdp_l0-NGC641_02D02-0000-residual.fft.fits_outliers.csv'
antenna_file = '1525580010_sdp_l0-NGC641_02D03-MFS-image.fits.csv'

# Load the data
try:
    outliers_df = pd.read_csv(outliers_file)
    antenna_df = pd.read_csv(antenna_file)
    
    print("Outliers DataFrame Preview:")
    print(outliers_df.head())
    print("\nAntenna DataFrame Preview:")
    print(antenna_df.head())
    
    # Check column names to ensure we're working with the correct columns
    print("\nOutliers DataFrame Columns:", outliers_df.columns.tolist())
    print("Antenna DataFrame Columns:", antenna_df.columns.tolist())
    
    # Assuming the antenna_df has columns 'i', 'j', 'u_lambda', 'v_lambda'
    # and the outliers_df has 'u_lambda', 'v_lambda'
    
    # Create a function to find the closest antenna pair for each outlier point
    def find_matching_antenna_pair(outlier_row, antenna_df):
        # Extract the u_lambda and v_lambda from the outlier row
        u = outlier_row['u_lambda']
        v = outlier_row['v_lambda']
        
        # Calculate Euclidean distance to all antenna pairs
        distances = np.sqrt((antenna_df['u_lambda'] - u)**2 + (antenna_df['v_lambda'] - v)**2)
        
        # Find the index of the minimum distance
        min_idx = distances.idxmin()
        
        # Return the antenna pair info
        return antenna_df.loc[min_idx, ['i', 'j', 'u_lambda', 'v_lambda']]
    
    # Create a new dataframe to store the results
    result_data = []
    
    # Process each outlier
    for idx, row in outliers_df.iterrows():
        # Find the matching antenna pair
        antenna_match = find_matching_antenna_pair(row, antenna_df)
        
        # Combine the data
        combined_row = {
            'outlier_idx': idx,
            'outlier_u_lambda': row['u_lambda'],
            'outlier_v_lambda': row['v_lambda'],
            'antenna_i': antenna_match['i'],
            'antenna_j': antenna_match['j'],
            'antenna_u_lambda': antenna_match['u_lambda'],
            'antenna_v_lambda': antenna_match['v_lambda'],
            'distance': np.sqrt((row['u_lambda'] - antenna_match['u_lambda'])**2 + 
                                (row['v_lambda'] - antenna_match['v_lambda'])**2)
        }
        
        # Add any other columns from the outliers dataframe that you want to keep
        for col in outliers_df.columns:
            if col not in ['u_lambda', 'v_lambda']:
                combined_row[f'outlier_{col}'] = row[col]
        
        result_data.append(combined_row)
    
    # Create the result dataframe
    result_df = pd.DataFrame(result_data)
    
    # Sort by distance to see the best matches first
    result_df = result_df.sort_values('distance')
    
    print("\nMatching Results (sorted by distance):")
    print(result_df.head(10))
    
    # Save the results to a new CSV file
    result_file = 'matched_antenna_outliers.csv'
    result_df.to_csv(result_file, index=False)
    print(f"\nResults saved to {result_file}")
    
    # Optional: Create a visualization to verify the matches
    plt.figure(figsize=(10, 8))
    
    # Plot antenna pairs
    plt.scatter(antenna_df['u_lambda'], antenna_df['v_lambda'], 
                s=10, alpha=0.5, color='blue', label='Antenna Pairs')
    
    # Plot outliers
    plt.scatter(outliers_df['u_lambda'], outliers_df['v_lambda'], 
                s=30, alpha=0.8, color='red', label='Outliers')
    
    # Plot the matches
    for idx, row in result_df.head(20).iterrows():
        plt.plot([row['outlier_u_lambda'], row['antenna_u_lambda']], 
                 [row['outlier_v_lambda'], row['antenna_v_lambda']], 
                 'g-', alpha=0.3)
    
    plt.xlabel('u_lambda')
    plt.ylabel('v_lambda')
    plt.title('Matching Outliers to Antenna Pairs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('antenna_matching_visualization.png')
    plt.close()
    print("Visualization saved to 'antenna_matching_visualization.png'")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the file paths are correct.")
except Exception as e:
    print(f"An error occurred: {e}")
    
    # If the columns are named differently, provide some guidance
    if 'outliers_df' in locals() and 'antenna_df' in locals():
        print("\nColumn name suggestions:")
        print("Outliers DataFrame Columns:", outliers_df.columns.tolist())
        print("Antenna DataFrame Columns:", antenna_df.columns.tolist())
        print("\nYou may need to adjust the column names in the code.")
