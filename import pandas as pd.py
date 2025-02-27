import pandas as pd
import numpy as np
from scipy import stats, integrate, interpolate
import matplotlib.pyplot as plt

# -------------------------------
# Data Processor Class
# -------------------------------
class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        print("Data loaded successfully")
        print("Data columns:", self.data.columns.tolist())  # Debug column names
        print("Number of rows:", len(self.data))
    
    def filter_regular_season(self):
        # Fix: Use Regular_Season with underscore instead of Regular Season
        if 'Stage' not in self.data.columns:
            print("Error: 'Stage' column not found")
            print("Available columns:", self.data.columns.tolist())
            return self.data  # Return original data as fallback
            
        # Check what values are in the Stage column
        print("Values in Stage column:", self.data['Stage'].unique())
        
        # Fix: Use Regular_Season with underscore
        self.filtered_data = self.data[self.data['Stage'] == 'Regular_Season']
        print(f"Filtered data size: {len(self.filtered_data)} rows")
        
        # If no Regular Season data, return all data
        if len(self.filtered_data) == 0:
            print("Warning: No 'Regular_Season' data found, using all data")
            self.filtered_data = self.data
            
        return self.filtered_data

    def get_player_with_most_seasons(self):
        # Safety check
        if not hasattr(self, 'filtered_data') or len(self.filtered_data) == 0:
            print("No filtered data available")
            return "No player found"
            
        # Count unique seasons per player
        season_counts = self.filtered_data.groupby('Player')['Season'].nunique()
        print("Players and their season counts:")
        print(season_counts.sort_values(ascending=False).head())
        
        if len(season_counts) == 0:
            print("No players with seasons found")
            return "No player found"
            
        return season_counts.idxmax()

# -------------------------------
# Player Statistics Class
# -------------------------------
class PlayerStats:
    def __init__(self, player_name, player_data):
        self.player_name = player_name
        self.data = player_data.copy()
        
        # Check if necessary columns exist
        if '3PM' in self.data.columns and '3PA' in self.data.columns:
            # Calculate three point accuracy (%)
            self.data['ThreeP_Accuracy'] = self.data.apply(
                lambda row: (row['3PM'] / row['3PA'] * 100) 
                if row['3PA'] > 0 else np.nan, axis=1
            )
        else:
            print("Warning: 3-point columns not found. Available columns:", 
                  self.data.columns.tolist())
            self.data['ThreeP_Accuracy'] = np.nan
            
        # Extract season start year
        # Fix: Handle spaces in season format (e.g. "1999 - 2000")
        if 'Season' in self.data.columns:
            try:
                self.data['Season_Start'] = self.data['Season'].apply(
                    lambda x: int(x.split('-')[0].strip()) if isinstance(x, str) else np.nan
                )
                self.data.sort_values('Season_Start', inplace=True)
            except Exception as e:
                print(f"Error processing Season column: {e}")
                print("Sample Season values:", self.data['Season'].head())
                self.data['Season_Start'] = np.nan
        else:
            print("Warning: Season column not found")
            self.data['Season_Start'] = np.nan
    
    def get_accuracy_by_season(self):
        return self.data[['Season', 'Season_Start', 'ThreeP_Accuracy']]

    def interpolate_missing_seasons(self, missing_seasons):
        # Prepare data for interpolation using SciPy's interp1d.
        valid_data = self.data[['Season_Start', 'ThreeP_Accuracy']].dropna()
        x_vals = valid_data['Season_Start'].values
        y_vals = valid_data['ThreeP_Accuracy'].values
        
        # Create an interpolation function (extrapolation allowed)
        interp_func = interpolate.interp1d(x_vals, y_vals, kind='linear', fill_value='extrapolate')
        
        interpolated = {}
        for season in missing_seasons:
            start_year = int(season.split('-')[0])
            interpolated[season] = float(interp_func(start_year))
        return interpolated

# -------------------------------
# Regression Model Class (using SciPy)
# -------------------------------
class RegressionModel:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def fit(self):
        # Using SciPy's linregress for linear regression
        self.regression = stats.linregress(self.x, self.y)
        self.slope = self.regression.slope
        self.intercept = self.regression.intercept
        return self.slope, self.intercept  # Return these values

    def predict(self, x_val):
        # x_val can be a scalar or an array
        return self.slope * np.array(x_val) + self.intercept

    def integrate_fit_line(self, x_start, x_end):
        # Use SciPy's integrate.quad to integrate the regression line f(x)=m*x+b.
        func = lambda x: self.slope * x + self.intercept
        integral, _ = integrate.quad(func, x_start, x_end)
        average = integral / (x_end - x_start)
        return average

# -------------------------------
# Statistical Analyzer Class
# -------------------------------
class StatisticalAnalyzer:
    def __init__(self, data):
        self.fgm = data['FGM']
        self.fga = data['FGA']

    def descriptive_stats(self):
        stats_dict = {
            'FGM': {
                'mean': np.mean(self.fgm),
                'variance': np.var(self.fgm, ddof=1),
                'skew': stats.skew(self.fgm, bias=False),
                'kurtosis': stats.kurtosis(self.fgm, bias=False)
            },
            'FGA': {
                'mean': np.mean(self.fga),
                'variance': np.var(self.fga, ddof=1),
                'skew': stats.skew(self.fga, bias=False),
                'kurtosis': stats.kurtosis(self.fga, bias=False)
            }
        }
        return stats_dict

    def t_tests(self):
        paired = stats.ttest_rel(self.fgm, self.fga)
        independent = stats.ttest_ind(self.fgm, self.fga, equal_var=False)
        return {'paired': paired, 'independent': independent}

# -------------------------------
# Main Routine
# -------------------------------
def main():
    file_path = 'C:\\HW\\1\\data.csv'  # Update file path as needed
    dp = DataProcessor(file_path)
    filtered_data = dp.filter_regular_season()
    print("Filtered data:\n", filtered_data.head())  # Debugging statement

    # Identify player with most regular seasons
    top_player = dp.get_player_with_most_seasons()
    print("Player with most regular seasons:", top_player)

    # Extract data for the selected player
    player_data = filtered_data[filtered_data['Player'] == top_player]
    ps = PlayerStats(top_player, player_data)
    
    # Debug to check if columns were created
    print("\nPlayer data columns:", ps.data.columns.tolist())
    print("Season_Start exists:", 'Season_Start' in ps.data.columns)
    print("ThreeP_Accuracy exists:", 'ThreeP_Accuracy' in ps.data.columns)
    
    season_accuracy = ps.get_accuracy_by_season()
    print("\nSeason-by-season three point accuracy:")
    print(season_accuracy)

    # Continue with the rest of the main function as before
    # But ensure we're accessing columns that exist
    if 'Season_Start' in ps.data.columns and 'ThreeP_Accuracy' in ps.data.columns:
        reg_data = ps.data[['Season_Start', 'ThreeP_Accuracy']].dropna()
        
        if len(reg_data) > 1:  # Need at least 2 points for regression
            x = reg_data['Season_Start'].values
            y = reg_data['ThreeP_Accuracy'].values
            
            # Rest of the regression analysis
            reg_model = RegressionModel(x, y)
            slope, intercept = reg_model.fit()
            print("\nRegression line: Accuracy = {:.2f} * Season + {:.2f}".format(slope, intercept))
            
            # Integrated average accuracy from regression line using SciPy integration
            x_start = x.min()
            x_end = x.max()
            integrated_avg = reg_model.integrate_fit_line(x_start, x_end)
            print("\nIntegrated average three point accuracy (from regression): {:.2f}".format(integrated_avg))

            # Actual average three point accuracy
            actual_avg = np.mean(y)
            print("Actual average three point accuracy: {:.2f}".format(actual_avg))
            
            # Compare to average 3-pointers made
            avg_3p_made = player_data['3PM'].mean()
            print("Actual average 3-pointers made: {:.2f}".format(avg_3p_made))

            # Interpolate missing seasons: 2002-2003 and 2015-2016 using SciPy's interpolation
            missing_seasons = ["2002-2003", "2015-2016"]
            interpolated = ps.interpolate_missing_seasons(missing_seasons)
            print("\nInterpolated three point accuracy for missing seasons:")
            for season, acc in interpolated.items():
                print(f"{season}: {acc:.2f}")

            # Perform descriptive statistics and t-tests on FGM and FGA
            analyzer = StatisticalAnalyzer(filtered_data)
            desc_stats = analyzer.descriptive_stats()
            print("\nDescriptive statistics for FGM and FGA:")
            print(desc_stats)
            ttest_results = analyzer.t_tests()
            print("\nPaired t-test (FGM vs FGA):", ttest_results['paired'])
            print("Independent t-test (FGM vs FGA):", ttest_results['independent'])

            # Plot regression results
            plt.scatter(x, y, color='blue', label='Actual Accuracy')
            x_vals = np.linspace(x_start, x_end, 100)
            y_vals = reg_model.predict(x_vals)
            plt.plot(x_vals, y_vals, color='red', label='Regression Line')
            plt.xlabel('Season Start Year')
            plt.ylabel('Three Point Accuracy (%)')
            plt.title(f'Three Point Accuracy Trend for {top_player}')
            plt.legend()
            plt.show()

if __name__ == "__main__":
    main()
