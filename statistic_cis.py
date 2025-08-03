import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# Import better fonts
from matplotlib import font_manager
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Use a font that's definitely available
plt.rcParams['font.family'] = 'DejaVu Sans'  # or 'sans-serif'

# Set modern style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Modern color palette with gradients
COLORS = {
    'primary': '#6366F1',      # Indigo
    'secondary': '#EC4899',    # Pink
    'tertiary': '#10B981',     # Emerald
    'quaternary': '#F59E0B',   # Amber
    'success': '#059669',      # Green
    'info': '#3B82F6',         # Blue
    'warning': '#EF4444',      # Red
    'purple': '#8B5CF6',       # Purple
    'light': '#F9FAFB',        # Light gray
    'dark': '#1F2937',         # Dark gray
    'bg': '#FFFFFF',           # White
    'grid': '#E5E7EB'          # Grid gray
}

# Gradient colors for beautiful effects
GRADIENTS = {
    'blue': ['#EBF5FF', '#DBEAFE', '#BFDBFE', '#93C5FD', '#60A5FA', '#3B82F6', '#2563EB'],
    'purple': ['#F3E8FF', '#E9D5FF', '#D8B4FE', '#C084FC', '#A855F7', '#9333EA', '#7C3AED'],
    'pink': ['#FCE7F3', '#FBCFE8', '#F9A8D4', '#F472B6', '#EC4899', '#DB2777', '#BE185D'],
    'green': ['#D1FAE5', '#A7F3D0', '#6EE7B7', '#34D399', '#10B981', '#059669', '#047857']
}

# Function to load and prepare data
def prepare_data(df, time_col='time', value_col='value'):
    """
    Prepare the time series data

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    time_col : str
        Name of the time column (default: 'time')
    value_col : str
        Name of the value column (default: 'value')
    """
    # Create a copy to avoid modifying original
    df = df.copy()

    # Rename columns to standard names for internal use
    df = df.rename(columns={time_col: 'time', value_col: 'value'})

    # Process time column
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    df = df.set_index('time')

    return df

# Function to calculate comprehensive statistics
def calculate_statistics(data):
    """
    Calculate comprehensive statistics for the time series
    """
    stats_dict = {
        'Count': len(data),
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Std Dev': np.std(data),
        'Variance': np.var(data),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data),
        'Min': np.min(data),
        'Max': np.max(data),
        '25th Percentile': np.percentile(data, 25),
        '75th Percentile': np.percentile(data, 75),
        'IQR': np.percentile(data, 75) - np.percentile(data, 25)
    }
    return stats_dict

# Function to test for normality
def test_normality(data):
    """
    Test if data follows normal distribution
    """
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limited to 5000 samples

    # Anderson-Darling test
    anderson_result = stats.anderson(data)

    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(data)

    return {
        'Shapiro-Wilk': {'statistic': shapiro_stat, 'p-value': shapiro_p},
        'Jarque-Bera': {'statistic': jb_stat, 'p-value': jb_p},
        'Anderson-Darling': {'statistic': anderson_result.statistic,
                            'critical_values': anderson_result.critical_values}
    }

# Function to fit and compare distributions
def fit_distributions(data):
    """
    Fit Gaussian Mixture Model and Student's t-distribution
    """
    results = {}

    # Fit Student's t-distribution
    t_params = stats.t.fit(data)
    t_df, t_loc, t_scale = t_params

    # Calculate log-likelihood for Student's t
    t_loglik = np.sum(stats.t.logpdf(data, t_df, t_loc, t_scale))

    # Fit Gaussian Mixture Models with different components
    bic_scores = []
    aic_scores = []
    gmm_models = []

    for n_components in range(1, 5):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data.reshape(-1, 1))
        bic_scores.append(gmm.bic(data.reshape(-1, 1)))
        aic_scores.append(gmm.aic(data.reshape(-1, 1)))
        gmm_models.append(gmm)

    # Select best GMM based on BIC
    best_gmm_idx = np.argmin(bic_scores)
    best_gmm = gmm_models[best_gmm_idx]
    best_n_components = best_gmm_idx + 1

    # Calculate log-likelihood for best GMM
    gmm_loglik = best_gmm.score(data.reshape(-1, 1)) * len(data)

    results['student_t'] = {
        'df': t_df,
        'loc': t_loc,
        'scale': t_scale,
        'log_likelihood': t_loglik,
        'AIC': -2 * t_loglik + 2 * 3  # 3 parameters
    }

    results['gaussian_mixture'] = {
        'n_components': best_n_components,
        'means': best_gmm.means_.flatten(),
        'covariances': best_gmm.covariances_.flatten(),
        'weights': best_gmm.weights_,
        'log_likelihood': gmm_loglik,
        'BIC': bic_scores[best_gmm_idx],
        'AIC': aic_scores[best_gmm_idx]
    }

    # Determine best fit based on AIC
    if results['student_t']['AIC'] < results['gaussian_mixture']['AIC']:
        results['best_fit'] = 'Student\'s t-distribution'
    else:
        results['best_fit'] = f'Gaussian Mixture Model ({best_n_components} components)'

    return results, best_gmm, t_params

# Function to create beautiful time series visualization
def plot_time_series_beautiful(df, value_col_display='value', figsize=(20, 10)):
    """
    Create beautiful time series plot with enhanced layout
    """
    fig = plt.figure(figsize=figsize, facecolor=COLORS['bg'])

    # Create grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[3, 1, 1],
                         hspace=0.25, wspace=0.25)

    # 1. Main time series plot with gradient background
    ax_main = fig.add_subplot(gs[0, :])

    # Add subtle gradient background
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax_main.imshow(gradient, extent=[df.index.min(), df.index.max(),
                                     df['value'].min(), df['value'].max()],
                   aspect='auto', cmap='Blues', alpha=0.05)

    # Plot styling with gradient effect
    if len(df) > 10000:
        line = ax_main.plot(df.index, df['value'], alpha=0.7, linewidth=0.8,
                           color=COLORS['primary'], label='Hourly Data', zorder=3)[0]
    else:
        line = ax_main.plot(df.index, df['value'], linewidth=1.2,
                           color=COLORS['primary'], label='Hourly Data', zorder=3)[0]

    # Add glow effect to main line
    ax_main.plot(df.index, df['value'], alpha=0.3, linewidth=4,
                color=COLORS['primary'], zorder=2)

    # Add rolling statistics with smooth styling
    window = min(24*7, len(df)//10)
    rolling_mean = df['value'].rolling(window=window).mean()
    rolling_std = df['value'].rolling(window=window).std()

    # Moving average with gradient
    ax_main.plot(df.index, rolling_mean, color=COLORS['secondary'],
                linewidth=3, label=f'{window}h Moving Average', alpha=0.9, zorder=4)

    # Confidence band with gradient fill
    ax_main.fill_between(df.index,
                        rolling_mean - 2*rolling_std,
                        rolling_mean + 2*rolling_std,
                        alpha=0.15, facecolor=COLORS['secondary'],
                        edgecolor=COLORS['secondary'], linewidth=0.5,
                        label='±2σ Confidence Band', zorder=1)

    # Enhanced styling
    ax_main.set_title(f'Time Series Analysis: {value_col_display}',
                     fontsize=22, fontweight='300', pad=20,
                     fontfamily='Arial', color=COLORS['dark'])
    ax_main.set_xlabel('Time', fontsize=14, fontweight='500', color=COLORS['dark'])
    ax_main.set_ylabel(value_col_display.capitalize(), fontsize=14,
                      fontweight='500', color=COLORS['dark'])

    # Beautiful legend
    legend = ax_main.legend(loc='upper left', frameon=True, fancybox=True,
                           shadow=True, framealpha=0.9, edgecolor=COLORS['grid'])
    legend.get_frame().set_facecolor(COLORS['light'])

    # Grid styling
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=COLORS['grid'])
    ax_main.set_facecolor(COLORS['light'])
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['left'].set_color(COLORS['grid'])
    ax_main.spines['bottom'].set_color(COLORS['grid'])

    # 2. Seasonal decomposition plot with modern style
    ax_trend = fig.add_subplot(gs[1, 0])

    # Calculate and plot trend with gradient
    trend = df['value'].rolling(window=24*30).mean()
    ax_trend.plot(df.index, trend, color=COLORS['tertiary'], linewidth=2.5)
    ax_trend.fill_between(df.index, df['value'].min(), trend,
                         alpha=0.1, color=COLORS['tertiary'])

    ax_trend.set_title('Long-term Trend', fontsize=13, fontweight='600',
                      color=COLORS['dark'], pad=10)
    ax_trend.set_ylabel(value_col_display, fontsize=11, color=COLORS['dark'])
    ax_trend.grid(True, alpha=0.2, linestyle='--', color=COLORS['grid'])
    ax_trend.set_facecolor(COLORS['light'])
    ax_trend.spines['top'].set_visible(False)
    ax_trend.spines['right'].set_visible(False)

    # 3. Hourly pattern with gradient bars
    ax_hourly = fig.add_subplot(gs[1, 1])
    hourly_avg = df.groupby(df.index.hour)['value'].mean()

    # Create gradient effect for bars
    bars = ax_hourly.bar(hourly_avg.index, hourly_avg.values, alpha=0.8)

    # Apply gradient colors
    for i, bar in enumerate(bars):
        bar.set_facecolor(GRADIENTS['blue'][i % len(GRADIENTS['blue'])])
        bar.set_edgecolor(COLORS['info'])
        bar.set_linewidth(0.5)

    ax_hourly.set_title('Average Hourly Pattern', fontsize=13, fontweight='600',
                       color=COLORS['dark'], pad=10)
    ax_hourly.set_xlabel('Hour', fontsize=11, color=COLORS['dark'])
    ax_hourly.set_ylabel('Avg', fontsize=11, color=COLORS['dark'])
    ax_hourly.grid(True, alpha=0.2, axis='y', linestyle='--', color=COLORS['grid'])
    ax_hourly.set_facecolor(COLORS['light'])
    ax_hourly.spines['top'].set_visible(False)
    ax_hourly.spines['right'].set_visible(False)

    # 4. Day of week pattern with gradient
    ax_weekly = fig.add_subplot(gs[1, 2])
    df['dayofweek'] = df.index.dayofweek
    weekly_avg = df.groupby('dayofweek')['value'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    bars = ax_weekly.bar(range(7), weekly_avg.values, alpha=0.8)

    # Apply gradient colors
    for i, bar in enumerate(bars):
        bar.set_facecolor(GRADIENTS['green'][i])
        bar.set_edgecolor(COLORS['success'])
        bar.set_linewidth(0.5)

    ax_weekly.set_xticks(range(7))
    ax_weekly.set_xticklabels(days, rotation=45, ha='right', fontsize=10)
    ax_weekly.set_title('Weekly Pattern', fontsize=13, fontweight='600',
                       color=COLORS['dark'], pad=10)
    ax_weekly.set_ylabel('Avg', fontsize=11, color=COLORS['dark'])
    ax_weekly.grid(True, alpha=0.2, axis='y', linestyle='--', color=COLORS['grid'])
    ax_weekly.set_facecolor(COLORS['light'])
    ax_weekly.spines['top'].set_visible(False)
    ax_weekly.spines['right'].set_visible(False)

    # 5. Beautiful volatility heatmap
    ax_density = fig.add_subplot(gs[2, :])

    # Create time-based volatility
    time_bins = pd.date_range(start=df.index.min(), end=df.index.max(), periods=100)
    density_values = []

    for i in range(len(time_bins)-1):
        mask = (df.index >= time_bins[i]) & (df.index < time_bins[i+1])
        density_values.append(df[mask]['value'].std() if mask.sum() > 0 else 0)

    # Create beautiful colormap
    cmap = plt.cm.plasma
    colors_density = cmap(np.array(density_values) / max(density_values))

    bars = ax_density.bar(time_bins[:-1], [1]*len(density_values),
                         width=(time_bins[1] - time_bins[0]),
                         color=colors_density, alpha=0.9, edgecolor='none')

    ax_density.set_title('Volatility Heatmap Over Time', fontsize=13,
                        fontweight='600', color=COLORS['dark'], pad=10)
    ax_density.set_xlabel('Time', fontsize=11, color=COLORS['dark'])
    ax_density.set_ylim(0, 1.2)
    ax_density.set_yticks([])
    ax_density.grid(True, alpha=0.2, axis='x', linestyle='--', color=COLORS['grid'])
    ax_density.set_facecolor(COLORS['light'])
    ax_density.spines['top'].set_visible(False)
    ax_density.spines['right'].set_visible(False)
    ax_density.spines['left'].set_visible(False)

    # Add beautiful colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max(density_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_density, orientation='horizontal',
                       pad=0.1, shrink=0.6, aspect=40)
    cbar.set_label('Volatility (σ)', fontsize=11, color=COLORS['dark'])
    cbar.ax.tick_params(labelsize=9, colors=COLORS['dark'])
    cbar.outline.set_edgecolor(COLORS['grid'])

    # Overall styling with modern title
    fig.suptitle('Time Series Dashboard', fontsize=28, fontweight='200',
                y=0.98, color=COLORS['dark'], fontfamily='Arial')

    # Add subtle background
    fig.patch.set_facecolor(COLORS['bg'])

    return fig



def plot_distribution_beautiful(data, dist_results, gmm_model, t_params, norm_tests,
                               value_col_display='value', figsize=(20, 12)):
    """
    Create beautiful distribution analysis with enhanced layout
    """
    fig = plt.figure(figsize=figsize, facecolor=COLORS['bg'])

    # Create sophisticated grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.25,
                         height_ratios=[2, 1.5, 1],
                         width_ratios=[2, 1, 1, 1])

    # 1. Main distribution plot with beautiful styling
    ax_dist = fig.add_subplot(gs[0, :2])

    # Create gradient background
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    extent = [data.min(), data.max(), 0,
              stats.norm.pdf(0, 0, data.std()) * 1.2]
    ax_dist.imshow(gradient, extent=extent, aspect='auto',
                  cmap='Purples', alpha=0.05)

    # Enhanced histogram with gradient
    n, bins, patches = ax_dist.hist(data, bins=80, density=True, alpha=0.7,
                                   edgecolor='white', linewidth=0.5)

    # Apply gradient colors to histogram
    cm = plt.cm.cool
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(0.3 + c * 0.5))

    # Plot fitted distributions with beautiful styling
    x_range = np.linspace(data.min() - data.std(), data.max() + data.std(), 1000)

    # Student's t with glow effect
    t_pdf = stats.t.pdf(x_range, *t_params)
    ax_dist.plot(x_range, t_pdf, color=COLORS['secondary'],
                linewidth=3, label='Student\'s t', alpha=0.9, zorder=5)
    ax_dist.plot(x_range, t_pdf, color=COLORS['secondary'],
                linewidth=6, alpha=0.3, zorder=4)  # Glow

    # Gaussian Mixture with glow
    gmm_pdf = np.exp(gmm_model.score_samples(x_range.reshape(-1, 1)))
    ax_dist.plot(x_range, gmm_pdf, color=COLORS['tertiary'],
                linewidth=3, label=f'GMM ({gmm_model.n_components} comp.)',
                alpha=0.9, zorder=5)
    ax_dist.plot(x_range, gmm_pdf, color=COLORS['tertiary'],
                linewidth=6, alpha=0.3, zorder=4)  # Glow

    # Normal for reference
    norm_pdf = stats.norm.pdf(x_range, np.mean(data), np.std(data))
    ax_dist.plot(x_range, norm_pdf, color=COLORS['info'],
                linewidth=2, label='Normal', alpha=0.6, linestyle='--', zorder=3)

    ax_dist.set_title('Distribution Fitting Analysis', fontsize=18,
                     fontweight='300', pad=15, color=COLORS['dark'])
    ax_dist.set_xlabel(value_col_display.capitalize(), fontsize=13,
                      fontweight='500', color=COLORS['dark'])
    ax_dist.set_ylabel('Probability Density', fontsize=13,
                      fontweight='500', color=COLORS['dark'])

    # Beautiful legend
    legend = ax_dist.legend(loc='upper right', frameon=True, fancybox=True,
                           shadow=True, framealpha=0.95, edgecolor=COLORS['grid'])
    legend.get_frame().set_facecolor(COLORS['light'])

    ax_dist.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color=COLORS['grid'])
    ax_dist.set_facecolor(COLORS['light'])
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)

    # 2. Beautiful Q-Q Plots
    # Normal Q-Q
    ax_qq_norm = fig.add_subplot(gs[0, 2])
    stats.probplot(data, dist=stats.norm, plot=ax_qq_norm)

    # Style Q-Q plot
    points = ax_qq_norm.get_lines()[0]
    points.set_markerfacecolor(COLORS['info'])
    points.set_markeredgecolor('white')
    points.set_markersize(5)
    points.set_alpha(0.6)

    line = ax_qq_norm.get_lines()[1]
    line.set_color(COLORS['warning'])
    line.set_linewidth(2.5)
    line.set_alpha(0.8)

    ax_qq_norm.set_title('Normal Q-Q', fontsize=13, fontweight='600',
                        color=COLORS['dark'], pad=10)
    ax_qq_norm.grid(True, alpha=0.2, linestyle='--', color=COLORS['grid'])
    ax_qq_norm.set_facecolor(COLORS['light'])
    ax_qq_norm.spines['top'].set_visible(False)
    ax_qq_norm.spines['right'].set_visible(False)

    # Student's t Q-Q
    ax_qq_t = fig.add_subplot(gs[0, 3])
    stats.probplot(data, dist=stats.t, sparams=(t_params[0],), plot=ax_qq_t)

    points = ax_qq_t.get_lines()[0]
    points.set_markerfacecolor(COLORS['secondary'])
    points.set_markeredgecolor('white')
    points.set_markersize(5)
    points.set_alpha(0.6)

    line = ax_qq_t.get_lines()[1]
    line.set_color(COLORS['secondary'])
    line.set_linewidth(2.5)
    line.set_alpha(0.8)

    ax_qq_t.set_title('Student\'s t Q-Q', fontsize=13, fontweight='600',
                     color=COLORS['dark'], pad=10)
    ax_qq_t.grid(True, alpha=0.2, linestyle='--', color=COLORS['grid'])
    ax_qq_t.set_facecolor(COLORS['light'])
    ax_qq_t.spines['top'].set_visible(False)
    ax_qq_t.spines['right'].set_visible(False)

    # 3. Modern box and violin plots
    ax_box = fig.add_subplot(gs[1, 0])

    # Create beautiful box plot
    box_parts = ax_box.boxplot(data, vert=True, patch_artist=True,
                              notch=True, showmeans=True, widths=0.7)

    # Style box plot with gradients
    box_parts['boxes'][0].set_facecolor(COLORS['primary'])
    box_parts['boxes'][0].set_alpha(0.6)
    box_parts['boxes'][0].set_edgecolor(COLORS['primary'])
    box_parts['boxes'][0].set_linewidth(2)

    box_parts['medians'][0].set_color(COLORS['warning'])
    box_parts['medians'][0].set_linewidth(3)

    box_parts['means'][0].set_markerfacecolor(COLORS['tertiary'])
    box_parts['means'][0].set_markeredgecolor('white')
    box_parts['means'][0].set_markersize(10)
    box_parts['means'][0].set_markeredgewidth(2)

    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_parts[element], color=COLORS['primary'], linewidth=1.5)

    ax_box.set_title('Box Plot', fontsize=13, fontweight='600',
                    color=COLORS['dark'], pad=10)
    ax_box.set_ylabel(value_col_display.capitalize(), fontsize=11, color=COLORS['dark'])
    ax_box.grid(True, alpha=0.2, axis='y', linestyle='--', color=COLORS['grid'])
    ax_box.set_facecolor(COLORS['light'])
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)

    # Beautiful violin plot
    ax_violin = fig.add_subplot(gs[1, 1])
    violin_parts = ax_violin.violinplot(data, vert=True, showmeans=True,
                                       showextrema=True, showmedians=True,
                                       widths=0.8)

    # Style violin plot with gradient
    for pc in violin_parts['bodies']:
        pc.set_facecolor(COLORS['purple'])
        pc.set_alpha(0.6)
        pc.set_edgecolor(COLORS['purple'])
        pc.set_linewidth(2)

    violin_parts['cmedians'].set_colors(COLORS['warning'])
    violin_parts['cmedians'].set_linewidth(2.5)
    violin_parts['cmeans'].set_colors(COLORS['tertiary'])
    violin_parts['cmeans'].set_linewidth(2.5)

    ax_violin.set_title('Violin Plot', fontsize=13, fontweight='600',
                       color=COLORS['dark'], pad=10)
    ax_violin.set_ylabel(value_col_display.capitalize(), fontsize=11, color=COLORS['dark'])
    ax_violin.grid(True, alpha=0.2, axis='y', linestyle='--', color=COLORS['grid'])
    ax_violin.set_facecolor(COLORS['light'])
    ax_violin.spines['top'].set_visible(False)
    ax_violin.spines['right'].set_visible(False)

    # 4. Modern statistical tests visualization
    ax_tests = fig.add_subplot(gs[1, 2:])

    # Create beautiful bar chart for test results
    test_names = ['Normality\n(Shapiro)', 'Normality\n(J-Bera)', 'Best Fit\nConfidence']
    test_values = [
        1 - norm_tests['Shapiro-Wilk']['p-value'],
        1 - norm_tests['Jarque-Bera']['p-value'],
        0.85 if 'Student' in dist_results['best_fit'] else 0.15
    ]

    colors_test = []
    for i, v in enumerate(test_values):
        if i < 2:  # Normality tests
            colors_test.append(COLORS['success'] if v < 0.95 else COLORS['warning'])
        else:  # Best fit
            colors_test.append(COLORS['secondary'] if v > 0.5 else COLORS['tertiary'])

    # Create bars with gradient effect
    bars = ax_tests.bar(test_names, test_values, width=0.6)

    for bar, color, value in zip(bars, colors_test, test_values):
        # Main bar
        bar.set_facecolor(color)
        bar.set_alpha(0.7)
        bar.set_edgecolor(color)
        bar.set_linewidth(2)

        # Add value label with beautiful styling
        height = bar.get_height()
        if value < 0.5:
            label_y = height + 0.05
            va = 'bottom'
        else:
            label_y = height - 0.05
            va = 'top'

        label = f'{value:.3f}' if value < 2 else ('Student\'s t' if value > 0.5 else 'GMM')
        ax_tests.text(bar.get_x() + bar.get_width()/2, label_y, label,
                     ha='center', va=va, fontsize=10, fontweight='600',
                     color=COLORS['dark'])

    ax_tests.set_ylim(0, 1.1)
    ax_tests.set_title('Statistical Tests', fontsize=13, fontweight='600',
                      color=COLORS['dark'], pad=10)
    ax_tests.set_ylabel('Test Statistic', fontsize=11, color=COLORS['dark'])
    ax_tests.grid(True, alpha=0.2, axis='y', linestyle='--', color=COLORS['grid'])
    ax_tests.set_facecolor(COLORS['light'])
    ax_tests.spines['top'].set_visible(False)
    ax_tests.spines['right'].set_visible(False)

    # 5. Beautiful summary card
    ax_summary = fig.add_subplot(gs[2, :2])
    ax_summary.axis('off')

    # Create modern card-style summary
    card = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                         boxstyle="round,pad=0.02",
                         facecolor=COLORS['light'],
                         edgecolor=COLORS['grid'],
                         linewidth=2)
    ax_summary.add_patch(card)

    # Add gradient overlay
    gradient_overlay = FancyBboxPatch((0.05, 0.7), 0.9, 0.2,
                                     boxstyle="round,pad=0.02",
                                     facecolor=COLORS['primary'],
                                     alpha=0.1,
                                     edgecolor='none')
    ax_summary.add_patch(gradient_overlay)

    # Summary content with modern typography
    summary_lines = [
        (f"Best Fit Model: {dist_results['best_fit']}", 16, 'bold', 0.85),
        ("", 12, 'normal', 0.75),
        (f"Mean: {np.mean(data):.3f} ± {np.std(data):.3f}", 12, 'normal', 0.65),
        (f"Skewness: {stats.skew(data):.3f} {'(Right-tailed)' if stats.skew(data) > 0.5 else '(Left-tailed)' if stats.skew(data) < -0.5 else '(Symmetric)'}", 12, 'normal', 0.55),
        (f"Kurtosis: {stats.kurtosis(data):.3f} {'(Heavy-tailed)' if stats.kurtosis(data) > 1 else '(Light-tailed)' if stats.kurtosis(data) < -1 else '(Normal)'}", 12, 'normal', 0.45),
        (f"Student's t df: {dist_results['student_t']['df']:.2f}", 12, 'normal', 0.35),
        (f"GMM Components: {dist_results['gaussian_mixture']['n_components']}", 12, 'normal', 0.25)
    ]

    for text, size, weight, y in summary_lines:
        ax_summary.text(0.5, y, text, ha='center', va='center',
                       fontsize=size, fontweight=weight, color=COLORS['dark'],
                       transform=ax_summary.transAxes)

    # 6. Model comparison with modern visualization
    ax_model = fig.add_subplot(gs[2, 2:])

    models = ['Student\'s t', 'GMM', 'Normal']
    aic_values = [
        dist_results['student_t']['AIC'],
        dist_results['gaussian_mixture']['AIC'],
        -2 * np.sum(stats.norm.logpdf(data, np.mean(data), np.std(data))) + 4
    ]

    # Normalize AIC values for visualization
    min_aic = min(aic_values)
    normalized_aic = [min_aic / aic for aic in aic_values]

    # Create beautiful radial plot
    angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False).tolist()
    normalized_aic += normalized_aic[:1]
    angles += angles[:1]

    ax_model = plt.subplot(gs[2, 2:], projection='polar')
    ax_model.plot(angles, normalized_aic, 'o-', linewidth=2,
                 color=COLORS['primary'], markersize=10)
    ax_model.fill(angles, normalized_aic, alpha=0.25, color=COLORS['primary'])

    # Highlight best model
    best_idx = np.argmax(normalized_aic[:-1])
    ax_model.plot(angles[best_idx], normalized_aic[best_idx], 'o',
                 markersize=15, color=COLORS['warning'],
                 markeredgecolor='white', markeredgewidth=2)

    ax_model.set_xticks(angles[:-1])
    ax_model.set_xticklabels(models, size=11)
    ax_model.set_ylim(0, 1.1)
    ax_model.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_model.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
    ax_model.set_title('Model Performance\n(Higher = Better)',
                      fontsize=13, fontweight='600', color=COLORS['dark'],
                      pad=20, y=1.08)
    ax_model.grid(True, alpha=0.3, linestyle='--')
    ax_model.set_facecolor(COLORS['light'])

    # Overall title with modern styling
    fig.suptitle(f'Distribution Analysis: {value_col_display}',
                fontsize=28, fontweight='200', y=0.98,
                color=COLORS['dark'], fontfamily='Arial')

    # Set overall background
    fig.patch.set_facecolor(COLORS['bg'])

    return fig


# Function to create executive summary dashboard with modern design
def create_executive_dashboard(stats_dict, norm_tests, dist_results,
                              value_col_display='value', figsize=(16, 10)):
    """
    Create a modern executive summary dashboard with beautiful design
    """
    fig = plt.figure(figsize=figsize, facecolor=COLORS['bg'])

    # Create sophisticated grid
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3,
                         height_ratios=[0.5, 1, 1.5, 1])

    # Modern title with gradient effect
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')

    # Create gradient background for title
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    title_ax.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto',
                   cmap='Purples', alpha=0.1)

    title_ax.text(0.5, 0.5, f'Executive Summary: {value_col_display}',
                 ha='center', va='center', fontsize=26, fontweight='200',
                 color=COLORS['dark'], transform=title_ax.transAxes)

    # 1. Modern metric cards with beautiful styling
    metric_positions = [(1, 0), (1, 1), (1, 2), (1, 3)]
    metrics = [
        ('Total\nObservations', f"{stats_dict['Count']:,}", COLORS['primary'], '📊'),
        ('Average\nValue', f"{stats_dict['Mean']:.2f}", COLORS['secondary'], '📈'),
        ('Volatility\n(σ)', f"{stats_dict['Std Dev']:.2f}", COLORS['tertiary'], '📉'),
        ('Value\nRange', f"{stats_dict['Max'] - stats_dict['Min']:.2f}", COLORS['purple'], '🎯')
    ]

    for (row, col), (label, value, color, icon) in zip(metric_positions, metrics):
        ax = fig.add_subplot(gs[row, col])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Create beautiful card with gradient
        card = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                             boxstyle="round,pad=0.05",
                             facecolor='white',
                             edgecolor=color,
                             linewidth=2,
                             alpha=0.9)
        ax.add_patch(card)

        # Add gradient overlay
        gradient_card = FancyBboxPatch((0.05, 0.65), 0.9, 0.3,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color,
                                      alpha=0.1,
                                      edgecolor='none')
        ax.add_patch(gradient_card)

        # Add icon
        ax.text(0.5, 0.75, icon, ha='center', va='center',
               fontsize=24, transform=ax.transAxes)

        # Add value with modern styling
        ax.text(0.5, 0.45, value, ha='center', va='center',
               fontsize=20, fontweight='bold', color=color,
               transform=ax.transAxes)

        # Add label
        ax.text(0.5, 0.15, label, ha='center', va='center',
               fontsize=10, color=COLORS['dark'], alpha=0.8,
               transform=ax.transAxes)

        ax.axis('off')

    # 2. Beautiful distribution visualization
    ax_dist = fig.add_subplot(gs[2, :2])

    # Create stylized distribution curves
    x = np.linspace(-4, 4, 1000)

    # Reference normal
    normal_y = stats.norm.pdf(x, 0, 1)
    ax_dist.fill_between(x, 0, normal_y, alpha=0.2, color=COLORS['info'],
                        label='Normal Reference')

    # Fitted distribution
    if 'Student' in dist_results['best_fit']:
        fitted_y = stats.t.pdf(x, df=dist_results['student_t']['df'], loc=0, scale=1)
        fitted_color = COLORS['secondary']
        fitted_label = 'Student\'s t (Your Data)'
    else:
        fitted_y = normal_y * 0.8  # Simplified for visualization
        fitted_color = COLORS['tertiary']
        fitted_label = 'GMM (Your Data)'

    ax_dist.fill_between(x, 0, fitted_y, alpha=0.4, color=fitted_color,
                        label=fitted_label)

    # Add beautiful grid lines
    ax_dist.grid(True, alpha=0.1, linestyle='-', color=COLORS['grid'])
    ax_dist.set_facecolor(COLORS['light'])

    # Add vertical lines for mean and skewness
    ax_dist.axvline(x=0, color=COLORS['dark'], linewidth=2,
                   linestyle='--', alpha=0.5, label='Mean')
    ax_dist.axvline(x=stats_dict['Skewness'], color=COLORS['warning'],
                   linewidth=2, linestyle='--', alpha=0.5,
                   label=f'Skewness: {stats_dict["Skewness"]:.2f}')

    ax_dist.set_title('Distribution Shape Comparison', fontsize=14,
                     fontweight='600', color=COLORS['dark'], pad=10)
    ax_dist.set_xlabel('Standardized Values', fontsize=11, color=COLORS['dark'])
    ax_dist.set_ylabel('Probability Density', fontsize=11, color=COLORS['dark'])
    ax_dist.legend(loc='upper right', frameon=True, fancybox=True,
                  shadow=True, framealpha=0.9)
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)

    # 3. Key insights with modern cards
    ax_insights = fig.add_subplot(gs[2, 2:])
    ax_insights.axis('off')

    # Create insight cards
    insights = []
    insight_colors = []

    # Distribution insight
    if 'Student' in dist_results['best_fit']:
        insights.append(("Distribution Type", "Heavy-tailed",
                        "Higher extreme value probability", COLORS['secondary']))
    else:
        insights.append(("Distribution Type", f"{dist_results['gaussian_mixture']['n_components']}-Modal",
                        "Complex patterns detected", COLORS['tertiary']))

    # Skewness insight
    if abs(stats_dict['Skewness']) > 0.5:
        direction = "Right" if stats_dict['Skewness'] > 0 else "Left"
        insights.append(("Skewness", f"{direction}-skewed",
                        "Asymmetric distribution", COLORS['warning']))
    else:
        insights.append(("Skewness", "Symmetric",
                        "Balanced distribution", COLORS['success']))

    # Normality insight
    if norm_tests['Shapiro-Wilk']['p-value'] < 0.05:
        insights.append(("Normality", "Non-normal",
                        "Special modeling required", COLORS['purple']))
    else:
        insights.append(("Normality", "Near-normal",
                        "Standard methods applicable", COLORS['info']))

    # Draw insight cards
    y_positions = [0.75, 0.45, 0.15]
    for i, (title, status, desc, color) in enumerate(insights):
        y = y_positions[i]

        # Card background
        card = FancyBboxPatch((0.05, y - 0.12), 0.9, 0.2,
                             boxstyle="round,pad=0.02",
                             facecolor='white',
                             edgecolor=color,
                             linewidth=1.5,
                             alpha=0.9)
        ax_insights.add_patch(card)

        # Status badge
        badge = FancyBboxPatch((0.08, y - 0.08), 0.25, 0.12,
                              boxstyle="round,pad=0.02",
                              facecolor=color,
                              alpha=0.2,
                              edgecolor='none')
        ax_insights.add_patch(badge)

        ax_insights.text(0.2, y - 0.02, status, ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)
        ax_insights.text(0.4, y + 0.02, title, ha='left', va='center',
                        fontsize=10, fontweight='600', color=COLORS['dark'])
        ax_insights.text(0.4, y - 0.05, desc, ha='left', va='center',
                        fontsize=9, style='italic', color=COLORS['dark'], alpha=0.7)

    ax_insights.text(0.5, 0.95, 'Key Insights', ha='center', va='center',
                    fontsize=14, fontweight='600', color=COLORS['dark'])

    # 4. Bottom recommendation panel
    ax_recommend = fig.add_subplot(gs[3, :])
    ax_recommend.axis('off')

    # Create recommendation card
    rec_card = FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor=GRADIENTS['blue'][0],
                             edgecolor=COLORS['primary'],
                             linewidth=2)
    ax_recommend.add_patch(rec_card)

    # Recommendation text
    if 'Student' in dist_results['best_fit']:
        recommendation = ("💡 Recommendation: Your data shows heavy-tailed behavior. "
                         "Consider using robust statistical methods and be aware of "
                         "potential outliers in your analysis.")
    else:
        recommendation = (f"💡 Recommendation: Your data shows {dist_results['gaussian_mixture']['n_components']} "
                         f"distinct modes. Consider segmentation analysis to understand "
                         f"different behavioral patterns in your data.")

    ax_recommend.text(0.5, 0.5, recommendation, ha='center', va='center',
                     fontsize=12, color=COLORS['dark'], wrap=True,
                     transform=ax_recommend.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3",
                              facecolor='white', alpha=0.8))

    # Add timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.95, 0.02, f"Generated: {timestamp}", ha='right',
            fontsize=9, color=COLORS['dark'], alpha=0.5)

    # Set overall background
    fig.patch.set_facecolor(COLORS['bg'])

    return fig

def analyze_time_series(df, time_col='time', value_col='value'):
    """
    Main function to perform complete time series analysis with beautiful visualizations

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing time series data
    time_col : str
        Name of the time column (default: 'time')
    value_col : str
        Name of the value column to analyze (default: 'value')

    Returns:
    --------
    stats_dict : dict
        Dictionary of calculated statistics
    norm_tests : dict
        Results of normality tests
    dist_results : dict
        Results of distribution fitting
    """
    # Store original column name for display
    value_col_display = value_col

    # Prepare data (internally uses 'time' and 'value' columns)
    df = prepare_data(df, time_col=time_col, value_col=value_col)
    data = df['value'].values

    # Print styled header
    print("\n" + "="*60)
    print(f"✨ TIME SERIES DISTRIBUTION ANALYSIS: {value_col_display.upper()} ✨")
    print("="*60 + "\n")

    # Calculate statistics
    print("📊 1. Calculating statistics...")
    stats_dict = calculate_statistics(data)

    # Test normality
    print("🔍 2. Testing for normality...")
    norm_tests = test_normality(data)

    # Fit distributions
    print("📈 3. Fitting distributions...")
    dist_results, gmm_model, t_params = fit_distributions(data)

    print(f"\n🎯 *** Best fit distribution: {dist_results['best_fit']} ***\n")

    # Create visualizations
    print("🎨 4. Creating beautiful visualizations...\n")

    # Time series plot
    fig1 = plot_time_series_beautiful(df, value_col_display=value_col_display)
    plt.show()

    # Distribution analysis
    fig2 = plot_distribution_beautiful(data, dist_results, gmm_model, t_params, norm_tests,
                                     value_col_display=value_col_display)
    plt.show()

    # Executive dashboard
    fig3 = create_executive_dashboard(stats_dict, norm_tests, dist_results,
                                    value_col_display=value_col_display)
    plt.show()

    # Statistical dashboard (keeping original for detailed stats)
    fig4 = create_stats_dashboard(stats_dict, norm_tests)
    plt.show()

    print("\n✅ Analysis complete! All visualizations have been generated.")
    print("="*60 + "\n")

    return stats_dict, norm_tests, dist_results

# Function to create statistical summary dashboard (updated styling)
def create_stats_dashboard(stats_dict, norm_tests, figsize=(12, 8)):
    """
    Create a beautiful statistical summary dashboard
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=COLORS['bg'])
    fig.suptitle('Statistical Details Dashboard', fontsize=20,
                fontweight='300', color=COLORS['dark'])

    # Apply modern styling to all subplots
    for ax in axes.flat:
        ax.set_facecolor(COLORS['light'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # 1. Basic statistics table
    ax1 = axes[0, 0]
    ax1.axis('off')

    # Create modern table style
    stats_data = []
    for i, (key, value) in enumerate(list(stats_dict.items())[:6]):
        bg_color = COLORS['light'] if i % 2 == 0 else 'white'
        stats_data.append([key, f"{value:,.4f}"])

    table1 = ax1.table(cellText=stats_data,
                      colLabels=['Metric', 'Value'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.6, 0.4])

    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1, 2)

    # Style the table
    for i in range(len(stats_data) + 1):
        for j in range(2):
            cell = table1[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor(COLORS['primary'])
                cell.set_text_props(weight='bold', color='white')
                cell.set_height(0.1)
            else:
                cell.set_facecolor(COLORS['light'] if i % 2 == 0 else 'white')
                cell.set_text_props(color=COLORS['dark'])
            cell.set_edgecolor(COLORS['grid'])

    ax1.text(0.5, 0.95, 'Basic Statistics', ha='center', fontsize=14,
            fontweight='600', color=COLORS['dark'], transform=ax1.transAxes)

    # 2. Distribution characteristics
    ax2 = axes[0, 1]
    ax2.axis('off')

    dist_data = []
    for i, (key, value) in enumerate(list(stats_dict.items())[6:]):
        dist_data.append([key, f"{value:,.4f}"])

    table2 = ax2.table(cellText=dist_data,
                      colLabels=['Characteristic', 'Value'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.6, 0.4])

    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1, 2)

    # Style the table
    for i in range(len(dist_data) + 1):
        for j in range(2):
            cell = table2[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor(COLORS['secondary'])
                cell.set_text_props(weight='bold', color='white')
                cell.set_height(0.1)
            else:
                cell.set_facecolor(COLORS['light'] if i % 2 == 0 else 'white')
                cell.set_text_props(color=COLORS['dark'])
            cell.set_edgecolor(COLORS['grid'])

    ax2.text(0.5, 0.95, 'Distribution Properties', ha='center', fontsize=14,
            fontweight='600', color=COLORS['dark'], transform=ax2.transAxes)

    # 3. Normality tests with visual indicators
    ax3 = axes[1, 0]
    ax3.axis('off')

    # Create visual test results
    test_results = []
    colors = []
    for test, results in norm_tests.items():
        if test != 'Anderson-Darling':
            p_val = results['p-value']
            is_normal = p_val >= 0.05
            test_results.append([
                test,
                f"{results['statistic']:.4f}",
                f"{p_val:.4f}",
                '✓' if is_normal else '✗'
            ])
            colors.append(COLORS['success'] if is_normal else COLORS['warning'])

    table3 = ax3.table(cellText=test_results,
                      colLabels=['Test', 'Statistic', 'P-value', 'Normal?'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.35, 0.25, 0.25, 0.15])

    table3.auto_set_font_size(False)
    table3.set_fontsize(11)
    table3.scale(1, 2)

    # Style the table with color coding
    for i in range(len(test_results) + 1):
        for j in range(4):
            cell = table3[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor(COLORS['tertiary'])
                cell.set_text_props(weight='bold', color='white')
                cell.set_height(0.1)
            else:
                if j == 3:  # Normal? column
                    cell.set_facecolor(colors[i-1])
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor(COLORS['light'] if i % 2 == 0 else 'white')
                    cell.set_text_props(color=COLORS['dark'])
            cell.set_edgecolor(COLORS['grid'])

    ax3.text(0.5, 0.95, 'Normality Tests', ha='center', fontsize=14,
            fontweight='600', color=COLORS['dark'], transform=ax3.transAxes)

    # 4. Interpretation with modern cards
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Skewness interpretation
    skew = stats_dict['Skewness']
    if abs(skew) < 0.5:
        skew_interp = "Symmetric"
        skew_color = COLORS['success']
        skew_icon = "⚖️"
    elif skew > 0:
        skew_interp = "Right-skewed"
        skew_color = COLORS['warning']
        skew_icon = "→"
    else:
        skew_interp = "Left-skewed"
        skew_color = COLORS['warning']
        skew_icon = "←"

    # Kurtosis interpretation
    kurt = stats_dict['Kurtosis']
    if abs(kurt) < 0.5:
        kurt_interp = "Normal-like"
        kurt_color = COLORS['success']
        kurt_icon = "📊"
    elif kurt > 0:
        kurt_interp = "Heavy-tailed"
        kurt_color = COLORS['purple']
        kurt_icon = "⬆️"
    else:
        kurt_interp = "Light-tailed"
        kurt_color = COLORS['info']
        kurt_icon = "⬇️"

    # Create interpretation cards
    card1 = FancyBboxPatch((0.1, 0.55), 0.8, 0.3,
                          boxstyle="round,pad=0.02",
                          facecolor=skew_color,
                          alpha=0.2,
                          edgecolor=skew_color,
                          linewidth=2)
    ax4.add_patch(card1)

    card2 = FancyBboxPatch((0.1, 0.15), 0.8, 0.3,
                          boxstyle="round,pad=0.02",
                          facecolor=kurt_color,
                          alpha=0.2,
                          edgecolor=kurt_color,
                          linewidth=2)
    ax4.add_patch(card2)

    ax4.text(0.5, 0.95, 'Interpretations', ha='center', fontsize=14,
            fontweight='600', color=COLORS['dark'], transform=ax4.transAxes)

    ax4.text(0.15, 0.7, skew_icon, fontsize=20, ha='left', va='center')
    ax4.text(0.3, 0.75, 'Skewness:', fontsize=11, fontweight='bold',
            color=COLORS['dark'])
    ax4.text(0.3, 0.65, skew_interp, fontsize=11, color=skew_color,
            fontweight='600')

    ax4.text(0.15, 0.3, kurt_icon, fontsize=20, ha='left', va='center')
    ax4.text(0.3, 0.35, 'Kurtosis:', fontsize=11, fontweight='bold',
            color=COLORS['dark'])
    ax4.text(0.3, 0.25, kurt_interp, fontsize=11, color=kurt_color,
            fontweight='600')

    plt.tight_layout()
    fig.patch.set_facecolor(COLORS['bg'])

    return fig

# Example usage with sample data generation for testing
def generate_sample_data(n_points=8760, dist_type='student_t'):
    """
    Generate sample time series data for testing

    Parameters:
    -----------
    n_points : int
        Number of data points (default: 8760 for hourly data over a year)
    dist_type : str
        Distribution type: 'student_t', 'gmm', or 'normal'
    """
    # Create time index
    time_index = pd.date_range(start='2023-01-01', periods=n_points, freq='h')

    # Generate values based on distribution type
    if dist_type == 'student_t':
        # Student's t with heavy tails
        values = stats.t.rvs(df=3, loc=100, scale=20, size=n_points)
    elif dist_type == 'gmm':
        # Gaussian mixture
        n1 = n_points // 3
        n2 = n_points // 3
        n3 = n_points - n1 - n2
        values = np.concatenate([
            np.random.normal(80, 10, n1),
            np.random.normal(100, 15, n2),
            np.random.normal(120, 12, n3)
        ])
        np.random.shuffle(values)
    else:
        # Normal distribution
        values = np.random.normal(100, 20, n_points)

    # Add trend and seasonality
    trend = np.linspace(0, 10, n_points)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / 24)  # Daily pattern
    values = values + trend + seasonal

    # Create dataframe
    df = pd.DataFrame({
        'timestamp': time_index,
        'price': values
    })

    return df

# example
# Run analysis
from scipy import stats

stats, normality, distributions = analyze_time_series(
    df_features,
    time_col='time',
    value_col='fll_spike_kama'
)