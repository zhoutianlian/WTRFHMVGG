"""
Advanced Visualization Configuration

Professional financial/tech color scheme and styling configuration
optimized for dark theme interfaces and technical analysis.
"""

# Professional financial/tech color scheme
COLORS = {
    # Background colors
    'bg': '#0A0E27',           # Dark background
    'card_bg': '#1A1F3A',      # Card background  
    'surface': '#141B35',      # Surface color
    'border': '#2D3561',       # Border color
    
    # Primary accent colors
    'primary': '#00D4FF',      # Cyan accent (main brand)
    'secondary': '#7B61FF',    # Purple accent
    'tertiary': '#FF6B9D',     # Pink accent
    
    # Status colors
    'success': '#00F5A0',      # Green (bullish)
    'warning': '#FFB800',      # Orange (neutral)
    'danger': '#FF3366',       # Red (bearish)
    'info': '#3B82F6',         # Blue (informational)
    
    # Text colors
    'text': '#E4E7EB',         # Primary text (light)
    'text_secondary': '#9CA3AF', # Secondary text
    'text_muted': '#6B7280',   # Muted text
    'text_inverse': '#1F2937', # Inverse text (dark on light)
    
    # Grid and chart elements
    'grid': '#2D3561',         # Grid lines
    'axis': '#4B5563',         # Axis lines
    'hover': '#374151',        # Hover background
    
    # Gradient color sets
    'gradient_1': ['#00D4FF', '#7B61FF'],      # Cyan to Purple
    'gradient_2': ['#00F5A0', '#00D4FF'],      # Green to Cyan
    'gradient_3': ['#FFB800', '#FF3366'],      # Orange to Red
    'gradient_4': ['#7B61FF', '#FF6B9D'],      # Purple to Pink
    
    # Technical analysis colors
    'bullish': '#00F5A0',      # Bullish/uptrend
    'bearish': '#FF3366',      # Bearish/downtrend
    'neutral': '#FFB800',      # Neutral/sideways
    
    # Volume colors
    'volume_up': '#00F5A0',    # Volume on up days
    'volume_down': '#FF3366',  # Volume on down days
    
    # Indicator colors
    'ma_fast': '#00D4FF',      # Fast moving average
    'ma_slow': '#7B61FF',      # Slow moving average
    'signal_line': '#FFB800',  # Signal line
    'histogram': '#9CA3AF',    # Histogram bars
    
    # Bin/classification colors (9 levels)
    'bin_colors': [
        '#FF3366',  # Bin 0 - Extreme bearish
        '#FF4D7D',  # Bin 1
        '#FF6B9D',  # Bin 2
        '#FFB800',  # Bin 3
        '#FFC947',  # Bin 4 - Neutral
        '#FFDA6B',  # Bin 5
        '#7B61FF',  # Bin 6
        '#00D4FF',  # Bin 7
        '#00F5A0'   # Bin 8 - Extreme bullish
    ],
    
    # Correlation heatmap colors
    'correlation_positive': '#00F5A0',
    'correlation_negative': '#FF3366',
    'correlation_neutral': '#1A1F3A'
}

# Chart styling configuration
CHART_STYLE = {
    # Typography
    'font_family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'title_size': 24,
    'subtitle_size': 18,
    'label_size': 14,
    'tick_size': 12,
    'legend_size': 12,
    
    # Line styles
    'line_width': 2,
    'line_width_thin': 1,
    'line_width_thick': 3,
    
    # Marker styles
    'marker_size': 8,
    'marker_size_small': 4,
    'marker_size_large': 12,
    
    # Animation
    'animation_duration': 750,
    'transition_duration': 300,
    
    # Hover behavior
    'hover_mode': 'x unified',
    'hover_distance': 20,
    
    # Layout spacing
    'margin_top': 60,
    'margin_bottom': 60,
    'margin_left': 80,
    'margin_right': 80,
    'subplot_spacing': 0.08,
    
    # Grid styling
    'grid_width': 1,
    'grid_alpha': 0.3,
    
    # Chart dimensions
    'default_height': 500,
    'dashboard_height': 800,
    'thumbnail_height': 200,
    
    # Opacity levels
    'fill_alpha': 0.1,
    'area_alpha': 0.2,
    'overlay_alpha': 0.5,
    
    # Border radius for cards/containers
    'border_radius': 8,
    'border_width': 1
}

# Layout templates for different chart types
LAYOUT_TEMPLATES = {
    'dark_financial': {
        'template': 'plotly_dark',
        'plot_bgcolor': COLORS['bg'],
        'paper_bgcolor': COLORS['bg'],
        'font': {
            'family': CHART_STYLE['font_family'],
            'size': CHART_STYLE['label_size'],
            'color': COLORS['text']
        },
        'colorway': [
            COLORS['primary'],
            COLORS['secondary'], 
            COLORS['success'],
            COLORS['warning'],
            COLORS['danger'],
            COLORS['info'],
            COLORS['tertiary']
        ],
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': CHART_STYLE['grid_width'],
            'zerolinecolor': COLORS['axis'],
            'linecolor': COLORS['axis'],
            'tickcolor': COLORS['text_secondary'],
            'tickfont': {'size': CHART_STYLE['tick_size']},
            'titlefont': {'size': CHART_STYLE['label_size']}
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': CHART_STYLE['grid_width'],
            'zerolinecolor': COLORS['axis'],
            'linecolor': COLORS['axis'],
            'tickcolor': COLORS['text_secondary'],
            'tickfont': {'size': CHART_STYLE['tick_size']},
            'titlefont': {'size': CHART_STYLE['label_size']}
        },
        'hovermode': CHART_STYLE['hover_mode'],
        'hoverlabel': {
            'bgcolor': COLORS['card_bg'],
            'bordercolor': COLORS['border'],
            'font': {
                'size': CHART_STYLE['tick_size'],
                'family': CHART_STYLE['font_family'],
                'color': COLORS['text']
            }
        },
        'legend': {
            'font': {'size': CHART_STYLE['legend_size']},
            'bgcolor': 'rgba(0,0,0,0)',
            'bordercolor': COLORS['border'],
            'borderwidth': 1
        },
        'margin': {
            't': CHART_STYLE['margin_top'],
            'b': CHART_STYLE['margin_bottom'],
            'l': CHART_STYLE['margin_left'], 
            'r': CHART_STYLE['margin_right']
        }
    }
}

# Dash/HTML styling for dashboard components
DASH_STYLES = {
    'body': {
        'backgroundColor': COLORS['bg'],
        'fontFamily': CHART_STYLE['font_family'],
        'color': COLORS['text'],
        'margin': 0,
        'padding': 0
    },
    
    'container': {
        'maxWidth': '1400px',
        'margin': '0 auto',
        'padding': '20px'
    },
    
    'header': {
        'backgroundColor': COLORS['card_bg'],
        'padding': '20px',
        'marginBottom': '20px',
        'borderRadius': f"{CHART_STYLE['border_radius']}px",
        'border': f"{CHART_STYLE['border_width']}px solid {COLORS['border']}"
    },
    
    'card': {
        'backgroundColor': COLORS['card_bg'],
        'padding': '20px',
        'marginBottom': '20px',
        'borderRadius': f"{CHART_STYLE['border_radius']}px",
        'border': f"{CHART_STYLE['border_width']}px solid {COLORS['border']}",
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
    },
    
    'metric_card': {
        'backgroundColor': COLORS['card_bg'],
        'padding': '16px',
        'borderRadius': f"{CHART_STYLE['border_radius']}px",
        'border': f"{CHART_STYLE['border_width']}px solid {COLORS['border']}",
        'textAlign': 'center',
        'minHeight': '100px',
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center'
    },
    
    'sidebar': {
        'backgroundColor': COLORS['surface'],
        'padding': '20px',
        'height': '100vh',
        'borderRight': f"{CHART_STYLE['border_width']}px solid {COLORS['border']}"
    },
    
    'dropdown': {
        'backgroundColor': COLORS['surface'],
        'color': COLORS['text'],
        'border': f"{CHART_STYLE['border_width']}px solid {COLORS['border']}",
        'borderRadius': f"{CHART_STYLE['border_radius']}px"
    },
    
    'button_primary': {
        'backgroundColor': COLORS['primary'],
        'color': COLORS['text_inverse'],
        'border': 'none',
        'padding': '10px 20px',
        'borderRadius': f"{CHART_STYLE['border_radius']}px",
        'cursor': 'pointer',
        'fontSize': CHART_STYLE['label_size'],
        'fontWeight': '500'
    },
    
    'button_secondary': {
        'backgroundColor': 'transparent',
        'color': COLORS['text'],
        'border': f"{CHART_STYLE['border_width']}px solid {COLORS['border']}",
        'padding': '10px 20px',
        'borderRadius': f"{CHART_STYLE['border_radius']}px",
        'cursor': 'pointer',
        'fontSize': CHART_STYLE['label_size']
    },
    
    'table': {
        'backgroundColor': COLORS['card_bg'],
        'color': COLORS['text'],
        'width': '100%',
        'borderCollapse': 'collapse'
    },
    
    'table_header': {
        'backgroundColor': COLORS['surface'],
        'color': COLORS['text'],
        'padding': '12px',
        'textAlign': 'left',
        'borderBottom': f"{CHART_STYLE['border_width']}px solid {COLORS['border']}"
    },
    
    'table_cell': {
        'padding': '12px',
        'borderBottom': f"{CHART_STYLE['border_width']}px solid {COLORS['border']}"
    }
}

# Color utilities
def get_color_with_alpha(color: str, alpha: float) -> str:
    """Convert hex color to rgba with specified alpha"""
    if color.startswith('#'):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f'rgba({r},{g},{b},{alpha})'
    return color

def get_gradient_colors(start_color: str, end_color: str, steps: int) -> list:
    """Generate gradient colors between two colors"""
    # Simplified gradient generation
    return [start_color, end_color]  # For now, return start and end

def get_bin_color(bin_index: int, total_bins: int = 9) -> str:
    """Get color for specific bin index"""
    if 0 <= bin_index < len(COLORS['bin_colors']):
        return COLORS['bin_colors'][bin_index]
    return COLORS['neutral']

def get_trend_color(value: float, positive_threshold: float = 0) -> str:
    """Get color based on trend value"""
    if value > positive_threshold:
        return COLORS['bullish']
    elif value < -positive_threshold:
        return COLORS['bearish'] 
    else:
        return COLORS['neutral']