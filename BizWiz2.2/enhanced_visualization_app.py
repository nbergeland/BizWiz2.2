# === COMPLETE ENHANCED MULTI-CITY VISUALIZATION APP ===
# Save this as: enhanced_visualization_app.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pickle
import os
import json
import traceback
import logging
from datetime import datetime
from city_config import CityConfigManager

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === ENHANCED DATA LOADER ===
class EnhancedDataLoader:
    """Enhanced data loader that handles multiple cities"""
    
    def __init__(self):
        self.city_manager = CityConfigManager()
        self.current_data = None
        self.current_city_id = None
        
    def load_city_data(self, city_id: str):
        """Load data for a specific city"""
        try:
            if city_id == self.current_city_id and self.current_data:
                return self.current_data
                
            cache_dir = f"cache_{city_id}"
            processed_data_file = os.path.join(cache_dir, 'processed_location_data.pkl')
            
            if not os.path.exists(processed_data_file):
                logger.error(f"No processed data found for {city_id}")
                return None
                
            with open(processed_data_file, 'rb') as f:
                data = pickle.load(f)
            
            self.current_data = data
            self.current_city_id = city_id
            logger.info(f"Loaded data for {city_id}: {len(data['df_filtered'])} locations")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {city_id}: {e}")
            return None
    
    def get_available_cities(self):
        """Get list of cities with available data"""
        available = []
        for city_id in self.city_manager.list_cities():
            cache_dir = f"cache_{city_id}"
            processed_data_file = os.path.join(cache_dir, 'processed_location_data.pkl')
            if os.path.exists(processed_data_file):
                config = self.city_manager.get_config(city_id)
                available.append({
                    'city_id': city_id,
                    'display_name': config.display_name,
                    'file_path': processed_data_file
                })
        return available

# Initialize data loader
data_loader = EnhancedDataLoader()
available_cities = data_loader.get_available_cities()

if not available_cities:
    print("❌ No processed data found for any cities!")
    print("Please run enhanced_data_collection.py first")
    print("Example: python enhanced_data_collection.py --city grand_forks_nd")
    exit(1)

# Load initial city data
initial_city = available_cities[0]['city_id']
current_data = data_loader.load_city_data(initial_city)

if not current_data:
    print("Failed to load initial city data")
    exit(1)

# === DASH APP ===
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Enhanced BizWizV2"

# === SAFE HELPER FUNCTIONS ===
def safe_filter_data(df, filters):
    """Safely apply filters with validation"""
    try:
        if df is None or len(df) == 0:
            return df
        
        # Check required columns exist
        required_cols = ['predicted_revenue', 'distance_to_chickfila', 
                        'commercial_traffic_score', 'fast_food_competition']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            return df
        
        # Apply filters safely with defaults
        min_revenue = filters.get('min_revenue', 0) or 0
        max_distance = filters.get('max_distance', 999) or 999
        min_traffic = filters.get('min_traffic', 0) or 0
        max_competition = filters.get('max_competition', 999) or 999
        zoning_filter = filters.get('zoning_filter', 'all') or 'all'
        
        # Apply filters step by step
        filtered = df[
            (df['predicted_revenue'] >= min_revenue) &
            (df['distance_to_chickfila'] <= max_distance) &
            (df['commercial_traffic_score'] >= min_traffic) &
            (df['fast_food_competition'] <= max_competition)
        ]
        
        # Apply zoning filter if column exists
        if zoning_filter == 'compliant' and 'zoning_compliant' in df.columns:
            filtered = filtered[filtered['zoning_compliant'] == 1]
        
        return filtered
        
    except Exception as e:
        logger.error(f"Filter error: {e}")
        return df

def create_map_tab_safe(data, filtered):
    """Create map tab with comprehensive error handling"""
    try:
        logger.info(f"Creating map with {len(filtered)} locations")
        
        # Validate filtered data
        if len(filtered) == 0:
            return html.Div([
                html.H4("🎯 No locations match current filters", className="text-warning"),
                html.P("Try adjusting the filter settings to see more locations:"),
                html.Ul([
                    html.Li("Lower the minimum revenue threshold"),
                    html.Li("Increase the competitor distance"),
                    html.Li("Lower the commercial traffic requirement"),
                    html.Li("Change zoning filter to 'All Locations'")
                ]),
                html.P(f"Total locations available: {len(data.get('df_filtered', []))}"),
                html.Button("Reset Filters", id="reset-filters-btn", className="btn btn-primary mt-3")
            ], className="text-center mt-5")
        
        # Check required columns
        required_cols = ['latitude', 'longitude', 'predicted_revenue']
        missing_cols = [col for col in required_cols if col not in filtered.columns]
        if missing_cols:
            return html.Div(f"❌ Missing required columns: {missing_cols}", className="text-center mt-5")
        
        # Remove invalid coordinates
        valid_coords = filtered[
            (filtered['latitude'].notna()) & 
            (filtered['longitude'].notna()) &
            (filtered['latitude'] != 0) &
            (filtered['longitude'] != 0) &
            (filtered['latitude'].between(-90, 90)) &
            (filtered['longitude'].between(-180, 180))
        ].copy()
        
        if len(valid_coords) == 0:
            return html.Div("❌ No valid coordinates found", className="text-center mt-5")
        
        logger.info(f"Using {len(valid_coords)} locations with valid coordinates")
        
        # Create the map
        city_config = data.get('city_config')
        city_name = city_config.display_name if city_config else 'Selected City'
        
        # Prepare hover data safely
        hover_data = {}
        if 'commercial_traffic_score' in valid_coords.columns:
            hover_data['commercial_traffic_score'] = ':.0f'
        if 'distance_to_chickfila' in valid_coords.columns:
            hover_data['distance_to_chickfila'] = ':.1f'
        if 'median_income' in valid_coords.columns:
            hover_data['median_income'] = ':$,.0f'
        
        # Create base scatter plot with updated Plotly syntax
        fig = px.scatter_map(
            valid_coords, 
            lat='latitude', 
            lon='longitude', 
            size='predicted_revenue', 
            color='predicted_revenue',
            color_continuous_scale='RdYlGn', 
            size_max=25, 
            zoom=11,
            map_style='open-street-map',
            hover_data=hover_data,
            title=f"🍗 Potential Raising Cane's Locations in {city_name}"
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate="<b>Potential Location</b><br>" +
                         "Revenue: $%{marker.color:,.0f}<br>" +
                         "Lat: %{lat:.4f}<br>" +
                         "Lon: %{lon:.4f}<br>" +
                         "<extra></extra>"
        )
        
        # Set map center
        center_lat = valid_coords['latitude'].mean()
        center_lon = valid_coords['longitude'].mean()
        
        # Set map center - updated for scatter_map
        fig.update_layout(
            map=dict(
                center=dict(lat=center_lat, lon=center_lon)
            ),
            height=700,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
        )
        
        # Add competitor locations if available
        chickfila_locations = data.get('chickfila_locations', [])
        if chickfila_locations and len(chickfila_locations) > 0:
            try:
                chickfila_df = pd.DataFrame(chickfila_locations, columns=['latitude', 'longitude'])
                competitor_name = city_config.competitor_data.primary_competitor.replace('-', ' ').title() if city_config else "Primary Competitor"
                
                fig.add_trace(
                    go.Scatter(
                        x=chickfila_df['longitude'],
                        y=chickfila_df['latitude'],
                        mode='markers+text',
                        marker=dict(size=20, color='red', symbol='circle'),
                        text='🐔',
                        textfont=dict(size=16),
                        textposition='middle center',
                        name=f"{competitor_name} Locations",
                        hovertemplate=f"<b>{competitor_name}</b><br>" +
                                     "Lat: %{y:.4f}<br>" +
                                     "Lon: %{x:.4f}<br>" +
                                     "<extra></extra>"
                    )
                )
                logger.info(f"Added {len(chickfila_locations)} competitor locations")
            except Exception as comp_error:
                logger.warning(f"Could not add competitor locations: {comp_error}")
        
        # Add existing Raising Cane's locations
        raising_canes_locations = data.get('raising_canes_locations', [])
        if raising_canes_locations and len(raising_canes_locations) > 0:
            try:
                canes_df = pd.DataFrame(raising_canes_locations, columns=['latitude', 'longitude', 'name'])
                
                fig.add_trace(
                    go.Scatter(
                        x=canes_df['longitude'],
                        y=canes_df['latitude'],
                        mode='markers+text',
                        marker=dict(size=20, color='purple', symbol='circle'),
                        text='🍗',
                        textfont=dict(size=16),
                        textposition='middle center',
                        name="Existing Raising Cane's",
                        hovertemplate="<b>Existing Raising Cane's</b><br>" +
                                     "Location: %{customdata}<br>" +
                                     "<extra></extra>",
                        customdata=canes_df['name']
                    )
                )
                logger.info(f"Added {len(raising_canes_locations)} existing Raising Cane's locations")
            except Exception as canes_error:
                logger.warning(f"Could not add Raising Cane's locations: {canes_error}")
        
        return dcc.Graph(figure=fig, style={'height': '80vh'})
        
    except Exception as e:
        logger.error(f"Map creation error: {e}")
        return html.Div([
            html.H4("❌ Error creating map", className="text-danger"),
            html.P(f"Error: {str(e)}"),
            html.P(f"Attempting to display {len(filtered)} locations"),
            html.Details([
                html.Summary("Technical Details"),
                html.Pre(traceback.format_exc(), style={'font-size': '12px'})
            ])
        ], className="text-center mt-5")

def create_analytics_tab_safe(data, filtered):
    """Create analytics dashboard with error handling"""
    try:
        if len(filtered) == 0:
            return html.Div("No data matches current filters for analytics", className="text-center mt-5")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Distribution', 'Commercial Traffic vs Revenue', 
                           'Competition Analysis', 'Demographics Overview'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Revenue distribution histogram
        fig.add_trace(
            go.Histogram(x=filtered['predicted_revenue'], nbinsx=20, name="Revenue Distribution"),
            row=1, col=1
        )
        
        # Commercial traffic vs revenue scatter
        if 'commercial_traffic_score' in filtered.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered['commercial_traffic_score'],
                    y=filtered['predicted_revenue'],
                    mode='markers',
                    name="Traffic vs Revenue",
                    hovertemplate="Traffic: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>"
                ),
                row=1, col=2
            )
        
        # Competition analysis
        if 'fast_food_competition' in filtered.columns:
            comp_analysis = filtered.groupby('fast_food_competition')['predicted_revenue'].agg(['mean', 'count']).reset_index()
            comp_analysis = comp_analysis[comp_analysis['count'] >= 3]  # Only show groups with 3+ locations
            
            if len(comp_analysis) > 0:
                fig.add_trace(
                    go.Bar(
                        x=comp_analysis['fast_food_competition'],
                        y=comp_analysis['mean'],
                        name="Avg Revenue by Competition",
                        hovertemplate="Competition Level: %{x}<br>Avg Revenue: $%{y:,.0f}<extra></extra>"
                    ),
                    row=2, col=1
                )
        
        # Demographics - income vs age colored by revenue
        if 'median_age' in filtered.columns and 'median_income' in filtered.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered['median_age'],
                    y=filtered['median_income'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=filtered['predicted_revenue'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Revenue")
                    ),
                    name="Demographics",
                    hovertemplate="Age: %{x}<br>Income: $%{y:,.0f}<br>Revenue: %{marker.color:$,.0f}<extra></extra>"
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="📊 Location Analytics Dashboard")
        
        return dcc.Graph(figure=fig)
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return html.Div([
            html.H4("❌ Analytics Error", className="text-danger"),
            html.P(f"Error: {str(e)}"),
            html.P("Basic analytics:"),
            html.P(f"Locations: {len(filtered)}"),
            html.P(f"Avg Revenue: ${filtered['predicted_revenue'].mean():,.0f}" if len(filtered) > 0 else "No data")
        ])

def create_top_locations_tab_safe(data, filtered):
    """Create top locations analysis table with error handling"""
    try:
        if len(filtered) == 0:
            return html.Div("No data matches current filters", className="text-center mt-5")
        
        # Get top 20 locations
        display_cols = ['latitude', 'longitude', 'predicted_revenue']
        
        # Add optional columns if they exist
        optional_cols = ['commercial_traffic_score', 'road_accessibility_score', 
                        'distance_to_chickfila', 'fast_food_competition', 
                        'median_income', 'population', 'zoning_compliant']
        
        for col in optional_cols:
            if col in filtered.columns:
                display_cols.append(col)
        
        top_locations = filtered.nlargest(20, 'predicted_revenue')[display_cols].round(4)
        
        # Format for display
        display_df = top_locations.copy()
        display_df['predicted_revenue'] = display_df['predicted_revenue'].apply(lambda x: f"${x:,.0f}")
        
        if 'median_income' in display_df.columns:
            display_df['median_income'] = display_df['median_income'].apply(lambda x: f"${x:,.0f}")
        if 'population' in display_df.columns:
            display_df['population'] = display_df['population'].apply(lambda x: f"{x:,.0f}")
        if 'distance_to_chickfila' in display_df.columns:
            display_df['distance_to_chickfila'] = display_df['distance_to_chickfila'].apply(lambda x: f"{x:.1f} mi")
        if 'zoning_compliant' in display_df.columns:
            display_df['zoning_compliant'] = display_df['zoning_compliant'].apply(lambda x: "✅" if x else "❌")
        
        # Rename columns for display
        column_names = {
            'latitude': 'Latitude',
            'longitude': 'Longitude', 
            'predicted_revenue': 'Predicted Revenue',
            'commercial_traffic_score': 'Commercial Score',
            'road_accessibility_score': 'Road Access',
            'distance_to_chickfila': 'Distance to Competitor',
            'fast_food_competition': 'Competition Level',
            'median_income': 'Median Income',
            'population': 'Population',
            'zoning_compliant': 'Zoning OK'
        }
        
        display_df = display_df.rename(columns=column_names)
        
        table = dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in display_df.columns],
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'fontFamily': 'Arial',
                'fontSize': '14px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 0},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                }
            ],
            page_size=20,
            sort_action="native",
            style_table={'overflowX': 'auto'}
        )
        
        return html.Div([
            html.H4("🏆 Top Revenue Potential Locations", className="mb-3"),
            html.P(f"Showing top {min(20, len(filtered))} locations sorted by predicted revenue potential"),
            table
        ])
        
    except Exception as e:
        logger.error(f"Top locations error: {e}")
        return html.Div(f"❌ Error creating top locations table: {e}")

def create_model_tab_safe(data):
    """Create model performance analysis tab with error handling"""
    try:
        metrics = data.get('metrics', {})
        feature_importance = data.get('feature_importance')
        
        if not metrics:
            return html.Div("No model metrics available", className="text-center mt-5")
        
        # Model performance metrics
        metrics_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{metrics.get('train_r2', 0):.3f}", className="text-primary"),
                        html.P("R² Score", className="text-muted mb-0")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${metrics.get('cv_mae_mean', 0):,.0f}", className="text-success"),
                        html.P("CV Mean Absolute Error", className="text-muted mb-0")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${metrics.get('train_rmse', 0):,.0f}", className="text-warning"),
                        html.P("Root Mean Square Error", className="text-muted mb-0")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{metrics.get('feature_count', 0)}", className="text-info"),
                        html.P("Features Used", className="text-muted mb-0")
                    ])
                ])
            ], width=3)
        ])
        
        # Feature importance chart
        feature_chart = html.Div("Feature importance data not available")
        if feature_importance is not None and len(feature_importance) > 0:
            try:
                top_features = feature_importance.head(15)
                fig = px.bar(
                    top_features,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Most Important Features",
                    labels={'importance': 'Feature Importance', 'feature': 'Feature'},
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                feature_chart = dcc.Graph(figure=fig)
            except Exception as feat_error:
                logger.warning(f"Feature importance chart error: {feat_error}")
        
        # Model parameters
        best_params = metrics.get('best_parameters', {})
        params_table = html.Div([
            html.H5("🎯 Optimal Model Parameters"),
            html.Ul([
                html.Li(f"{param}: {value}") for param, value in best_params.items()
            ]) if best_params else html.P("No parameter data available")
        ])
        
        return html.Div([
            html.H4("🔬 Model Performance Analysis", className="mb-4"),
            metrics_cards,
            html.Hr(className="my-4"),
            dbc.Row([
                dbc.Col([feature_chart], width=8),
                dbc.Col([params_table], width=4)
            ])
        ])
        
    except Exception as e:
        logger.error(f"Model tab error: {e}")
        return html.Div(f"❌ Model analysis error: {e}")

# === APP LAYOUT ===
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🍗 Enhanced Raising Cane's Location Analysis", className="text-center mb-3"),
            html.P("Multi-City Commercial Location Intelligence System", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # City Selection and Metrics Row
    dbc.Row([
        dbc.Col([
            html.Label("Select City:", className="fw-bold"),
            dcc.Dropdown(
                id='city-dropdown',
                options=[
                    {'label': city['display_name'], 'value': city['city_id']} 
                    for city in available_cities
                ],
                value=initial_city,
                clearable=False
            )
        ], width=4),
        
        dbc.Col([
            html.Div(id='city-metrics', className="text-center")
        ], width=8)
    ], className="mb-4"),
    
    # Main content
    dbc.Row([
        # Filters Column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎯 Analysis Filters", className="mb-0")),
                dbc.CardBody([
                    html.Label("Minimum Predicted Revenue:", className="fw-bold"),
                    dcc.Slider(
                        id='revenue-slider', 
                        min=0, 
                        max=100000,
                        step=1000, 
                        value=20000,  # Relaxed default
                        tooltip={"placement": "bottom", "always_visible": True},
                        marks={}
                    ),
                    html.Br(),
                    
                    html.Label("Maximum Distance to Primary Competitor (miles):", className="fw-bold"),
                    dcc.Slider(
                        id='competitor-distance-slider', 
                        min=0, 
                        max=20, 
                        step=1, 
                        value=15,  # Relaxed default
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    
                    html.Label("Minimum Commercial Traffic Score:", className="fw-bold"),
                    dcc.Slider(
                        id='commercial-traffic-slider', 
                        min=0, 
                        max=200,
                        step=5, 
                        value=5,  # Relaxed default
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    
                    html.Label("Maximum Fast Food Competition:", className="fw-bold"),
                    dcc.Slider(
                        id='competition-slider', 
                        min=0, 
                        max=20, 
                        step=1, 
                        value=15,  # Relaxed default
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    
                    html.Label("Zoning Compliance:", className="fw-bold"),
                    dcc.RadioItems(
                        id='zoning-radio', 
                        options=[
                            {'label': 'All Locations', 'value': 'all'}, 
                            {'label': 'Only Compliant', 'value': 'compliant'}
                        ], 
                        value='all',  # Relaxed default
                        className="mb-3"
                    ),
                    
                    html.Label("Model Performance:", className="fw-bold"),
                    html.Div(id='model-score-display', className="text-muted mb-3"),
                    
                    html.Hr(),
                    html.Div(id='location-stats', className="mt-3")
                ])
            ])
        ], width=3),
        
        # Map and Analysis Column
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="📍 Location Map", tab_id="map-tab"),
                dbc.Tab(label="📊 Analytics Dashboard", tab_id="analytics-tab"),
                dbc.Tab(label="🏆 Top Locations", tab_id="top-locations-tab"),
                dbc.Tab(label="🔬 Model Performance", tab_id="model-tab")
            ], id="main-tabs", active_tab="map-tab"),
            
            html.Div(id='tab-content', style={'height': '85vh'})
        ], width=9)
    ])
], fluid=True)

# === CALLBACK FUNCTIONS ===

@app.callback(
    [Output('revenue-slider', 'max'),
     Output('revenue-slider', 'value'),
     Output('revenue-slider', 'marks'),
     Output('commercial-traffic-slider', 'max'),
     Output('city-metrics', 'children'),
     Output('model-score-display', 'children')],
    [Input('city-dropdown', 'value')]
)
def update_city_data(city_id):
    """Update all components when city changes"""
    try:
        data = data_loader.load_city_data(city_id)
        
        if not data:
            return 100000, 20000, {}, 200, "No data available", "No model data"
        
        df = data['df_filtered']
        metrics = data.get('metrics', {})
        city_config = data.get('city_config')
        
        # Update slider ranges with more conservative defaults
        max_revenue = int(df['predicted_revenue'].max())
        
        # Create revenue marks here instead of separate callback
        revenue_marks = {
            0: '$0',
            max_revenue//4: f'${max_revenue//4:,.0f}',
            max_revenue//2: f'${max_revenue//2:,.0f}',
            3*max_revenue//4: f'${3*max_revenue//4:,.0f}',
            max_revenue: f'${max_revenue:,.0f}'
        }
        
        max_commercial = max(int(df['commercial_traffic_score'].max()), 200)
        initial_revenue = max(int(df['predicted_revenue'].quantile(0.2)), 10000)  # More lenient
        
        # City metrics
        city_name = city_config.display_name if city_config else city_id
        city_metrics = dbc.Row([
            dbc.Col([
                html.H6("📍 Locations Analyzed", className="text-muted mb-1"),
                html.H4(f"{len(df):,}", className="text-primary mb-0")
            ], width=2),
            dbc.Col([
                html.H6("💰 Avg Revenue Potential", className="text-muted mb-1"),
                html.H4(f"${df['predicted_revenue'].mean():,.0f}", className="text-success mb-0")
            ], width=3),
            dbc.Col([
                html.H6("🏆 Top Location Revenue", className="text-muted mb-1"),
                html.H4(f"${df['predicted_revenue'].max():,.0f}", className="text-warning mb-0")
            ], width=3),
            dbc.Col([
                html.H6("🎯 Competitors Mapped", className="text-muted mb-1"),
                html.H4(f"{len(data.get('chickfila_locations', []))}", className="text-info mb-0")
            ], width=2),
            dbc.Col([
                html.H6("🍗 Existing Cane's", className="text-muted mb-1"),
                html.H4(f"{len(data.get('raising_canes_locations', []))}", className="text-danger mb-0")
            ], width=2)
        ])
        
        # Model performance
        if metrics:
            model_score = f"R² Score: {metrics.get('train_r2', 0):.3f} | CV MAE: ${metrics.get('cv_mae_mean', 0):,.0f}"
        else:
            model_score = "Model metrics not available"
        
        return max_revenue, initial_revenue, revenue_marks, max_commercial, city_metrics, model_score
        
    except Exception as e:
        logger.error(f"Error updating city data: {e}")
        return 100000, 20000, {}, 200, f"Error loading data: {e}", "Error loading model data"

@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'active_tab'),
     Input('city-dropdown', 'value'),
     Input('revenue-slider', 'value'), 
     Input('competitor-distance-slider', 'value'), 
     Input('commercial-traffic-slider', 'value'),
     Input('competition-slider', 'value'),
     Input('zoning-radio', 'value')],
    prevent_initial_call=False
)
def update_tab_content(active_tab, city_id, min_revenue, max_competitor_distance, 
                      min_commercial_traffic, max_competition, zoning_filter):
    """Update tab content based on active tab and filters - ERROR HANDLED VERSION"""
    
    try:
        logger.info(f"Updating tab content: {active_tab}, city: {city_id}")
        logger.info(f"Filters: revenue>={min_revenue}, dist<={max_competitor_distance}, traffic>={min_commercial_traffic}")
        
        # Validate inputs
        if not city_id:
            logger.warning("No city_id provided")
            return html.Div("Please select a city", className="text-center mt-5")
        
        if not active_tab:
            logger.warning("No active_tab provided")
            return html.Div("Please select a tab", className="text-center mt-5")
        
        # Load data with error handling
        data = data_loader.load_city_data(city_id)
        if not data:
            logger.error(f"No data available for {city_id}")
            return html.Div([
                html.H4("❌ No data available", className="text-danger"),
                html.P(f"No processed data found for {city_id}"),
                html.P("Please run data collection first:"),
                html.Code(f"python enhanced_data_collection.py --city {city_id}")
            ], className="text-center mt-5")
        
        df = data['df_filtered']
        if df is None or len(df) == 0:
            logger.error(f"Empty dataframe for {city_id}")
            return html.Div("No location data found", className="text-center mt-5")
        
        logger.info(f"Loaded {len(df)} locations for {city_id}")
        
        # Apply filters with safe defaults
        filters = {
            'min_revenue': min_revenue or 0,
            'max_distance': max_competitor_distance or 999,
            'min_traffic': min_commercial_traffic or 0,
            'max_competition': max_competition or 999,
            'zoning_filter': zoning_filter or 'all'
        }
        
        filtered = safe_filter_data(df, filters)
        logger.info(f"Applied filters: {len(filtered)} locations remain")
        
        # Generate tab content based on active tab
        if active_tab == "map-tab":
            return create_map_tab_safe(data, filtered)
        elif active_tab == "analytics-tab":
            return create_analytics_tab_safe(data, filtered)
        elif active_tab == "top-locations-tab":
            return create_top_locations_tab_safe(data, filtered)
        elif active_tab == "model-tab":
            return create_model_tab_safe(data)
        else:
            return html.Div(f"Unknown tab: {active_tab}", className="text-center mt-5")
    
    except Exception as e:
        logger.error(f"Callback error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return detailed error information for debugging
        return html.Div([
            html.H4("⚠️ Error Loading Content", className="text-danger"),
            html.P(f"Error: {str(e)}"),
            html.P(f"Tab: {active_tab}"),
            html.P(f"City: {city_id}"),
            html.P("This error has been logged. Please check your data files and try again."),
            html.Details([
                html.Summary("Technical Details (for debugging)"),
                html.Pre(traceback.format_exc(), style={'font-size': '11px', 'background': '#f8f9fa', 'padding': '10px'})
            ]),
            html.Hr(),
            html.P("Troubleshooting steps:"),
            html.Ol([
                html.Li("Check if data collection has been run for this city"),
                html.Li("Verify that all required columns exist in the dataset"),
                html.Li("Try selecting a different city"),
                html.Li("Clear browser cache and refresh the page")
            ])
        ], className="m-3")

@app.callback(
    Output('location-stats', 'children'),
    [Input('city-dropdown', 'value'),
     Input('revenue-slider', 'value'), 
     Input('competitor-distance-slider', 'value'), 
     Input('commercial-traffic-slider', 'value'),
     Input('competition-slider', 'value'),
     Input('zoning-radio', 'value')]
)
def update_location_stats(city_id, min_revenue, max_competitor_distance, 
                         min_commercial_traffic, max_competition, zoning_filter):
    """Update location statistics sidebar"""
    try:
        data = data_loader.load_city_data(city_id)
        if not data:
            return html.Div("No data available")
        
        df = data['df_filtered']
        chickfila_locations = data.get('chickfila_locations', [])
        raising_canes_locations = data.get('raising_canes_locations', [])
        
        # Apply filters
        filters = {
            'min_revenue': min_revenue or 0,
            'max_distance': max_competitor_distance or 999,
            'min_traffic': min_commercial_traffic or 0,
            'max_competition': max_competition or 999,
            'zoning_filter': zoning_filter or 'all'
        }
        
        filtered = safe_filter_data(df, filters)
        
        if len(filtered) > 0:
            best = filtered.loc[filtered['predicted_revenue'].idxmax()]
            avg_revenue = filtered['predicted_revenue'].mean()
            
            # Build stats with safe field access
            stats_items = [
                html.H5("📊 Analysis Summary", className="text-primary"),
                html.P(f"Filtered Locations: {len(filtered):,}"),
                html.P(f"Average Revenue: ${avg_revenue:,.0f}"),
                html.P(f"Competitors: {len(chickfila_locations)}"),
                html.P(f"Existing Cane's: {len(raising_canes_locations)}"),
                html.Hr(),
                html.H5("🎯 Top Location", className="text-success"),
                html.P(f"📍 {best['latitude']:.4f}, {best['longitude']:.4f}"),
                html.P(f"💰 Revenue: ${best['predicted_revenue']:,.0f}")
            ]
            
            # Add optional fields if they exist
            optional_fields = [
                ('commercial_traffic_score', '🏪 Commercial Score', ':.0f'),
                ('road_accessibility_score', '🛣️ Road Access', ':.0f'),
                ('gas_station_proximity', '⛽ Gas Proximity', ':.0f'),
                ('distance_to_chickfila', '🎯 Competitor Distance', ':.1f mi'),
                ('fast_food_competition', '🏢 Competition', ':.0f'),
                ('median_age', '👥 Median Age', ':.0f'),
                ('median_income', '💵 Median Income', ':$,.0f'),
                ('zoning_compliant', '🏠 Zoning', None)
            ]
            
            for field, label, fmt in optional_fields:
                if field in best.index:
                    if field == 'zoning_compliant':
                        value = '✅' if best[field] else '❌'
                        stats_items.append(html.P(f"{label}: {value}"))
                    elif fmt:
                        if 'mi' in fmt:
                            stats_items.append(html.P(f"{label}: {best[field]:.1f} mi"))
                        elif '$' in fmt:
                            stats_items.append(html.P(f"{label}: ${best[field]:,.0f}"))
                        else:
                            stats_items.append(html.P(f"{label}: {best[field]:.0f}"))
            
            stats = html.Div(stats_items)
        else:
            stats = html.Div([
                html.H5("⚠️ No Locations Found", className="text-warning"),
                html.P("Try adjusting your filters to see more locations."),
                html.P(f"Total Dataset: {len(df):,} locations"),
                html.P(f"Competitors: {len(chickfila_locations)}"),
                html.P(f"Existing Cane's: {len(raising_canes_locations)}"),
                html.Hr(),
                html.P("💡 Quick fixes:"),
                html.Ul([
                    html.Li("Lower minimum revenue"),
                    html.Li("Increase competitor distance"),
                    html.Li("Set zoning to 'All Locations'"),
                    html.Li("Reduce commercial traffic requirement")
                ])
            ])
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats update error: {e}")
        return html.Div([
            html.H5("❌ Stats Error", className="text-danger"),
            html.P(f"Error: {str(e)}")
        ])

# === MAIN APPLICATION RUNNER ===
def main():
    """Main function to run the app"""
    print("🚀 Starting Enhanced Visualization App")
    print(f"📍 Available cities: {[city['display_name'] for city in available_cities]}")
    print(f"🌐 Open your browser to: http://127.0.0.1:8050")
    print("📊 App features:")
    print("   - Interactive location mapping")
    print("   - Advanced filtering controls")
    print("   - Analytics dashboard")
    print("   - Model performance metrics")
    print("   - Error handling and debugging")
    
    try:
        app.run(debug=True, host='127.0.0.1', port=8050)
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("🔧 Troubleshooting:")
        print("   1. Make sure port 8050 is available")
        print("   2. Check that all dependencies are installed")
        print("   3. Verify data files exist")
        print("   4. Try running: pip install -r requirements.txt")

if __name__ == '__main__':
    main()