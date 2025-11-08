"""
Production Options Analytics Dashboard using Plotly Dash.
Run with: python dashboard_app.py
"""
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our modules
from data_loader import OptionsDataLoader
from preprocessing import OptionsPreprocessor
from iv_surface import IVSurface, SVICalibrator
from rn_density import RiskNeutralDensity, HistoricalDensity
from backtest_engine import BacktestEngine, RiskReversalStrategy

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Global data loader
loader = OptionsDataLoader("hood_options.db")
preprocessor = OptionsPreprocessor(risk_free_rate=0.05)

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Options Analytics Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.Hr()
    ]),
    
    # Left sidebar - Filters
    html.Div([
        html.Div([
            html.H3("Filters", style={'color': '#34495e'}),
            
            # Underlying selector
            html.Label("Underlying ID:"),
            dcc.Dropdown(
                id='underlying-dropdown',
                options=[],
                value=None,
                style={'marginBottom': '15px'}
            ),
            
            # Date picker
            html.Label("Snapshot Date:"),
            dcc.DatePickerSingle(
                id='date-picker',
                date=None,
                style={'marginBottom': '15px'}
            ),
            
            # Expiry selector
            html.Label("Expiration Date:"),
            dcc.Dropdown(
                id='expiry-dropdown',
                options=[],
                value=None,
                style={'marginBottom': '15px'}
            ),
            
            # Volume/OI filters
            html.Label("Min Volume:"),
            dcc.Input(
                id='min-volume',
                type='number',
                value=10,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            html.Label("Min Open Interest:"),
            dcc.Input(
                id='min-oi',
                type='number',
                value=50,
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            
            # Load button
            html.Button('Load Data', id='load-button', n_clicks=0,
                       style={'width': '100%', 'padding': '10px', 
                              'backgroundColor': '#3498db', 'color': 'white',
                              'border': 'none', 'borderRadius': '5px',
                              'cursor': 'pointer', 'marginTop': '10px'})
        ], style={
            'width': '20%',
            'padding': '20px',
            'backgroundColor': '#ecf0f1',
            'height': '100vh',
            'position': 'fixed',
            'overflowY': 'auto'
        })
    ]),
    
    # Main content area
    html.Div([
        # KPI Row
        html.Div([
            html.Div([
                html.Div([
                    html.H4("Spot Price", style={'margin': '0', 'fontSize': '16px'}),
                    html.P("Current underlying price", style={'margin': '0', 'fontSize': '11px', 'color': '#888'})
                ]),
                html.H2(id='spot-price', children='--', style={'margin': '5px 0 0 0', 'color': '#27ae60'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white', 
                     'marginRight': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.Div([
                    html.H4("ATM IV", style={'margin': '0', 'fontSize': '16px'}),
                    html.P("At-the-money implied volatility", style={'margin': '0', 'fontSize': '11px', 'color': '#888'})
                ]),
                html.H2(id='atm-iv', children='--', style={'margin': '5px 0 0 0', 'color': '#e74c3c'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white',
                     'marginRight': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.Div([
                    html.H4("30D RR (25Δ)", style={'margin': '0', 'fontSize': '16px'}),
                    html.P("Risk Reversal: 25Δ Call IV - Put IV (skew measure)", style={'margin': '0', 'fontSize': '11px', 'color': '#888'})
                ]),
                html.H2(id='risk-reversal', children='--', style={'margin': '5px 0 0 0', 'color': '#9b59b6'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white',
                     'marginRight': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.Div([
                    html.H4("Butterfly", style={'margin': '0', 'fontSize': '16px'}),
                    html.P("(25Δ Call + Put)/2 - ATM (convexity measure)", style={'margin': '0', 'fontSize': '11px', 'color': '#888'})
                ]),
                html.H2(id='butterfly', children='--', style={'margin': '5px 0 0 0', 'color': '#f39c12'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white',
                     'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # Tabs for different views
        dcc.Tabs(id='tabs', value='surface-tab', children=[
            # IV Surface Tab
            dcc.Tab(label='IV Surface', value='surface-tab', children=[
                html.Div([
                    dcc.Graph(id='iv-surface-3d', style={'height': '600px'}),
                    dcc.Graph(id='iv-smile-slice', style={'height': '400px'})
                ], style={'padding': '20px'})
            ]),
            
            # Risk-Neutral Density Tab
            dcc.Tab(label='Risk-Neutral Density', value='density-tab', children=[
                html.Div([
                    dcc.Graph(id='rn-density-plot', style={'height': '500px'}),
                    html.Div([
                        html.H4("Quantiles"),
                        html.Div(id='quantiles-table')
                    ], style={'marginTop': '20px'})
                ], style={'padding': '20px'})
            ]),
            
            # Greeks Tab
            dcc.Tab(label='Greeks Heatmap', value='greeks-tab', children=[
                html.Div([
                    html.Div([
                        html.H4("Greeks Overview", style={'marginBottom': '10px'}),
                        html.P([
                            html.Strong("Delta: "), "Rate of change of option price vs underlying (directional exposure). Calls: 0 to 1, Puts: -1 to 0.",
                            html.Br(),
                            html.Strong("Gamma: "), "Rate of change of delta vs underlying (curvature risk). Peaks ATM, higher for near-term options.",
                            html.Br(),
                            html.Strong("Theta: "), "Time decay per day. Negative for long options (you lose money each day).",
                            html.Br(),
                            html.Strong("Vega: "), "Sensitivity to 1% change in implied volatility. Higher for ATM and longer-dated options."
                        ], style={'fontSize': '14px', 'color': '#555', 'marginBottom': '20px'})
                    ]),
                    dcc.Graph(id='greeks-heatmap-all', style={'height': '800px'})
                ], style={'padding': '20px'})
            ]),
            
            # Backtest Tab
            dcc.Tab(label='Strategy Backtest', value='backtest-tab', children=[
                html.Div([
                    html.H3("Risk Reversal Backtest"),
                    html.Button('Run Backtest', id='backtest-button', n_clicks=0,
                               style={'padding': '10px 20px', 'marginBottom': '20px'}),
                    html.Div(id='backtest-results'),
                    dcc.Graph(id='equity-curve', style={'height': '400px'})
                ], style={'padding': '20px'})
            ])
        ])
    ], style={'marginLeft': '22%', 'padding': '20px'})
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f5f5'})


# Callbacks

@app.callback(
    Output('underlying-dropdown', 'options'),
    Input('load-button', 'n_clicks')
)
def load_underlyings(n_clicks):
    """Populate underlying dropdown."""
    underlyings = loader.get_available_underlyings()
    return [{'label': f"ID {row['stocks_id']} ({row['n_records']:,} records)", 
             'value': row['stocks_id']} 
            for _, row in underlyings.iterrows()]


@app.callback(
    [Output('date-picker', 'min_date_allowed'),
     Output('date-picker', 'max_date_allowed'),
     Output('date-picker', 'date')],
    Input('underlying-dropdown', 'value')
)
def update_date_range(stocks_id):
    """Update available date range for selected underlying."""
    if not stocks_id:
        return None, None, None
    
    min_date, max_date = loader.get_date_range(stocks_id)
    return min_date, max_date, max_date


@app.callback(
    Output('expiry-dropdown', 'options'),
    [Input('underlying-dropdown', 'value'),
     Input('date-picker', 'date')]
)
def update_expiries(stocks_id, date):
    """Update available expiries for selected underlying and date."""
    if not stocks_id or not date:
        return []
    
    expiries = loader.get_expiries_for_date(stocks_id, date)
    return [{'label': exp, 'value': exp} for exp in expiries]


@app.callback(
    [Output('spot-price', 'children'),
     Output('atm-iv', 'children'),
     Output('risk-reversal', 'children'),
     Output('butterfly', 'children'),
     Output('iv-surface-3d', 'figure'),
     Output('iv-smile-slice', 'figure'),
     Output('rn-density-plot', 'figure'),
     Output('quantiles-table', 'children'),
     Output('greeks-heatmap-all', 'figure')],
    [Input('load-button', 'n_clicks')],
    [State('underlying-dropdown', 'value'),
     State('date-picker', 'date'),
     State('expiry-dropdown', 'value'),
     State('min-volume', 'value'),
     State('min-oi', 'value')]
)
def update_dashboard(n_clicks, stocks_id, date, expiry, min_vol, min_oi):
    """Main callback to update dashboard visualizations."""
    
    # Return empty if no data selected
    if not stocks_id or not date:
        empty_fig = go.Figure()
        return '--', '--', '--', '--', empty_fig, empty_fig, empty_fig, '', empty_fig
    
    # Load data
    df = loader.load_chain_snapshot(stocks_id, date, expiry, min_vol, min_oi)
    
    if len(df) == 0:
        empty_fig = go.Figure()
        return '--', '--', '--', '--', empty_fig, empty_fig, empty_fig, 'No data', empty_fig
    
    # Preprocess
    df = preprocessor.preprocess_chain(df, min_vol, min_oi)
    
    # KPIs
    spot = f"${df['underlying_price'].iloc[0]:.2f}"
    
    # ATM IV
    atm_df = df[df['is_atm']]
    atm_iv_val = atm_df['iv'].mean() if len(atm_df) > 0 else df['iv'].median()
    atm_iv = f"{atm_iv_val*100:.1f}%"
    
    # Risk reversal (25Δ)
    calls_25d = df[(df['call_put'] == 'C') & (df['delta_bucket'] == '25D')]
    puts_25d = df[(df['call_put'] == 'P') & (df['delta_bucket'] == '25D')]
    
    if len(calls_25d) > 0 and len(puts_25d) > 0:
        rr = calls_25d['iv'].mean() - puts_25d['iv'].mean()
        rr_str = f"{rr*100:.1f}%"
    else:
        rr_str = '--'
    
    # Butterfly
    if len(calls_25d) > 0 and len(puts_25d) > 0 and len(atm_df) > 0:
        bf = (calls_25d['iv'].mean() + puts_25d['iv'].mean()) / 2 - atm_iv_val
        bf_str = f"{bf*100:.1f}%"
    else:
        bf_str = '--'
    
    # IV Surface 3D
    surface_fig = create_iv_surface(df)
    
    # IV Smile slice
    smile_fig = create_iv_smile(df, expiry if expiry else df['expiration_date'].iloc[0])
    
    # Risk-neutral density
    density_fig, quantiles_div = create_rn_density_plot(df, expiry if expiry else df['expiration_date'].iloc[0])
    
    # Greeks heatmap (all 4 in one)
    greeks_fig = create_greeks_heatmap_all(df)
    
    return spot, atm_iv, rr_str, bf_str, surface_fig, smile_fig, density_fig, quantiles_div, greeks_fig


# Visualization functions

def create_iv_surface(df):
    """Create 3D IV surface plot."""
    # Pivot data
    pivot = df.pivot_table(
        values='iv',
        index='price_strike',
        columns='expiration_date',
        aggfunc='mean'
    )
    
    # Convert to percentage
    z_values = pivot.values * 100
    
    fig = go.Figure(data=[go.Surface(
        z=z_values,
        x=list(range(len(pivot.columns))),  # Use numeric indices
        y=pivot.index,
        colorscale='RdYlGn_r',  # Red (high IV) to Green (low IV)
        colorbar=dict(title='IV %', ticksuffix='%'),
        customdata=np.array([[str(col) for col in pivot.columns] for _ in pivot.index]),
        hovertemplate='<b>Strike:</b> %{y:.2f}<br>' +
                      '<b>Expiry:</b> %{customdata}<br>' +
                      '<b>IV:</b> %{z:.1f}%<extra></extra>'
    )])
    
    # Create tick labels for expiries
    expiry_labels = [str(exp)[:10] for exp in pivot.columns]  # Truncate to date only
    
    fig.update_layout(
        title={
            'text': 'Implied Volatility Surface<br><sub>Higher values (red) indicate expensive options, lower (green) indicate cheap options</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis=dict(
                title='Expiration Date',
                ticktext=expiry_labels,
                tickvals=list(range(len(pivot.columns))),
                tickangle=-45
            ),
            yaxis_title='Strike Price ($)',
            zaxis=dict(title='Implied Volatility (%)', ticksuffix='%'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=600,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    return fig


def create_iv_smile(df, expiry):
    """Create IV smile for a specific expiry."""
    exp_df = df[df['expiration_date'] == expiry]
    
    fig = go.Figure()
    
    # Calls
    calls = exp_df[exp_df['call_put'] == 'C'].sort_values('price_strike')
    fig.add_trace(go.Scatter(
        x=calls['price_strike'],
        y=calls['iv'] * 100,
        mode='markers+lines',
        name='Calls',
        marker=dict(size=8)
    ))
    
    # Puts
    puts = exp_df[exp_df['call_put'] == 'P'].sort_values('price_strike')
    fig.add_trace(go.Scatter(
        x=puts['price_strike'],
        y=puts['iv'] * 100,
        mode='markers+lines',
        name='Puts',
        marker=dict(size=8)
    ))
    
    # ATM line
    spot = exp_df['underlying_price'].iloc[0]
    fig.add_vline(x=spot, line_dash='dash', annotation_text='ATM')
    
    fig.update_layout(
        title=f'IV Smile - {expiry}',
        xaxis_title='Strike',
        yaxis_title='Implied Volatility %',
        height=400
    )
    
    return fig


def create_rn_density_plot(df, expiry):
    """Create risk-neutral density plot."""
    exp_df = df[df['expiration_date'] == expiry]
    
    if len(exp_df) == 0:
        return go.Figure(), 'No data for selected expiry'
    
    # Extract density
    rnd = RiskNeutralDensity()
    
    calls = exp_df[exp_df['call_put'] == 'C'].sort_values('price_strike')
    strikes = calls['price_strike'].values
    prices = calls['mid_price'].values
    T = calls['T'].iloc[0]
    
    try:
        K_grid, density, _ = rnd.breeden_litzenberger(strikes, prices, T)
        
        # Compute quantiles
        quantiles_result = rnd.compute_quantiles(K_grid, density)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=K_grid,
            y=density,
            mode='lines',
            name='RN Density',
            fill='tozeroy'
        ))
        
        # Add quantile lines
        for q, v in zip(quantiles_result['quantiles'], quantiles_result['values']):
            fig.add_vline(x=v, line_dash='dash', 
                         annotation_text=f'{q*100:.0f}%: ${v:.2f}')
        
        fig.update_layout(
            title='Risk-Neutral Probability Density',
            xaxis_title='Price',
            yaxis_title='Probability Density',
            height=500
        )
        
        # Create quantiles table
        quantiles_df = pd.DataFrame({
            'Quantile': [f'{q*100:.0f}%' for q in quantiles_result['quantiles']],
            'Price': [f'${v:.2f}' for v in quantiles_result['values']]
        })
        
        quantiles_div = dash_table.DataTable(
            data=quantiles_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in quantiles_df.columns],
            style_cell={'textAlign': 'center'}
        )
        
        return fig, quantiles_div
        
    except Exception as e:
        return go.Figure(), f'Error computing density: {str(e)}'


def create_greeks_heatmap_all(df):
    """Create combined heatmap showing all 4 Greeks."""
    from plotly.subplots import make_subplots
    
    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Delta (Directional)', 'Gamma (Curvature)', 
                       'Theta (Time Decay)', 'Vega (Vol Sensitivity)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    greeks = ['delta', 'gamma', 'theta', 'vega']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    colormaps = ['RdBu', 'YlOrRd', 'Blues_r', 'Greens']
    
    for greek, (row, col), colormap in zip(greeks, positions, colormaps):
        # Pivot data
        pivot = df.pivot_table(
            values=greek,
            index='price_strike',
            columns='expiration_date',
            aggfunc='mean'
        )
        
        # Determine if we should center at zero (delta, theta) or not (gamma, vega)
        zmid = 0 if greek in ['delta', 'theta'] else None
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=[str(col)[:10] for col in pivot.columns],
                y=pivot.index,
                colorscale=colormap,
                zmid=zmid,
                showscale=True,
                colorbar=dict(
                    len=0.4,
                    y=0.75 if row == 1 else 0.25,
                    x=1.02 if col == 2 else -0.02,
                    xanchor='left' if col == 2 else 'right'
                ),
                hovertemplate='<b>Strike:</b> %{y:.2f}<br>' +
                              '<b>Expiry:</b> %{x}<br>' +
                              f'<b>{greek.capitalize()}:</b> %{{z:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="Expiration Date", tickangle=-45, row=2, col=1)
    fig.update_xaxes(title_text="Expiration Date", tickangle=-45, row=2, col=2)
    fig.update_yaxes(title_text="Strike Price", row=1, col=1)
    fig.update_yaxes(title_text="Strike Price", row=2, col=1)
    
    fig.update_layout(
        title={
            'text': 'Greeks Heatmap - All Options',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        showlegend=False
    )
    
    return fig


def create_greeks_heatmap(df, greek):
    """Create heatmap of selected greek."""
    pivot = df.pivot_table(
        values=greek,
        index='price_strike',
        columns='expiration_date',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title=f'{greek.capitalize()} Heatmap',
        xaxis_title='Expiration',
        yaxis_title='Strike',
        height=600
    )
    
    return fig


# Run app
if __name__ == '__main__':
    app.run(debug=True, port=8050)
    print("Dashboard running at http://localhost:8050")