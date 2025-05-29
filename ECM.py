import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import datetime
import random

# ---------- 1. Data Extraction and Preprocessing ----------
def simulate_data(days=30):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    usage = np.random.normal(loc=30, scale=5, size=days)
    usage[5] += 20  # Inject anomaly
    usage[12] += 25
    return pd.DataFrame({'date': dates, 'usage_kWh': usage})

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

# ---------- 2. Consumption Pattern Analysis + 3. Anomaly Detection ----------
def analyze_and_flag_anomalies(df):
    model = IsolationForest(contamination=0.1)
    df['anomaly'] = model.fit_predict(df[['usage_kWh']])
    return df

# Initialize data
df = preprocess_data(simulate_data())

# ---------- Dash App Initialization ----------
app = dash.Dash(__name__)
server = app.server

# ---------- 4. Real-time Tracking (Simulated with Interval) ----------
app.layout = html.Div([
    html.H1("⚡ Energy Consumption Monitoring Dashboard", style={'textAlign': 'center'}),

    dcc.Graph(id='usage-graph'),
    
    html.Div([
        html.H4("🏆 Gamification: Efficiency Score"),
        html.Div(id='efficiency-score', style={'fontSize': 22, 'color': 'green'}),
    ], style={'margin': '20px 0'}),

    html.Div([
        html.Label("🗣️ Submit Feedback:"),
        dcc.Input(id='feedback-input', type='text', placeholder='Your feedback...', style={'width': '70%'}),
        html.Button('Submit', id='submit-btn', n_clicks=0),
        html.Div(id='feedback-output', style={'marginTop': '10px'})
    ]),

    dcc.Interval(id='interval-update', interval=60000, n_intervals=0),  # Every 60s
])

# ---------- Callbacks ----------
@app.callback(
    Output('usage-graph', 'figure'),
    Input('interval-update', 'n_intervals')
)
def update_graph(n):
    global df
    new_row = {
        'date': pd.Timestamp.now(),
        'usage_kWh': np.random.normal(30, 5)
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = preprocess_data(df)
    df = analyze_and_flag_anomalies(df)

    fig = px.line(df, x='date', y='usage_kWh', title='Energy Usage Over Time')
    anomalies = df[df['anomaly'] == -1]
    fig.add_scatter(x=anomalies['date'], y=anomalies['usage_kWh'],
                    mode='markers', name='Anomalies',
                    marker=dict(color='red', size=10))
    return fig

@app.callback(
    Output('efficiency-score', 'children'),
    Input('interval-update', 'n_intervals')
)
def calculate_efficiency(n):
    recent = df.tail(7)
    avg = recent['usage_kWh'].mean()
    score = max(0, 100 - int(avg))  # Basic efficiency gamification
    return f"Your score: {score}/100 🌱"

@app.callback(
    Output('feedback-output', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('feedback-input', 'value')
)
def collect_feedback(n_clicks, feedback):
    if n_clicks > 0 and feedback:
        print("User feedback:", feedback)  # In real app: Save to DB
        return "✅ Thanks for your feedback!"
    return ""

# ---------- Run App ----------
if __name__ == '__main__':
    app.run_server(debug=True)
