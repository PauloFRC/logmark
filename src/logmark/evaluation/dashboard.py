import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any

class LogmarkDashboard:
    """
    A simple but production-grade evaluation dashboard using Dash.
    """
    def __init__(self, port: int = 8050):
        self.port = port
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.DARKLY]
        )
        self.results_df = pd.DataFrame()

    def load_results(self, results: List[Dict[str, Any]]):
        """
        Loads a list of evaluation result dictionaries.
        Expected keys: 'dataset', 'model', 'accuracy', 'precision', 'recall', 'f1'
        """
        self.results_df = pd.DataFrame(results)
        self._build_layout()

    def _build_layout(self):
        """
        Constructs the UI layout dynamically.
        """
        if self.results_df.empty:
            self.app.layout = html.Div(
                [html.H2("Logmark Evaluation Dashboard", className="text-center mt-4"),
                 html.P("No results loaded yet.", className="text-center")],
            )
            return

        # Build charts
        fig_f1 = px.bar(
            self.results_df, 
            x="dataset", 
            y="f1", 
            color="model", 
            barmode="group",
            title="F1 Score Comparison by Dataset & Model",
            template="plotly_dark",
            range_y=[0, 1]
        )

        fig_acc = px.bar(
            self.results_df, 
            x="dataset", 
            y="accuracy", 
            color="model", 
            barmode="group",
            title="Accuracy Comparison by Dataset & Model",
            template="plotly_dark",
            range_y=[0, 1]
        )

        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Logmark Evaluation Dashboard", className="text-center mt-4 mb-4"))
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_f1), md=6),
                dbc.Col(dcc.Graph(figure=fig_acc), md=6)
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Table.from_dataframe(
                        self.results_df.round(4), 
                        striped=True, 
                        bordered=True, 
                        hover=True,
                        color="dark"
                    ),
                    className="mt-4"
                )
            ])
        ], fluid=True)

    def run(self):
        """
        Runs the Dash server on the specified port.
        """
        print(f"Starting dashboard on port {self.port}...")
        self.app.run(debug=False, port=self.port)

def launch_dashboard(results: List[Dict[str, Any]], port: int = 8050):
    """
    Helper method to quickly launch the dashboard with a set of results.
    """
    dash_app = LogmarkDashboard(port=port)
    dash_app.load_results(results)
    dash_app.run()
