#Import the required libraries
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

from config import settings
from cetm46_project_engine import dataWrangler, EDA, graphBuilder, cluster


app = dash.Dash(__name__, external_stylesheets = [dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME])
app.title = "EDA and Customer Segmentation Dashboard"

#Clean your data
path = settings.file_path

#Read in the data
raw_data = pd.read_csv(path)

#Instantiate data wrangler and read in the filepath
wrangler = dataWrangler()
#Clean the data
clean_data = wrangler.data_handler(raw_data)

#Preprocess the data
scaled_data = wrangler.preprocess_data(clean_data)

#Transform the data into 2 dimensions
x_t_df = wrangler.pca(scaled_data)


colors = ["#3366cc", "#2ca02c", "#d62728", "rgb(217, 95, 2)", "#829dc9", "#9b2cad", "#478b87", "#188f97", "#411897"]


#Instantiate the graph builder
graph_builder = graphBuilder()

use_case = "The graph above shows a graphical segmentation of clusters selected at the slider section. To create a new \
            clusters, click on the slider to select the number of clusters and wait a few seconds for the application to \
            perform clustering and display the output. Further insight can be explored at the EDA section."

bar_features = [ {"label":"Term Deposit Subscription", "value": "term_deposit_subscribed"}, {"label": "Account Balance", "value": "balance"} 
                ]
pie_features = [ {"label": "Personal Loan", "value": "personal_loan"},  {"label": "Housing Loan", "value": "housing_loan"}]

# Main Layout

app.layout = dbc.Container(
    [
    dcc.Store(id="clean-data", data=clean_data.to_dict("records")),
    dcc.Store(id="x-t-df", data= x_t_df.to_dict("records")),
    dbc.Row(
        dbc.Col(
            html.H2(
                "EDA and Customer Segmentation Dashboard", style={"color": "white"}
            ), className="text-center bg-primary p-2"
        )
    ),
       dbc.Row(
            [
                                    dbc.Col(
                [
                    dbc.Card(
                        [
                            html.Label("N_Cluster:"), 
                            html.Div(
                                dcc.Slider(
                                    id="n_cluster",
                                    min=3,
                                    max=10,
                                    marks={i: f'Label {i}' if i == 3 else str(i) for i in range(3, 11)},
                                    value=6,
                                    vertical=True,
                                ),
                                style={
                                    "display": "flex",
                                    "justify-content": "center",
                                    "align-items": "center",
                                    "border": "none",
                                }
                            )
                        ],
                        className="mt-4",
                        style={
                            "width": "100%",
                            "border": "none",
                            "border-radius": "0px",
                            "background-color": "transparent",
                            "font-size": "14px",
                            "font-weight": "400",
                            "color": "#212529",
                            "padding": "8px 16px",
                            "box-shadow": "none",
                            "outline": "none",
                        },
                    )
                ],
                width={"size": 2},
            ),
                dbc.Col([
                        dcc.Graph(
                            id="cluster_graph",
                        )
                    ],
                    width={"size": 10, "offset": 0}),
                
            ]
        ),
        dbc.Row(dbc.Col(dbc.Card([dbc.CardBody([html.P(use_case, className="card-text")])], style={"text-align": "center"}))),
    dbc.Row(
        [
         dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H4("EDA Section", style={"text-align":"center"})),
                            dbc.CardBody(
                                [
                                    html.Label("Bar Chart", className="control_label"),
                                    dcc.Dropdown(
                                        id="bar_types", 
                                        options=bar_features, 
                                        value=bar_features[0]["value"], 
                                        className="dcc_control", 
                                        clearable=False,
                                    ),
                                    html.P("", id="bar_type_desc",  className="card-text"),
                                    html.Label("Pie Chart", className="control_label"),
                                    dcc.Dropdown(
                                        id="pie_types", 
                                        options=pie_features, 
                                        value=pie_features[0]["value"], 
                                        className="dcc_control",
                                        clearable=False
                                    ),
                                    html.Div("", id="pie_type_desc", className = "card-text"),
                                ]
                            )
                        ],
                        className = "mt-4",
                        style={"border": "none"}
                    ), 
                    width={"size":12}, 
                    lg=3, 
                    className="mt-4"
                ),


        dbc.Col(dcc.Graph(id="bar-chart", className="pb-4")),
        dbc.Col(dcc.Graph(id="pie-chart", className="pb-4"))
        ], className="ms-1")

    ],
    fluid =True,
)
@app.callback(
        Output("cluster_graph", "figure"),
        Output("bar_type_desc", "children"),
        Output("bar-chart", "figure"),
        Output("pie_type_desc", "children"),
        Output("pie-chart", "figure"),
        Input("n_cluster", "value"),
        Input("bar_types", "value"),
        Input("pie_types", "value"),
        State("clean-data", "data"),
        State("x-t-df", "data")    
)
def cluster_graph(n_cluster, bar_value, pie_value, clean_data, x_t_df):
    
    clean_data = pd.DataFrame.from_records(clean_data)
    x_t_df = pd.DataFrame.from_records(x_t_df)

    #Generate the labels
    #global labels
    labels, _ = cluster(x_t_df, n_cluster)

    #Add the labels to dataframe
    clean_data["cluster"] = labels

    #Instantiate the EDA Services
    eda = EDA(clean_data)
    
    #eda_bar = None
    if bar_value == "balance":
        #generate the insight
        eda_bar = eda.bal_eda("cluster", bar_value)

        #Sort the values in descending order
        sorted_bal_eda = eda_bar.sort_values(by="Average Balance", ascending=False)

        #Extract the clusters with the highest average balance 
        cluster_1 = sorted_bal_eda["cluster"].iloc[0]
        cluster_2 = sorted_bal_eda["cluster"].iloc[1]
        cluster_3 = sorted_bal_eda["cluster"].iloc[2]

        #The first message content
        content_1 = f"The three top clusters with the highest average balance are \
            Cluster {cluster_1}, Cluster {cluster_2}, Cluster {cluster_3}."
            
        #The second message content
        content_2 = f"Kindly consider selling term deposit subscription to the members of these clusters."

        #Combine the message and display them in bullet points
        description_message = html.Ul([html.Li(content_1), html.Li(content_2)])

        #Plot the graph
        fig2 = graph_builder.bar_chart(eda_bar, "cluster", "Average Balance")

    elif bar_value == "term_deposit_subscribed":
        #generate the insight
        eda_bar = eda.eda_num("cluster", bar_value)

        #Sort the values
        sorted_eda_bar = eda_bar.sort_values(by = f"percentage {bar_value}", ascending=False)

        #Extract the clusters 
        cluster_1 = sorted_eda_bar["cluster"].iloc[0]
        cluster_2 = sorted_eda_bar["cluster"].iloc[1]
        cluster_3 = sorted_eda_bar["cluster"].iloc[2]
        content_1 = f"The top three clusters with the highest term deposit subscription are \
            Cluster {cluster_1}, Cluster {cluster_2}, Cluster {cluster_3}." 
            
        content_2 = f"Kindly consider selling house mortage loan to the members of these clusters."
        description_message = html.Ul([html.Li(content_1), html.Li(content_2)])
        fig2 = graph_builder.bar_chart(eda_bar, "cluster", f"percentage {bar_value}")
    else:
        #Note, this statement will never execute, however, I intentionally left it for further expansion
        eda_bar = eda.eda_obj("cluster", bar_value, n_cluster)
        fig2 = graph_builder.bar_chart(eda_bar, "cluster", f"percentage {bar_value} subscription")

    eda_pie = eda.eda_obj("cluster", pie_value, n_cluster)
    sorted_pie = eda_pie.sort_values(by=f"percentage {pie_value} subscription", ascending=False)

    pie_cluster_1 = sorted_pie["cluster"].iloc[0]
    pie_cluster_2 = sorted_pie["cluster"].iloc[1]
    pie_cluster_3 = sorted_pie["cluster"].iloc[2]
    pie_content_1 = f"The top three clusters with the highest percentage subscription are {pie_cluster_1},  \
                         {pie_cluster_2}, and  {pie_cluster_3}."
    pie_content_2 = f"The members of these clusters are likely to buy our insurance products, \
                    kindly consider engaging them with such products."
    pie_description_message = html.Ul([html.Li(pie_content_1), html.Li(pie_content_2)])
    fig3 = graph_builder.pie_chart(eda_pie, "cluster", f"percentage {pie_value} subscription")

    #Plot the graph
    fig1 = graph_builder.cluster_graph(labels, x_t_df, k=n_cluster)
    return fig1, description_message, fig2, pie_description_message, fig3


if __name__ == '__main__':
    app.run_server(debug=True)