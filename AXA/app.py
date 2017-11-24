import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import re
import flask
import os
from functools import reduce
import numpy as np
import seaborn as sns
import json


pal = sns.color_palette("BrBG", 100)
#input_path = 'C:\\Users\\mhenaff.EKI\\Documents\\AXA Italy\\Data\\outputs\\media\\'
input_path = ''
app = dash.Dash(__name__)
app.config['suppress_callback_exceptions']=True

css_directory = os.getcwd()
#stylesheets = ['tuto_stylesheet.css', 'my_stylesheet.css']
stylesheets = []
static_css_route = '/AXA/static'

r2 = pd.read_csv(input_path + "campaigns_mean_R2.csv", sep=";", decimal=",", header=0)
coefs = pd.read_csv(input_path + "campaigns_mean_coef.csv", sep=";", decimal=",", header=0)
pvalues = pd.read_csv(input_path + "campaigns_mean_pvalue.csv", sep=";", decimal=",", header=0)
data = pd.read_csv(input_path + "data_NB.csv", sep=";", decimal=",", header=0, index_col=0)
data['base'] = 1
y = data.loc[:,'NB']
time = list(data.index)

file = open(input_path + "categories.json", "r")
var_df = pd.DataFrame.from_dict(json.loads(file.read()))
file.close()

variables = [v.replace(" bool", "") for v in r2.columns if re.search(' bool', v)]
categories = var_df.category.unique()
ordered_categories = var_df[['category', 'order']].drop_duplicates().sort_values('order').category

var_combinations = ~r2[[v for v in r2.columns if re.search(' bool', v)]].T
var_combinations['variable'] = [v.replace(' bool', '') for v in var_combinations.index]
categories_empty = var_combinations.merge(var_df[['variable', 'category']]).groupby('category').sum().min(axis=1)
mandatory_cat = list(categories_empty.index[categories_empty > 0])
mandatory_cat_var = dict()
for cat in mandatory_cat:
    mandatory_cat_var[cat] = list(var_df.loc[(var_df.category == cat) & (var_df.variable.isin(variables)), 'variable'])

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

def make_dropdown(cat):
    return html.Div(
        style={'backgroundColor': 'rgb({}, {}, {})'.format(*list(var_df.loc[var_df.category == cat, 'color'])[0]),
               'display': 'inline-block',
               'color': var_df.loc[var_df.category == cat, 'font_col'].values[0],
               'width': '33%',
               'height': '45%',
               'paddingBottom': '10px',
               'fontSize': '10px'},
        children=[
            dcc.Markdown(' **' + cat + '**'),
            dcc.Dropdown(
                id='input-vars-' + cat,
                options=[{'label': var_df.loc[var_df.variable == v, 'name'], 'value': v}
                         for v in var_df.loc[var_df.category == cat, 'variable'] if v in variables],
                placeholder=[cat],
                multi=True
                #value=[var_df.loc[var_df.category == cat, 'variable'].values[0]]
            )]
    )


app.layout = html.Div([
    html.Div(style={'height': '300px', 'width': '100%', 'marginBottom': '10px'}, children=[
        html.Div(style={'height': '100%', 'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}, children=[
            make_dropdown('Trend'),
            make_dropdown('Seasonality'),
            make_dropdown('Competition'),
            make_dropdown('Discounts'),
            make_dropdown('Price'),
            make_dropdown('Distribution'),
        ]),
        html.Div(style={'height':'100%', 'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}, children=[
            html.Div(style={'height': '38px', 'fontSize': '10px'}, children=[
                dcc.Dropdown(
                    id='hist-input-var',
                    options=[{'label': var_df.loc[var_df.variable == v, 'name'], 'value': v}
                             for v in var_df.sort_values('name').variable],
                    multi=False,
                )
            ]),
            dcc.Graph(id='histograms', style={'height': '262px'})
        ])
    ]),

    html.Div(
        children=[
            html.Div(
                style={'width': '90%', 'display': 'table-cell', 'verticalAlign': 'middle', 'height': '80px'},
                id='result-table'
            ),
            html.Div(
                style={'width': '10%', 'display': 'table-cell', 'verticalAlign': 'middle', 'height': '80px'},
                id='result-r2'
            )
        ],
    ),

    html.Div(
        children=[
            html.Div(
                style={'width': '50%', 'display': 'inline-block', 'height': '350px', 'verticalAlign': 'top'},
                children=dcc.Graph(id='courbe-couches')
            ),
            html.Div(
                style={'width': '50%', 'display': 'inline-block', 'height': '350px', 'verticalAlign': 'top'},
                children=dcc.Graph(id='bar-leviers')
            )
        ]
    ),

    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(id='selected-model-contrib', style={'display': 'none'}),
    html.Div(id='selected-model-table', style={'display': 'none'}),
    html.Div(id='selected-model-r2', style={'display': 'none'}),
])


def update_options(cat):
    def fun(jsonified_light_var_df):
        if jsonified_light_var_df is not None:
            light_var_df = pd.read_json(jsonified_light_var_df)
            return [{'label': light_var_df.loc[light_var_df.variable == v, 'name'], 'value': v}
                    for v in light_var_df.loc[light_var_df.category == cat, 'variable']]
    return fun


for cat in ['Competition', 'Trend', 'Seasonality', 'Discounts', 'Price', 'Distribution']:
    app.callback(
        Output('input-vars-' + cat, 'options'),
        [Input('intermediate-value', 'children')])(update_options(cat))


@app.callback(
    Output('intermediate-value', 'children'),
    [Input('input-vars-Trend', 'value'),
     Input('input-vars-Seasonality', 'value'),
     Input('input-vars-Competition', 'value'),
     Input('input-vars-Discounts', 'value'),
     Input('input-vars-Price', 'value'),
     Input('input-vars-Distribution', 'value')
     ])
def select_data(*var_cat):
    vars_flat = flatten(var_cat)
    if vars_flat is not None:
        light_var_df = var_df[var_df.variable.isin(
            list(coefs[eval('&'.join(["~coefs['" + v + " bool']" for v in vars_flat]))][[c for c in coefs.columns if not re.search(" bool", c)]]\
                 .dropna(axis=1, how='all').columns))]
        return light_var_df.to_json()
    else:
        return var_df.to_json()


def make_table_row(df, name, is_header=False, is_footer=False):
    row_func = html.Th if is_header else html.Td
    op = '0.2' if not is_header else '1'
    return [html.Tr([
            row_func(
                style={
                    'fontSize': '12px',
                    'fontWeight': 'bold' if is_header else 'normal',
                    'backgroundColor': 'rgba({}, {}, {}, {})'.format(*(list(df.loc[df.variable == v, 'color'].values[0]) + [op])),
                    'color': df.loc[df.variable == v, 'font_col'].values[0] if is_header else 'black',
                    'borderTopLeftRadius': '10px' if is_header and i==0 else '0px',
                    'borderTopRightRadius': '10px' if is_header and i == len(df.variable) - 1 else '0px',
                    'borderBottomLeftRadius': '10px' if is_footer and i == 0 else '0px',
                    'borderBottomRightRadius': '10px' if is_footer and i == len(df.variable) - 1 else '0px',
                },
                children=df.loc[df.variable == v, name].values[0]
            ) for v, i in zip(df.variable, range(len(df.variable)))])]


@app.callback(
    Output('result-table', 'children'),
    [Input('selected-model-table', 'children')])
def update_table(jsonified_res_table):
    if jsonified_res_table is not None:
        res_table = pd.read_json(jsonified_res_table).T
        res_table['coef'] = res_table['coef'].map("{:,.2e}".format)
        res_table['pvalue'] = res_table['pvalue'].map("{:,.1e}".format)
    else:
        res_table = pd.DataFrame(index=coefs.columns)
        res_table['coef'] = None
        res_table['pvalue'] = None

    res_table['variable'] = res_table.index
    res_table2 = res_table.merge(var_df).sort_values('order')
    return html.Table(
        style={"height": '80px'},
        children = make_table_row(res_table2, 'name', is_header=True)\
                   + make_table_row(res_table2, 'coef')\
                   + make_table_row(res_table2, 'pvalue', is_footer=True)
        )


@app.callback(
    Output('result-r2', 'children'),
    [Input('selected-model-r2', 'children')])
def update_table(jsonified_res_r2):
    if jsonified_res_r2 is not None:
        res_r2= pd.read_json(jsonified_res_r2)
        r2_val = res_r2.R2.values[0]
        color_r2 = 'rgba({}, {}, {}, 0.5)'.format(*[int(x*256) for x in pal[max(int((r2_val - 0.5)*200) - 1, 0)]])
        return html.Div(
            style={'backgroundColor': color_r2, 'display': 'block', 'height': '100%', 'width': '100%',
                   'textAlign': 'center', 'verticalAlign': 'middle', 'borderRadius': '10px'},
            children=html.Span(
                style={'paddingTop': '25px', 'padding': '45px 20px 45px 20px', 'height': '30px',
                       'verticalAlign': 'middle', 'display': 'table-cell'},
                children='{:.3f}'.format(r2_val)
            )
        )
#'backgroundColor': 'rgb({}, {}, {})'.format(*color_r2)


@app.callback(
    Output('courbe-couches', 'figure'),
    [Input('selected-model-contrib', 'children')])
def update_couche(jsonified_contribs_min):
    if jsonified_contribs_min is not None:
        contribs_min = pd.read_json(jsonified_contribs_min)
        contribs_couche = make_couche(contribs_min)
        graph_data = [go.Scatter(
            x=time,
            y=contribs_couche.loc[:, cc],
            name=var_df.loc[var_df.variable == cc, 'name'].values[0],
            text=var_df.loc[var_df.variable == cc, 'name'].values[0],
            fill='tonexty',
            mode='lines',
            fillcolor='rgb({}, {}, {})'.format(*list(var_df.loc[var_df.variable == cc, 'color'])[0]),
            line=dict(
                width=0,
                color='rgb({}, {}, {})'.format(*list(var_df.loc[var_df.variable == cc, 'color'])[0])
            )
        ) for cc in contribs_couche] \
                     + [go.Scatter(x=time, y=y, name="New Businesses", mode='lines', line=dict(width=3, color='black'))]
        return {
            'data': graph_data,
            'layout': go.Layout(
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                height=350,
                hovermode='closest',
                xaxis={'title': 'Time'},
                yaxis={'title': 'New Businesses'},
                showlegend=False
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                height=350,
                hovermode='closest',
                xaxis={'title': 'Time'},
                yaxis={'title': 'New Businesses'}
            )
        }


@app.callback(
    Output('bar-leviers', 'figure'),
    [Input('selected-model-contrib', 'children')])
def update_levier(jsonified_contribs_min):
    if jsonified_contribs_min is not None:
        contribs_min = pd.read_json(jsonified_contribs_min)
        leviers_df = \
        pd.DataFrame([contribs_min.columns, contribs_min.sum(axis=0) / y.sum()], index=['variable', 'leviers']).T \
            .merge(var_df, how='outer') \
            .fillna(0) \
            .groupby('category')["leviers"].sum()
        bar_data = [go.Bar(
            x=[cat],
            y=[leviers_df.loc[cat]],
            name=cat,
            text=cat,
            marker=dict(
                color='rgb({}, {}, {})'.format(*list(var_df.loc[var_df.category == cat, 'color'])[0]),
            )
        ) for cat in ordered_categories]
        return {
            'data': bar_data,
            'layout': go.Layout(
                margin={'l': 40, 'b': 60, 't': 10, 'r': 60},
                height=350,
                hovermode='closest',
                yaxis={'title': '%'},
                showlegend=False
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(
                margin={'l': 40, 'b': 60, 't': 10, 'r': 60},
                height=350,
                hovermode='closest',
                yaxis={'title': '%'}
            )
        }

@app.callback(
    Output('histograms', 'figure'),
    [Input('hist-input-var', 'value')])
def update_hist(var):
    if var is not None:
        hist_data = [go.Histogram(
            x=coefs.loc[:, var],
            name=var,
            text=var,
            marker=dict(
                color='rgb({}, {}, {})'.format(*list(var_df.loc[var_df.variable == var, 'color'])[0]),
            )
        )]
        return {
            'data': hist_data,
            'layout': go.Layout(
                margin={'l': 40, 'b': 20, 't': 10, 'r': 10},
                hovermode='closest',
                height=262,
                showlegend=False
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(
                margin={'l': 40, 'b': 20, 't': 10, 'r': 10},
                hovermode='closest',
                height=262,
            )
        }


def make_filter_cond(selected_vars):
    coefs_temp = coefs[eval('&'.join(["~coefs['" + v + " bool']" for v in selected_vars]))][
        [c for c in coefs.columns if not re.search(" bool", c)]].dropna(axis=1, how='all')
    unselected_vars = [c for c in coefs_temp.columns[coefs_temp.isnull().any()] if c in variables]
    for cat, var_list in mandatory_cat_var.items():
        if (len([v for v in selected_vars if v in var_list]) == 0) and (var_list[0] in unselected_vars):
            print("Selecting " + var_list[0] + " as default " + cat)
            unselected_vars.remove(var_list[0])
    return '&'.join(["~coefs['" + v + " bool']" for v in selected_vars] + ["coefs['" + v + " bool']" for v in unselected_vars])


def make_table(selected_vars):
    filter_cond = make_filter_cond(selected_vars)
    coefs_temp = coefs[eval(filter_cond)][[c for c in coefs.columns if not re.search(" bool", c)]]\
        .dropna(axis=1, how='all')
    if coefs_temp.shape[0] == 1:
        pvalues_temp = pvalues[eval(filter_cond)][[c for c in coefs.columns if not re.search(" bool", c)]] \
            .dropna(axis=1, how='all')
        pvalues_temp['base'] = None
        table_res = pd.concat([coefs_temp, pvalues_temp], axis=0)
        table_res.index = ['coef', 'pvalue']
        return table_res


def make_r2(selected_vars):
    filter_cond = make_filter_cond(selected_vars)
    r2_temp = r2[eval(filter_cond)][[c for c in r2.columns if not re.search(" bool", c)]]
    if r2_temp.shape[0] == 1:
        return r2_temp


def make_contribs(selected_vars):
    filter_cond = make_filter_cond(selected_vars)
    coefs_temp = coefs[eval(filter_cond)][[c for c in coefs.columns if not re.search(" bool", c)]] \
        .dropna(axis=1, how='all')
    if coefs_temp.shape[0] == 1:
        contribs = pd.DataFrame(data[coefs_temp.columns].values * coefs_temp.values, columns=coefs_temp.columns,
                                index=data.index)
        min_val = contribs.min(axis=0).drop('base')
        min_val['base'] = - min_val.sum()
        contribs_min = contribs - min_val
        return contribs_min


def select_model(res_type):
    if res_type == 'contrib':
        make_fun = make_contribs
    if res_type == 'table':
        make_fun = make_table
    if res_type == 'r2':
        make_fun = make_r2

    def select_model_type(*var_cat):
        list_vars = [x if isinstance(x, list) else [x] for x in var_cat if x is not None]
        if len(list_vars) > 0:
            selected_vars = reduce(lambda x, y: x + y, list_vars)
            type_res = make_fun(selected_vars)
            if type_res is not None:
                return type_res.to_json()

    return select_model_type


for res_type in ['contrib', 'table', 'r2']:
    app.callback(
        Output('selected-model-' + res_type, 'children'),
        [Input('input-vars-Trend', 'value'),
         Input('input-vars-Seasonality', 'value'),
         Input('input-vars-Competition', 'value'),
         Input('input-vars-Discounts', 'value'),
         Input('input-vars-Price', 'value'),
         Input('input-vars-Distribution', 'value')])(select_model(res_type))


def make_couche(contribs_min):
    return contribs_min[var_df[var_df.variable.isin(contribs_min.columns)].sort_values('order').variable].cumsum(axis=1)


def flatten(var_cat):
    new_var_cat = [x if isinstance(x, list) else [x] for x in var_cat if x is not None]
    if len(new_var_cat) > 0:
        return reduce(lambda x, y: x + y, new_var_cat)


@app.server.route('{}<stylesheet>'.format(static_css_route))
def serve_stylesheet(stylesheet):
    if stylesheet not in stylesheets:
        raise Exception(
            '"{}" is excluded from the allowed static files'.format(
                stylesheet
            )
        )
    return flask.send_from_directory(css_directory, stylesheet)


for stylesheet in stylesheets:
    app.css.append_css({"external_url": "{}/{}".format(static_css_route, stylesheet)})


if __name__ == '__main__':
    app.run_server(debug=True)