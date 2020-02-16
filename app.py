import time
import importlib

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier

import utils.dash_reusable_components as drc
import utils.figures as figs

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.config["suppress_callback_exceptions"] = True
server = app.server


def generate_data(n_samples, dataset, noise):
    if dataset == "moons":
        return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)

    elif dataset == "circles":
        return datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=1
        )

    elif dataset == "linear":
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1,
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            "Data type incorrectly specified. Please choose an existing dataset."
        )


app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Decision Boundary Visualizer",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                ),
                                html.Img(
                                    src=app.get_asset_url("visualizerlogo.png"),
                                    style={
                                        "width": "auto",
                                        "height": "auto",
                                        "max-width": "60px",
                                        "max-height": "60px",
                                        "padding-left": "20px",
                                        "padding-bottom": "25px",
                                    },
                                ),
                            ],
                        ),
                        html.A(
                            id="attribution-logo",
                            children=[
                                html.Img(src=app.get_asset_url("dash-logo-new.png")),
                                html.H3("Made with"),
                            ],
                            href="https://plot.ly/products/dash/",
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select Model",
                                            id="dropdown-select-model",
                                            options=[
                                                {
                                                    "label": "Support Vector Machine",
                                                    "value": "SVM",
                                                },
                                                {
                                                    "label": "Logistic Regression",
                                                    "value": "LogReg",
                                                },
                                                {
                                                    "label": "Linear Discriminant Analysis",
                                                    "value": "LDA",
                                                },
                                                {
                                                    "label": "Quadratic Discriminant Analysis",
                                                    "value": "QDA",
                                                },
                                                {
                                                    "label": "Multilayer Perceptron",
                                                    "value": "MLP",
                                                },
                                                {
                                                    "label": "Decision Tree",
                                                    "value": "DTree",
                                                },
                                                {
                                                    "label": "Random Forest",
                                                    "value": "RForest",
                                                },
                                                {
                                                    "label": "AdaBoost",
                                                    "value": "ABoost",
                                                },
                                                {
                                                    "label": "XGBoost",
                                                    "value": "XGBoost",
                                                },
                                                {
                                                    "label": "k Nearest Neighbors",
                                                    "value": "kNN",
                                                },
                                            ],
                                            clearable=False,
                                            searchable=True,
                                            value="SVM",
                                        ),
                                        drc.NamedDropdown(
                                            name="Select Dataset",
                                            id="dropdown-select-dataset",
                                            options=[
                                                {"label": "Moons", "value": "moons"},
                                                {
                                                    "label": "Linearly Separable",
                                                    "value": "linear",
                                                },
                                                {
                                                    "label": "Circles",
                                                    "value": "circles",
                                                },
                                            ],
                                            clearable=False,
                                            searchable=False,
                                            value="moons",
                                        ),
                                        drc.NamedSlider(
                                            name="Sample Size",
                                            id="slider-dataset-sample-size",
                                            min=100,
                                            max=500,
                                            step=100,
                                            marks={
                                                str(i): str(i)
                                                for i in [100, 200, 300, 400, 500]
                                            },
                                            value=300,
                                        ),
                                        drc.NamedSlider(
                                            name="Noise Level",
                                            id="slider-dataset-noise-level",
                                            min=0,
                                            max=1,
                                            marks={
                                                i / 10: str(i / 10)
                                                for i in range(0, 11, 2)
                                            },
                                            step=0.1,
                                            value=0.2,
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-threshold",
                                            min=0,
                                            max=1,
                                            value=0.5,
                                            step=0.01,
                                        ),
                                        html.Button(
                                            "Reset Threshold",
                                            id="button-zero-threshold",
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="last-card",
                                    children=[
                                        html.Div(
                                            id="svm-params",
                                            children=[
                                                drc.NamedDropdown(
                                                    name="Kernel",
                                                    id="dropdown-svm-parameter-kernel",
                                                    options=[
                                                        {
                                                            "label": "Radial basis function (RBF)",
                                                            "value": "rbf",
                                                        },
                                                        {
                                                            "label": "Linear",
                                                            "value": "linear",
                                                        },
                                                        {
                                                            "label": "Polynomial",
                                                            "value": "poly",
                                                        },
                                                        {
                                                            "label": "Sigmoid",
                                                            "value": "sigmoid",
                                                        },
                                                    ],
                                                    value="rbf",
                                                    clearable=False,
                                                    searchable=False,
                                                ),
                                                drc.NamedSlider(
                                                    name="Cost (C) for Slack",
                                                    id="slider-svm-parameter-C-power",
                                                    min=-2,
                                                    max=4,
                                                    value=0,
                                                    marks={
                                                        i: "{}".format(10 ** i)
                                                        for i in range(-2, 5)
                                                    },
                                                ),
                                                drc.FormattedSlider(
                                                    id="slider-svm-parameter-C-coef",
                                                    min=1,
                                                    max=9,
                                                    value=1,
                                                ),
                                                drc.NamedSlider(
                                                    name="Degree",
                                                    id="slider-svm-parameter-degree",
                                                    min=2,
                                                    max=10,
                                                    value=3,
                                                    step=1,
                                                    marks={
                                                        str(i): str(i)
                                                        for i in range(2, 11, 2)
                                                    },
                                                ),
                                                drc.NamedSlider(
                                                    name="Gamma",
                                                    id="slider-svm-parameter-gamma-power",
                                                    min=-5,
                                                    max=0,
                                                    value=-1,
                                                    marks={
                                                        i: "{}".format(10 ** i)
                                                        for i in range(-5, 1)
                                                    },
                                                ),
                                                drc.FormattedSlider(
                                                    id="slider-svm-parameter-gamma-coef",
                                                    min=1,
                                                    max=9,
                                                    value=5,
                                                ),
                                                html.Div(
                                                    id="shrinking-container",
                                                    children=[
                                                        html.P(children="Shrinking"),
                                                        dcc.RadioItems(
                                                            id="radio-svm-parameter-shrinking",
                                                            labelStyle={
                                                                "margin-right": "7px",
                                                                "display": "inline-block",
                                                            },
                                                            options=[
                                                                {
                                                                    "label": " Enabled",
                                                                    "value": "True",
                                                                },
                                                                {
                                                                    "label": " Disabled",
                                                                    "value": "False",
                                                                },
                                                            ],
                                                            value="True",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="logreg-params",
                                            children=[
                                                drc.NamedDropdown(
                                                    name="Regularization Type",
                                                    id="dropdown-logreg-regtype",
                                                    options=[
                                                        {"label": "L1", "value": "l1",},
                                                        {"label": "L2", "value": "l2",},
                                                        {
                                                            "label": "ElasticNet",
                                                            "value": "elasticnet",
                                                        },
                                                        {
                                                            "label": "None",
                                                            "value": "none",
                                                        },
                                                    ],
                                                    value="none",
                                                    clearable=False,
                                                    searchable=True,
                                                ),
                                                drc.NamedSlider(
                                                    name="Cost (C) for Regularization",
                                                    id="slider-logreg-C-power",
                                                    min=-2,
                                                    max=5,
                                                    value=0,
                                                    marks={
                                                        i: "{}".format(10 ** i)
                                                        for i in range(-2, 5)
                                                    },
                                                ),
                                                drc.FormattedSlider(
                                                    id="slider-logreg-C-coef",
                                                    min=1,
                                                    max=9,
                                                    value=1,
                                                ),
                                                drc.NamedSlider(
                                                    name="L1 Contribution (ElasticNet)",
                                                    id="slider-logreg-l1-ratio",
                                                    min=0,
                                                    max=1,
                                                    step=0.01,
                                                    value=0.5,
                                                    marks={
                                                        0: "0.0",
                                                        0.5: "0.5",
                                                        1: "1.0",
                                                    },
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="mlp-params",
                                            children=[
                                                drc.NamedInput(
                                                    name="Hidden layer sizes",
                                                    id="input-mlp-layers",
                                                    type="text",
                                                    value="100, 100",
                                                    placeholder="layer 1, layer 2, ...",
                                                    debounce=True,
                                                    style={"color": "inherit"},
                                                ),
                                                drc.NamedDropdown(
                                                    name="Activation Function",
                                                    id="dropdown-mlp-activation",
                                                    options=[
                                                        {"label": act, "value": act}
                                                        for act in (
                                                            "identity",
                                                            "logistic",
                                                            "tanh",
                                                            "relu",
                                                        )
                                                    ],
                                                    clearable=False,
                                                    searchable=True,
                                                    value="relu",
                                                ),
                                                drc.NamedSlider(
                                                    name="Batch Size",
                                                    id="slider-mlp-batch-size",
                                                    min=1,
                                                    included=False,
                                                ),
                                                drc.NamedSlider(
                                                    name="L2 penalty",
                                                    id="slider-mlp-penalty-power",
                                                    min=-2,
                                                    max=5,
                                                    value=0,
                                                    marks={
                                                        i: "{}".format(10 ** i)
                                                        for i in range(-2, 5)
                                                    },
                                                ),
                                                drc.FormattedSlider(
                                                    id="slider-mlp-penalty-coef",
                                                    min=1,
                                                    max=9,
                                                    value=1,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="adaboost-params",
                                            children=[
                                                drc.NamedSlider(
                                                    name="Number of Stumps",
                                                    id="slider-ab-n-estim",
                                                    min=1,
                                                    max=300,
                                                    value=50,
                                                    marks={
                                                        i: str(i)
                                                        for i in [1]
                                                        + list(range(0, 301, 50))
                                                    },
                                                )
                                            ],
                                        ),
                                        html.Div(
                                            id="xgboost-params",
                                            children=[
                                                drc.NamedSlider(
                                                    name="Number of Weak Learners",
                                                    id="slider-xg-n-estim",
                                                    min=1,
                                                    max=300,
                                                    value=50,
                                                    marks={
                                                        i: str(i)
                                                        for i in [1]
                                                        + list(range(0, 301, 50))
                                                    },
                                                ),
                                                drc.NamedSlider(
                                                    name="Minimum Leaf Size",
                                                    id="slider-xg-min-leaf",
                                                    min=1,
                                                    max=50,
                                                    value=1,
                                                    marks={
                                                        i: str(i)
                                                        for i in [1]
                                                        + list(range(0, 51, 10))
                                                    },
                                                ),
                                                drc.NamedSlider(
                                                    name="Maximum Depth of Learners",
                                                    id="slider-xg-max-depth",
                                                    min=1,
                                                    max=50,
                                                    value=3,
                                                    marks={
                                                        i: str(i)
                                                        for i in [1]
                                                        + list(range(0, 51, 10))
                                                    },
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="knn-params",
                                            children=[
                                                drc.NamedSlider(
                                                    name="k",
                                                    id="slider-knn-k",
                                                    min=1,
                                                    max=50,
                                                    value=5,
                                                    marks={
                                                        i: str(i)
                                                        for i in [1]
                                                        + list(range(10, 51, 10))
                                                    },
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)


@app.callback(Output("svm-params", "style"), [Input("dropdown-select-model", "value")])
def show_svm_params(model):
    if model == "SVM":
        return {"visibility": "visible"}
    else:
        return {"display": "none"}


@app.callback(
    Output("logreg-params", "style"), [Input("dropdown-select-model", "value")]
)
def show_logreg_params(model):
    if model == "LogReg":
        return {"visibility": "visible"}
    else:
        return {"display": "none"}


@app.callback(Output("mlp-params", "style"), [Input("dropdown-select-model", "value")])
def show_mlp_params(model):
    if model == "MLP":
        return {"visibility": "visible"}
    else:
        return {"display": "none"}


@app.callback(
    Output("adaboost-params", "style"), [Input("dropdown-select-model", "value")]
)
def show_adaboost_params(model):
    if model == "ABoost":
        return {"visibility": "visible"}
    else:
        return {"display": "none"}


@app.callback(
    Output("xgboost-params", "style"), [Input("dropdown-select-model", "value")]
)
def show_adaboost_params(model):
    if model == "XGBoost":
        return {"visibility": "visible"}
    else:
        return {"display": "none"}


@app.callback(Output("knn-params", "style"), [Input("dropdown-select-model", "value")])
def show_knn_params(model):
    if model == "kNN":
        return {"visibility": "visible"}
    else:
        return {"display": "none"}


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "marks"),
    [Input("slider-svm-parameter-gamma-power", "value")],
)
def update_slider_svm_parameter_gamma_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-svm-parameter-C-coef", "marks"),
    [Input("slider-svm-parameter-C-power", "value")],
)
def update_slider_svm_parameter_C_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-threshold", "value"), [Input("button-zero-threshold", "n_clicks")],
)
def reset_threshold_center(n_clicks):
    return 0.499


# Disable Sliders if kernel not in the given list
@app.callback(
    Output("slider-svm-parameter-degree", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_degree(kernel):
    return kernel != "poly"


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_coef(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("slider-svm-parameter-gamma-power", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_power(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("slider-logreg-C-coef", "marks"), [Input("slider-logreg-C-power", "value")],
)
def update_slider_logreg_C_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


# disable elasticNet slider if other regularization
@app.callback(
    Output("slider-logreg-l1-ratio", "disabled"),
    [Input("dropdown-logreg-regtype", "value")],
)
def disable_slider_logreg_l1_ratio(reg_type):
    return reg_type != "elasticnet"


# set max, value, and marks for batch size slider for MLP
@app.callback(
    Output("slider-mlp-batch-size", "max"),
    [Input("slider-dataset-sample-size", "value")],
)
def update_mlp_batch_size_max(sample_size):
    return sample_size


@app.callback(
    Output("slider-mlp-batch-size", "value"),
    [Input("slider-dataset-sample-size", "value")],
)
def update_mlp_batch_size_value(sample_size):
    return sample_size // 2


@app.callback(
    Output("slider-mlp-batch-size", "marks"),
    [Input("slider-dataset-sample-size", "value")],
)
def update_mlp_batch_size_marks(sample_size):
    return {1: "1", sample_size: str(sample_size)}


@app.callback(
    Output("slider-mlp-penalty-coef", "marks"),
    [Input("slider-mlp-penalty-power", "value")],
)
def update_slider_mlp_penalty_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("div-graphs", "children"),
    [
        Input("dropdown-select-model", "value"),
        Input("dropdown-svm-parameter-kernel", "value"),
        Input("slider-svm-parameter-degree", "value"),
        Input("slider-svm-parameter-C-coef", "value"),
        Input("slider-svm-parameter-C-power", "value"),
        Input("slider-svm-parameter-gamma-coef", "value"),
        Input("slider-svm-parameter-gamma-power", "value"),
        Input("dropdown-select-dataset", "value"),
        Input("slider-dataset-noise-level", "value"),
        Input("radio-svm-parameter-shrinking", "value"),
        Input("slider-threshold", "value"),
        Input("slider-dataset-sample-size", "value"),
        Input("dropdown-logreg-regtype", "value"),
        Input("slider-logreg-C-coef", "value"),
        Input("slider-logreg-C-power", "value"),
        Input("slider-logreg-l1-ratio", "value"),
        Input("input-mlp-layers", "value"),
        Input("dropdown-mlp-activation", "value"),
        Input("slider-mlp-batch-size", "value"),
        Input("slider-mlp-penalty-coef", "value"),
        Input("slider-mlp-penalty-power", "value"),
        Input("slider-ab-n-estim", "value"),
        Input("slider-xg-n-estim", "value"),
        Input("slider-xg-min-leaf", "value"),
        Input("slider-xg-max-depth", "value"),
        Input("slider-knn-k", "value"),
    ],
)
def update_svm_graph(
    model,
    kernel,
    degree,
    C_coef,
    C_power,
    gamma_coef,
    gamma_power,
    dataset,
    noise,
    shrinking,
    threshold,
    sample_size,
    logreg_reg_type,
    logreg_C_coef,
    logreg_C_power,
    logreg_l1_ratio,
    mlp_layers,
    mlp_activation,
    mlp_batch_size,
    mlp_l2_coef,
    mlp_l2_pow,
    aboost_n_estimators,
    xg_n_estimators,
    xg_min_leaf_size,
    xg_max_depth,
    knn_k,
):
    h = 0.3  # step size in the mesh

    # Data Pre-processing
    X, y = generate_data(n_samples=sample_size, dataset=dataset, noise=noise)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min = X[:, 0].min() - 1.5
    x_max = X[:, 0].max() + 1.5
    y_min = X[:, 1].min() - 1.5
    y_max = X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    if model == "SVM":
        C = C_coef * 10 ** C_power
        gamma = gamma_coef * 10 ** gamma_power

        if shrinking == "True":
            flag = True
        else:
            flag = False

        clf = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            shrinking=flag,
            probability=True,
        )

    elif model == "LogReg":
        C = logreg_C_coef * 10 ** logreg_C_power

        if logreg_reg_type == "none" or logreg_reg_type == "elasticnet":
            solver = "saga"
        else:
            solver = "liblinear"

        clf = LogisticRegression(
            penalty=logreg_reg_type, C=C, l1_ratio=logreg_l1_ratio, solver=solver
        )

    elif model == "LDA":
        clf = LinearDiscriminantAnalysis()

    elif model == "QDA":
        clf = QuadraticDiscriminantAnalysis()

    elif model == "MLP":
        hidden_layers = tuple(map(int, mlp_layers.split(", ")))
        l2_penalty = mlp_l2_coef * 10 ** mlp_l2_pow

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=mlp_activation,
            batch_size=mlp_batch_size,
            alpha=l2_penalty,
        )

    elif model == "DTree":
        clf = DecisionTreeClassifier()

    elif model == "RForest":
        clf = RandomForestClassifier()

    elif model == "ABoost":
        clf = AdaBoostClassifier(n_estimators=aboost_n_estimators)

    elif model == "XGBoost":
        clf = GradientBoostingClassifier(
            n_estimators=xg_n_estimators,
            min_samples_leaf=xg_min_leaf_size,
            max_depth=xg_max_depth,
        )

    elif model == "kNN":
        clf = KNeighborsClassifier(n_neighbors=knn_k)

    else:
        raise ValueError(f"Unsupported model: {model}")
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    prediction_figure = figs.serve_prediction_plot(
        model=clf,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        Z=Z,
        xx=xx,
        yy=yy,
        mesh_step=h,
        threshold=threshold,
    )

    roc_figure = figs.serve_roc_curve(model=clf, X_test=X_test, y_test=y_test)

    confusion_figure = figs.serve_pie_confusion_matrix(
        model=clf, X_test=X_test, y_test=y_test, Z=Z, threshold=threshold
    )

    return [
        html.Div(
            id="svm-graph-container",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="graph-sklearn-svm", figure=prediction_figure),
                style={"display": "none"},
            ),
        ),
        html.Div(
            id="graphs-container",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-pie-confusion-matrix", figure=confusion_figure
                    ),
                ),
            ],
        ),
    ]


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
