from agents import Agent, Runner, ModelSettings, function_tool, InputGuardrail, GuardrailFunctionOutput
from agents.extensions.visualization import draw_graph
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import os

# Load datasets from uploaded files
df_accesso = pd.read_csv("Accesso_English.csv")
df_stipendi = pd.read_csv("Stipendi_English.csv")
df_reddito = pd.read_csv("Reddito_English.csv")
df_pendolarismo = pd.read_csv("Pendolarismo_English.csv")

from typing import Dict
from agents import function_tool
@function_tool
def process_data(query: str) -> Dict:
    """
    Interprets a user query and maps it to the appropriate dataset and column.
    Returns:
    - dataset_name: name of the dataset
    - column: suggested numeric column to analyze
    - columns: list of all available columns
    """
    query = query.lower().strip()
    # Keyword mapping → dataset, dataframe, default column
    keyword_mapping = {
        "accesso": {
            "keywords": [ "region of residence", "residential", "region of domicile", "resident area",
                         "administration", "agency", "institution", "public entity", "gender", "sex", "demographic gender",
                          "age max", "maximum age", "oldest", "age min", "minimum age", "youngest", "authentication method",
                          "login method", "access mode", "authentication type"
                "access", "badge", "entry", "entrance", "exit", "presence", "attendance", "in-out"
            ],
            "dataset": "accesso",
            "dataframe": df_accesso,
            "default_column": "num_occurrences"
        },
        "stipendi": {
            "keywords": [
                "number of salaries", "number of people", "office", "workplace", "bureau", "agency", "headquarters", "station",
                "municipality", "town", "city", "borough", "district", "community", "administration", "management", "governance", "supervision",
                "executive", "authority", "minimum age", "lowest age", "youngest age", "entry age", "starting age", "legal age", "maximum age",
                "highest age", "oldest age", "upper age", "peak age", "limit age", "gender", "sex", "identity", "gender identity", "biological sex",
                "gender category", "payment method", "payment type", "mode of payment", "payment option", "transaction method", "payment form", "salary",
                "wage", "pay", "remuneration", "earnings", "income", "payment", "age"
            ],
            "dataset": "stipendi",
            "dataframe": df_stipendi,
            "default_column": "number"
        },
        "reddito": {
            "keywords": [
                "bracket", "tax bracket", "economic level", "earning class", "range", "taxes", "income level", "wage", "pay", "payment", "accredit",
                "remuneration", "compensation", "sector", "field", "domain", "division", "region of residence", "area of residence", "location", "district",
                "territory", "gender", "sex", "identity", "gender identity", "biological sex", "gender category", "age_min", "minimum age", "lowest age",
                "youngest age", "entry age", "starting age", "age max", "maximum age", "highest age", "oldest age", "upper age", "limit age", "maximum taxation rate",
                "highest tax rate", "top tax rate", "maximum taxation", "tax ceiling", "upper tax rate", "income_bracket_min", "minimum income range",
                "lowest salary range", "lowest income group", "bottom earnings bracket", "starting income bracket", "income bracket",
                "maximum income range", "highest salary range","salary range" ,"top income group", "peak earnings bracket", "upper income bracket", "employee_count",
                "number of employees", "staff size", "workforce count", "personnel number", "headcount","tax rate",
            ],
            "dataset": "reddito",
            "dataframe": df_reddito,
            "default_column": "number"
        },
        "pendolarismo": {
            "keywords": [ "province_of_office","office_province","province_location","administrative_province","province_of_employment",
    "office_municipality","municipality_of_office","office_town","local_municipality","workplace_municipality",
    "same_municipality","resides_and_works_same_municipality","local_worker_flag", "same_town_flag","local_residence_match",
    "organization","employer","institution","entity","administrative_body",
    "employee_count","staff_total","workforce_number","number_of_employees","personnel_count",
    "distance_min_km","minimum_distance_km","shortest_commute_km","min_travel_distance","commute_lower_bound_km",
    "distance_max_km","maximum_distance_km","longest_commute_km","max_travel_distance","commute_upper_bound_km"
            ],
            "dataset": "pendolarismo",
            "dataframe": df_pendolarismo,
            "default_column": "number_of_employees"
        }
    }

    "province_of_office","office_province","province location","administrative province","province of employment",
    "office municipality","municipality of office","office town","local municipality","workplace municipality",
    "same municipality","resides and works same municipality","local worker flag", "same town flag","local residence match",
    "organization","employer","institution","entity","administrative_body",
    "employee count","staff total","workforce number","number of employees","personnel count",
    "distance min km","minimum distance","shortest commute","minimum travel distance","commute lower bound",
    "distance max km","maximum distance","longest commute","maximum travel distance","commute upper bound"

    # Match query with keywords
    for key, info in keyword_mapping.items():
        if any(kw in query for kw in info["keywords"]):
            df = info["dataframe"]
            dataset = info["dataset"]
            default_col = info["default_column"]
            # Fallback if column is not present
            if default_col not in df.columns:
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                default_col = numeric_cols[0] if numeric_cols else df.columns[0]
            return {
                "dataset_name": dataset,
                "column": default_col,
                "columns": list(df.columns),
                "description": f"Matched to '{dataset}' dataset. Default column for analysis: '{default_col}'."
            }

    # No match found
    return {
        "error": "❌ No recognized dataset. Please specify if you mean 'stipendi', 'accesso', 'reddito', or 'pendolarismo'."
    }

from typing import Dict, Any

@function_tool
def analyze_data(dataset_name: str, column: str) -> Dict[str, Any]:
    """
    Performs mathematical and statistical analysis on a given dataset and numeric column.
    Parameters:
    - dataset_name: one of ['stipendi', 'accesso', 'reddito', 'pendolarismo']
    - column: the numeric column to analyze
    Returns a dictionary of results: mean, median, min, max, std, quantiles, outliers, skewness, etc.
    """
    # Mapping name → DataFrame
    dataset_map = {
        "stipendi": df_stipendi,
        "accesso": df_accesso,
        "reddito": df_reddito,
        "pendolarismo": df_pendolarismo
    }

    if dataset_name not in dataset_map:
        return {"error": f"❌ Dataset '{dataset_name}' not found."}

    df = dataset_map[dataset_name]
    if column not in df.columns:
        return {"error": f"❌ Column '{column}' not present in dataset '{dataset_name}'."}

    # Clean and select column
    series = pd.to_numeric(df[column], errors='coerce').dropna()
    if series.empty:
        return {"error": f"❌ Column '{column}' does not contain valid numeric values."}

    # Statistical calculations
    summary = {
        "count": int(series.count()),
        "mean": round(series.mean(), 2),
        "median": round(series.median(), 2),
        "std_dev": round(series.std(), 2),
        "min": round(series.min(), 2),
        "max": round(series.max(), 2),
        "quantiles": {
            "Q1 (25%)": round(series.quantile(0.25), 2),
            "Q2 (50%)": round(series.quantile(0.50), 2),
            "Q3 (75%)": round(series.quantile(0.75), 2),
            "90th percentile": round(series.quantile(0.90), 2),
            "95th percentile": round(series.quantile(0.95), 2),
            "99th percentile": round(series.quantile(0.99), 2),
        },
        "skewness": round(series.skew(), 2),
        "kurtosis": round(series.kurt(), 2),
        "range": round(series.max() - series.min(), 2),
        "interquartile_range (IQR)": round(series.quantile(0.75) - series.quantile(0.25), 2),
    }

    # Outlier detection using IQR method
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_info = {
        "num_outliers": int(outliers.count()),
        "outlier_percentage": round((outliers.count() / series.count()) * 100, 2),
        "outlier_bounds": {
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
        }
    }

    # Frequencies of unique values (if there are few)
    unique_values = series.value_counts()
    if len(unique_values) <= 10:
        value_frequencies = unique_values.to_dict()
    else:
        value_frequencies = "Too many unique values to display"

    # Final output
    return {
        "dataset": dataset_name,
        "column": column,
        "summary_statistics": summary,
        "outliers": outlier_info,
        "value_frequencies": value_frequencies
    }


@function_tool
def grouped_analysis(
    dataset_name: str,
    group_by: str,
    target_column: str,
    agg_function: str
) -> Dict[str, Any]:
    """
    Groups the dataset by a categorical column and computes an aggregation on a numeric column.
    Parameters:
    - dataset_name: one of ['stipendi', 'accesso', 'reddito', 'pendolarismo']
    - group_by: categorical column used to group the data (e.g., 'administration', 'region', 'gender', 'age_range')
    - target_column: numeric column on which to apply the aggregation
    - agg_function: aggregation function ('mean', 'median', 'max', 'min', 'sum', 'std', 'count')
    Returns:
    - Dictionary with top and bottom group + full grouped results
    """
    # Dataset mapping
    dataset_map = {
        "stipendi": df_stipendi,
        "accesso": df_accesso,
        "reddito": df_reddito,
        "pendolarismo": df_pendolarismo
    }

    if dataset_name not in dataset_map:
        return {"error": f"❌ Dataset '{dataset_name}' not found."}

    df = dataset_map[dataset_name]

    # Handle common synonyms and variations in column names
    column_synonyms = {
        # Region-related
        "region": ["region", "regions", "area", "location", "geographical_area", "territory"],
        "north": ["north", "northern", "nord"],
        "south": ["south", "southern", "sud"],

        # Gender-related
        "gender": ["gender", "sex", "m/f"],
        "male": ["male", "m", "men", "man"],
        "female": ["female", "f", "women", "woman"],

        # Age-related
        "age": ["age", "age_range", "age_group", "ages", "years"],

        # Authentication-related
        "auth_method": ["auth_method", "authentication", "authentication_method", "login_method", "access_method"],
        "spid": ["spid", "digital_identity"],
        "cie": ["cie", "electronic_id", "id_card"],

        # Administration-related
        "administration": ["administration", "admin", "public_administration", "institution", "organization"],

        # Salary-related
        "salary": ["salary", "wage", "income", "payment", "stipendio"],

        # Access-related
        "access": ["access", "accesses", "entry", "entries", "login"],
    }

    # Try to find the right column based on synonyms if the exact match fails
    if group_by not in df.columns:
        found = False
        for actual_col, synonyms in column_synonyms.items():
            if group_by.lower() in synonyms and actual_col in df.columns:
                group_by = actual_col
                found = True
                break

        if not found:
            # Try partial matches if no exact synonym found
            possible_columns = []
            for col in df.columns:
                if group_by.lower() in col.lower():
                    possible_columns.append(col)

            if possible_columns:
                group_by = possible_columns[0]  # Choose the first match
            else:
                return {"error": f"❌ Grouping column '{group_by}' not found in dataset."}

    # Similar synonym handling for target column
    if target_column not in df.columns:
        found = False
        for actual_col, synonyms in column_synonyms.items():
            if target_column.lower() in synonyms and actual_col in df.columns:
                target_column = actual_col
                found = True
                break

        if not found:
            # Try partial matches if no exact synonym found
            possible_columns = []
            for col in df.columns:
                if target_column.lower() in col.lower():
                    possible_columns.append(col)

            if possible_columns:
                target_column = possible_columns[0]  # Choose the first match
            else:
                return {"error": f"❌ Target column '{target_column}' not found in dataset."}

    # Support additional aggregation synonyms
    agg_synonyms = {
        "mean": ["mean", "average", "avg"],
        "median": ["median", "middle", "med"],
        "max": ["max", "maximum", "highest", "largest"],
        "min": ["min", "minimum", "lowest", "smallest"],
        "sum": ["sum", "total"],
        "std": ["std", "standard deviation", "deviation", "variance"],
        "count": ["count", "number", "quantity", "frequency"]
    }

    # Normalize aggregation function
    agg_function_normalized = None
    for agg, synonyms in agg_synonyms.items():
        if agg_function.lower() in synonyms:
            agg_function_normalized = agg
            break

    if not agg_function_normalized:
        return {"error": f"❌ Aggregation '{agg_function}' not supported. Choose from: mean, median, max, min, sum, std, count."}

    agg_function = agg_function_normalized

    # Handle special cases: filter by specific region, gender, etc.
    filter_condition = None
    if group_by.lower() == "region" and any(region in ["north", "south", "center", "islands"] for region in df["region"].unique()):
        north_regions = ["north", "northern", "nord"]
        south_regions = ["south", "southern", "sud"]
        center_regions = ["center", "central", "centro"]
        islands_regions = ["islands", "isole"]

        if any(region.lower() in north_regions for region in [target_column]):
            filter_condition = df["region"].str.contains("North", case=False, na=False)
        elif any(region.lower() in south_regions for region in [target_column]):
            filter_condition = df["region"].str.contains("South", case=False, na=False)
        elif any(region.lower() in center_regions for region in [target_column]):
            filter_condition = df["region"].str.contains("Center", case=False, na=False)
        elif any(region.lower() in islands_regions for region in [target_column]):
            filter_condition = df["region"].str.contains("Islands", case=False, na=False)

    # Apply filter if needed
    if filter_condition is not None:
        df = df[filter_condition]
        if df.empty:
            return {"error": "❌ No data matching the filter criteria."}

    # Data cleaning
    df_clean = df[[group_by, target_column]].dropna()

    # Handle the case where target column might need special processing
    if df_clean[target_column].dtype == 'object':
        # Try to convert strings to numeric if possible
        df_clean[target_column] = pd.to_numeric(df_clean[target_column].str.replace('[^0-9.]', '', regex=True), errors='coerce')
    else:
        df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors='coerce')

    df_clean = df_clean.dropna()

    if df_clean.empty:
        return {"error": "❌ No valid data after cleaning."}

    # Add 'count' to supported aggregations
    agg_map = {
        "mean": df_clean.groupby(group_by)[target_column].mean(),
        "median": df_clean.groupby(group_by)[target_column].median(),
        "max": df_clean.groupby(group_by)[target_column].max(),
        "min": df_clean.groupby(group_by)[target_column].min(),
        "sum": df_clean.groupby(group_by)[target_column].sum(),
        "std": df_clean.groupby(group_by)[target_column].std(),
        "count": df_clean.groupby(group_by)[target_column].count()
    }

    result_series = agg_map[agg_function].dropna().sort_values(ascending=False)

    if result_series.empty:
        return {"error": f"❌ No data available for {agg_function} of {target_column} grouped by {group_by}."}

    top_group = result_series.idxmax()
    bottom_group = result_series.idxmin()

    # Enhance results with more statistics and insights
    grouped_stats = df_clean.groupby(group_by)[target_column].agg(['count', 'mean', 'sum', 'min', 'max'])
    total_records = df_clean[target_column].count()
    total_sum = df_clean[target_column].sum()

    # Calculate percentage distribution
    percentages = {}
    if agg_function == "count":
        for group, count in result_series.items():
            percentages[str(group)] = round((count / total_records) * 100, 2)
    elif agg_function == "sum":
        for group, sum_val in result_series.items():
            percentages[str(group)] = round((sum_val / total_sum) * 100, 2)

    # Create a more detailed result
    return {
        "dataset": dataset_name,
        "aggregation": {
            "group_by": group_by,
            "target_column": target_column,
            "agg_function": agg_function
        },
        "top_group": {
            "group": str(top_group),
            "value": round(float(result_series[top_group]), 2),
            "percentage": percentages.get(str(top_group), None),
            "count": int(grouped_stats.loc[top_group, 'count'])
        },
        "bottom_group": {
            "group": str(bottom_group),
            "value": round(float(result_series[bottom_group]), 2),
            "percentage": percentages.get(str(bottom_group), None),
            "count": int(grouped_stats.loc[bottom_group, 'count'])
        },
        "overall_stats": {
            "total_groups": len(result_series),
            "total_records": int(total_records),
            "total_sum": round(float(total_sum), 2),
            "overall_mean": round(float(df_clean[target_column].mean()), 2)
        },
        "full_grouped_result": {str(k): round(float(v), 2) for k, v in result_series.to_dict().items()}
    }

from typing import Dict
from agents import function_tool
from sentence_transformers import SentenceTransformer, util

# Load the model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Dataset info: descriptions only (no keyword mapping)
dataset_info = {
    "accesso": {
        "description": "authentication method, login events, demographic age and gender, region of access",
        "dataframe": df_accesso
    },
    "stipendi": {
        "description": "salaries, wages, number of employees, region, gender, administration",
        "dataframe": df_stipendi
    },
    "reddito": {
        "description": "income brackets, tax ranges, gender, age, employment data",
        "dataframe": df_reddito
    },
    "pendolarismo": {
        "description": "commuting distances, provinces of work and residence, employee count",
        "dataframe": df_pendolarismo
    }
}

@function_tool
def process_data(query: str) -> Dict:
    """
    Select the most relevant dataset and columns based on a natural-language query, extract and compute all necessary data points or aggregates, and provide a clear, data-driven answer to the user’s question by following this structured developer workflow:

    1. Explicitly State the Goal
       The primary purpose of this tool is to retrieve and process the data needed to answer the user’s question accurately.

    2. Deeply Understand the Query
       - Parse the user’s intent, identifying key variables, time frames, filters, comparisons and metrics.

    3. Investigate Available Data
       - Review `dataset_info` keys and each description.
       - Remember that you have to interface only with this four datasets:'/content/Accesso_English.csv','/content/Pendolarismo_English.csv','/content/Reddito_English.csv','/content/Stipendi_English.csv'.
       - Confirm schemas, column names, data types and dimensions of the four datasets.
       - Always verify that any referenced column exists in the selected dataset.

    4. Plan Data Extraction and Computation
       - Determine whether raw values or aggregated metrics (sum, mean, count, distribution) are needed.
       - Identify any grouping or filtering columns (e.g., region, year, category).

    5. Implement Matching Logic
       - Embed query and dataset descriptions using SentenceTransformer.
       - Compute cosine similarities to choose the most relevant dataset.
       - Within that dataset, select the best numeric column(s) and any categorical column(s) for grouping (only when is needed) or filtering.
       - Extract raw data or compute aggregates as dictated by the query.
       - Ensure you know exactly which dataset, columns, and filters are required.

    6. Handle Non‑Specific or Ambiguous Queries
       - Once you have followed the previous step and you see that the query lacks specificity (e.g., unexisting column,missing period, metric, or grouping).
       - Respond always that the query is not well formulated, explain why and ask if the user want to be reformulate a more specific question.

    7. Generate the Data‑Driven Prompt
       - Compose a concise Prompt that gives to the generate python code tools the best probability to be right.

    8. Validate and Test
       - Confirm the chosen `dataset_name` matches one of the known files.
       - Verify each referenced column exists and has the expected type.
       - Ensure computed metrics are correct and fully support the narrative.
       - Handle edge cases (empty results, no suitable columns) by informing the user or requesting additional input.

    """
    # Step 1: Embed query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # Step 2: Match dataset by semantic similarity
    dataset_scores = {
        name: float(util.cos_sim(query_embedding, embedding_model.encode(info["description"], convert_to_tensor=True)))
        for name, info in dataset_info.items()
    }
    best_dataset = max(dataset_scores, key=dataset_scores.get)
    df = dataset_info[best_dataset]["dataframe"]

    # Step 3: Match best numeric column
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    best_numeric = None
    if numeric_cols:
        best_numeric = max(
            numeric_cols,
            key=lambda col: float(util.cos_sim(query_embedding, embedding_model.encode(col, convert_to_tensor=True)))
        )

    # Step 4: Only match group-by column if grouping is explicitly mentioned
    best_groupby = None
    if " by " in query.lower() or "group" in query.lower():
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        if categorical_cols:
            best_groupby = max(
                categorical_cols,
                key=lambda col: float(util.cos_sim(query_embedding, embedding_model.encode(col, convert_to_tensor=True)))
            )

    return {
        "dataset_name": best_dataset,
        "column": best_numeric,
        "group_by": best_groupby,
        "columns": list(df.columns),
        "description": (
            f"Matched to '{best_dataset}' with semantic similarity {round(dataset_scores[best_dataset], 4)}.\n"
            f"Suggested numeric column: '{best_numeric}'\n"
            f"Suggested group-by column: '{best_groupby}'" if best_groupby else ""
        )
    }

from agents import function_tool

@function_tool
def build_visualization_prompt(query: str) -> str:
    """
    Generates a structured visualization prompt based on the DataProcessor Agent query.
    The prompt describes the chart type, involved variable(s), and the context.
    """

    query = query.lower()

    if any(word in query for word in ["trend", "time", "evoluzione", "crescita", "andamento"]):
        chart_type = "line chart"
    elif any(word in query for word in ["confronta", "comparison", "raffronto", "differenza"]):
        chart_type = "bar chart"
    elif any(word in query for word in ["distribuzione", "distribution", "frequenza"]):
        chart_type = "histogram"
    elif any(word in query for word in ["percentuale", "quota", "ripartizione"]):
        chart_type = "pie chart"
    else:
        chart_type = "bar chart"  # default

    if "stipendi" in query or "salary" in query:
        variable = "stipendi"
    elif "accessi" in query or "badge" in query:
        variable = "accessi"
    elif "pendolarismo" in query:
        variable = "pendolarismo"
    elif "reddito" in query:
        variable = "reddito"
    else:
        variable = "una variabile rilevante"

    prompt = (
        f"Create a {chart_type} to visualize '{variable}' "
        f"based on the query: '{query}'. Use an appropriate X-axis (category or time) "
        "and Y-axis (numeric metric)."
    )

    return prompt

from agents import function_tool

@function_tool
def generate_python_plot_code(prompt: str) -> str:
    """
    Converts a visualization prompt into Python code using pandas and matplotlib.
    """

    prompt = prompt.lower()

    if "line chart" in prompt:
        plot_code = "df.plot(x='DATE', y='VALUE', kind='line')"
    elif "bar chart" in prompt:
        plot_code = "df.plot(x='CATEGORY', y='VALUE', kind='bar')"
    elif "pie chart" in prompt:
        plot_code = "df.groupby('CATEGORY')['VALUE'].sum().plot.pie(autopct='%1.1f%%')"
    elif "histogram" in prompt:
        plot_code = "df['VALUE'].plot.hist(bins=20)"
    else:
        plot_code = "# Unknown chart type"

    code = f"""
import pandas as pd
import matplotlib.pyplot as plt

# Assumes df is already defined or loaded before this code runs

{plot_code}

plt.title("Generated Chart")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.grid(True)
plt.tight_layout()
plt.show()
"""
    return code.strip()

from agents import function_tool
import io
import contextlib
import base64
import matplotlib.pyplot as plt

@function_tool
def execute_code(code: str) -> dict:
    """
    Executes Python code and returns both printed output and a base64-encoded PNG image
    if a chart is generated.
    """

    stdout = io.StringIO()
    image_data = None

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {})  # run code in a clean namespace

        # Capture current matplotlib plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        output = stdout.getvalue().strip()

    except Exception as e:
        output = f"❌ Error during execution: {str(e)}"
        image_data = None

    return {
        "output": output,
        "image_base64": image_data
    }

# Memory buffer setup
memory_buffer = []
max_memory_length = 20

def update_memory_buffer(memory_buffer, user_prompt: str, agent_response: str):
    memory_buffer.append({
        "user": user_prompt,
        "agent": agent_response
    })
    if len(memory_buffer) > max_memory_length:
        memory_buffer.pop(0)

async def triage_with_memory(user_query: str, agent):
    """
    Enhances the prompt with summarized memory,
    detects duplicate user questions, runs the agent,
    and logs both the question and the answer as structured memory.
    """

    # Step 1: Normalize current query
    normalized_query = user_query.strip().lower()

    # Step 2: Detect repeat (if memory has structured entries)
    if all(isinstance(entry, dict) for entry in memory_buffer):
        user_prompts = [entry["user"].strip().lower() for entry in memory_buffer]
    else:
        user_prompts = [
            entry.split("\n")[0].replace("[User] ", "").strip().lower()
            for entry in memory_buffer
            if isinstance(entry, str)
        ]

    if normalized_query in user_prompts:
        print("⚠️ You've already asked this! The agent will still respond, but consider rephrasing.")

    # Step 3: Build summarized context
    summarized_context = "Conversation so far:\n"
    for entry in memory_buffer[-3:]:
        if isinstance(entry, dict):
            summarized_context += f"- User asked: {entry['user']} → Agent answered: {entry['agent']}\n"
        elif isinstance(entry, str):
            summarized_context += f"- {entry}\n"

    summarized_context += f"\nNow the user asks:\n\"{user_query}\""

    # Step 4: Run agent
    result = await Runner.run(agent, summarized_context)
    response = result.final_output

    # Step 5: Log interaction as structured memory
    memory_buffer.append({
        "user": user_query,
        "agent": response
    })

    # Step 6: Return answer
    return response

def print_memory():
    print("\n--- MEMORY STATE ---")
    for i, line in enumerate(memory_buffer):
        print(f"{i+1}. {line}")
    print("---------------------")


visualization_agent = Agent(
    name="Visualizer",
    instructions=(
        "You are a visualization expert. Your task is to take a user's query and return a chart.\n"
        "First, transform the user query into a structured visualization prompt using `build_visualization_prompt`.\n"
        "Then, convert that prompt into valid Python code using `generate_python_plot_code`.\n"
        "Finally, execute the Python code using `execute_code` to generate a visualization.\n"
        "If the execution returns an image, return the image directly. If an error occurs, return the error.\n"
        "Always follow this 3-step path without asking the user any clarification.\n"
        "Do not explain the chart unless explicitly asked."
    ),
    tools=[
        build_visualization_prompt,
        generate_python_plot_code,
        execute_code
    ]
)

# ----- DATA PROCESSING AGENT -----
data_agent = Agent(
    name="DataProcessor",
    instructions=(
        "You are the data analysis agent.\n\n"
        "You MUST follow these rules strictly:\n\n"
        "1. ALWAYS call the 'process_data' tool to extract:\n"
        "   - dataset_name\n"
        "   - column (numeric)\n\n"
        "2. If the user is asking for statistics on a column (e.g., average, median, percentiles, outliers, etc.),\n"
        "   IMMEDIATELY call 'analyze_data' with the extracted dataset_name and column.\n\n"
        "3. If the user is asking for a comparison across groups (e.g., per region, per administration),\n"
        "   call 'grouped_analysis' and pass:\n"
        "   - dataset_name (from 'process_data')\n"
        "   - group_by (the grouping column mentioned by the user)\n"
        "   - target_column (the numeric column to analyze)\n"
        "   - agg_function (e.g., 'mean', 'max', 'sum', etc.)\n\n"
        "RULES:\n"
        "- ONLY return the specific result the user asked for.\n"
        "- NEVER ask the user anything.\n"
        "- NEVER explain or describe your steps.\n"
        "- NEVER add context, reasoning, or metadata.\n"
        "- Your output must be minimal, direct, and clean.\n"
        "- If input is ambiguous, make a reasonable assumption. DO NOT ask for clarification."
    ),
    tools=[
        process_data,
        analyze_data,
        grouped_analysis
    ]
)


# ----- CONVERSATION AGENT (ORCHESTRATOR) -----

triage_agent = Agent(
    name="ConversationAgent",
    instructions=(
    "You are a conversational interface that uses memory.\n"
    "When interpreting the user's query, always consider previous exchanges if provided.\n"
    "The memory contains past questions and answers. Use them to resolve references, pronouns, or follow-ups.\n"
    "Route the query to the most appropriate specialized agent (DataProcessor or Visualizer) "
    "based on the nature of the user's request."
    "Be consistent in logic, and avoid repeating the same operations unless requested."

    ),
    handoffs=[data_agent, visualization_agent],
    tools=[]  # No direct tools needed here
)

draw_graph(triage_agent)