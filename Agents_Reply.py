from agents import Agent, Runner, ModelSettings, function_tool, InputGuardrail, GuardrailFunctionOutput
#from agents.extensions.visualization import draw_graph
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
import io
import contextlib

# Paste your real API key here (you can also use getpass for extra security)
os.environ["OPENAI_API_KEY"] = "sk-..."
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load datasets from uploaded files
df_accesso = pd.read_csv("Accesso_English.csv")
df_stipendi = pd.read_csv("Stipendi_English.csv")
df_reddito = pd.read_csv("Reddito_English.csv")
df_pendolarismo = pd.read_csv("Pendolarismo_English.csv")
import os
import pandas as pd
from typing import Dict
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from agents import function_tool

# Load environment variables
load_dotenv()

# Load embedding model
embedding_model = OpenAIEmbedding()

# Assume df_accesso, df_pendolarismo, df_reddito, df_stipendi are already loaded globally
dataset_info = {
    "accesso": {
        "description": "Region of residence, type of administration, gender, age groups, authentication methods (e.g., SPID), and total number of employees per group",
        "dataframe": df_accesso
    },
    "pendolarismo": {
        "description": "Commuting distances (min, max, average), province and municipality of the office, organization name, whether residence matches office location, and number of commuting employees",
        "dataframe": df_pendolarismo
    },
    "reddito": {
        "description": "Income bracket ranges, maximum tax rates, employee age and gender distribution, economic sector, region of residence, and employee count per bracket",
        "dataframe": df_reddito
    },
    "stipendi": {
        "description": "Payment methods, salary counts, gender and age ranges, type of administration, and office municipality",
        "dataframe": df_stipendi
    }
}

@function_tool
def processo_data(query: str) -> Dict:
    """
    Select the most relevant dataset and columns based on a natural-language query, extract and compute all necessary data points or aggregates, and provide a clear, data-driven answer by following this updated developer workflow using LlamaIndex for document-level understanding.

    1. Explicitly State the Goal
       - Retrieve and interpret relevant information from the four datasets to answer the user’s query with clarity and correctness.

    2. Deeply Understand the Query
       - Analyze the user’s intent, identifying: metrics, comparisons, dimensions, filters, and any implicit requests (e.g. grouping, distributions, summaries).

    3. Investigate Available Data
       - Work only with these datasets:
        -"Accesso_English.csv"
        -"Stipendi_English.csv"
        -"Reddito_English.csv"
        -"Pendolarismo_English.csv"
       - Reference the ⁠ dataset_info ⁠ dictionary for schema descriptions.
       - Always validate that referenced columns exist and data types match expectations.

    4. Plan Data Extraction and Computation
       - Based on user intent, determine whether the output requires raw data, aggregated values (sum, mean, count), distributions, or comparisons.
       - Identify possible grouping variables (e.g., region, gender, office location) and appropriate filters (e.g., high income, long commute).

    5. Implement Matching Logic
       - Use LlamaIndex to embed and semantically compare the *full dataset descriptions as documents* with the query.
       - Avoid sentence-level transformers; leverage the semantic document matching capabilities of LlamaIndex.
       - Select the most relevant dataset based on document-level similarity scoring.
       - Within the selected dataset, semantically match the best numeric column (and grouping column, if needed).

    6. Handle Non-Specific or Ambiguous Queries
       - If key query elements are vague, missing, or unresolvable (e.g., column not found, unclear metric), return an error message.
       - Always explain why the query is insufficient and offer the user the chance to reformulate their question with clearer details.

    7. Generate the Data-Driven Prompt
       - Formulate a concise, well-structured prompt that will enable the "generate_python_code" tool to correctly compute the result or generate the appropriate plot.
       - Include selected dataset, relevant columns, type of metric, and any grouping logic if applicable.
       - Make sure that when you call dataset and columns they are in quotes, so the "generate_python_code" tool will correctly understand.

    8. Validate and Test
       - Ensure that the chosen "dataset_name" is valid.
       - Ensure that the chosen columns exist in the dataset, if not reiterate the process.
       - Verify referenced column names against actual dataframe columns.
       - Make sure that when you call dataset and columns they are in quotes, so the "generate_python_code" tool will correctly understand.
       - Confirm all requested operations are feasible with the selected data.
       - Handle edge cases (e.g., empty results, lack of numeric fields) with graceful error messages.
    """

    # Step 1 – Embed the query
    query_embedding = embedding_model.get_query_embedding(query)

    # Step 2 – Select dataset by semantic similarity
    dataset_scores = {
        name: float(embedding_model.similarity(query_embedding, embedding_model.get_text_embedding(info["description"])))
        for name, info in dataset_info.items()
    }
    best_dataset = max(dataset_scores, key=dataset_scores.get)
    df = dataset_info[best_dataset]["dataframe"]

    # Step 3 – Select best numeric column (fallback if none)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return {
            "error": f"No numeric columns found in the selected dataset '{best_dataset}'. Please refine your query."
        }

    similarities = {
        col: float(embedding_model.similarity(query_embedding, embedding_model.get_text_embedding(col)))
        for col in numeric_cols
    }
    best_numeric = max(similarities, key=similarities.get)

    # Step 4 – Optionally select group-by column
    best_groupby = None
    if " by " in query.lower() or "group" in query.lower():
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        if categorical_cols:
            cat_similarities = {
                col: float(embedding_model.similarity(query_embedding, embedding_model.get_text_embedding(col)))
                for col in categorical_cols
            }
            best_groupby = max(cat_similarities, key=cat_similarities.get)

    # Step 5 – Return structured info
    return {
        "dataset_name": best_dataset,
        "column": best_numeric,
        "group_by": best_groupby,
        "columns": list(df.columns),
        "description": (
            f"Matched to '{best_dataset}' with similarity score {round(dataset_scores[best_dataset], 4)}.\n"
            f"Selected numeric column: '{best_numeric}'.\n"
            f"{'Group by: ' + best_groupby if best_groupby else 'No group-by column identified.'}"
        )
    }


@function_tool
def generate_python_code(prompt: str) -> str:
    """
    Generate Python code from a natural-language prompt, following a clear developer workflow:

    1. Explicitly State the Goal
       - Understand whether the user wants data loading, transformation, analysis, modeling, or other tasks.

    2. Deeply Understand the Prompt
       - Parse the user’s intent: key operations, inputs, outputs, libraries required.
       - If any requirement is unclear or under‑specified, ask the user for clarification.

    3. Handle Ambiguity

       - For vague requests, request details (e.g. variable names, file paths, function signatures).
       - If the column called by the processo_data tool doesn't exist replace it with the most similar column

    4. Context: Available Datasets
       - Four CSVs are available under these paths:
        • "Accesso_English.csv"
        • "Pendolarismo_English.csv"
        • "Reddito_English.csv"
        • "Stipendi_English.csv"
       - If the prompt refers to data from any of these files, verify in code that the referenced columns exist in the chosen dataset before using them.

    5. Plan Code Structure
       - Identify necessary imports (e.g. pandas, numpy, matplotlib, scikit-learn).
       - Outline function definitions or script sections: data loading, preprocessing, computation, output.

    6. Handle the marge:
        Given two CSV files and a natural‑language question about the relationship between a “feature” in the first file and a “target” in the second, automatically:

        1. *Load* both files into pandas DataFrames, e.g. ⁠ df1 ⁠, ⁠ df2 ⁠.
        2. *Inspect* their schemas:
          a. Print ⁠ df1.columns ⁠ and ⁠ df2.columns ⁠.
          b. Identify possible join keys by looking for identical or semantically matching column names (e.g. “administration” vs. “organization”).
          c. If multiple candidates exist, choose the one with the greatest overlap of non‑null values. Rename both sides to a common key name ⁠ KEY ⁠.
        3. *Determine*:
          - In ⁠ df1 ⁠, which column(s) correspond to the “feature” (categorical or numeric) mentioned in the question.
          - In ⁠ df2 ⁠, which column is the “target” numeric variable (e.g. ⁠ average_commuting_distance ⁠).
        4. *Aggregate*:
          - If the feature is categorical, group ⁠ df1 ⁠ by ⁠ KEY ⁠ and that feature, sum or count as appropriate, pivot to one‑row‑per‑⁠ KEY ⁠ with proportions for each category.
          - If it’s numeric, compute the mean (or other summary) per ⁠ KEY ⁠.
        5. *Aggregate* the target in ⁠ df2 ⁠ to one row per ⁠ KEY ⁠ (e.g. weighted average if there’s a count column).
        6. If necessary given the request, *Merge* the two aggregated tables on ⁠ KEY ⁠, dropping rows where either side is missing.
            - for example, if the merge is between df_pendolarismo and df_accesso, merge on administration.
        7. *Compute* the statistic your question asks for (e.g. Pearson correlation between each feature‑column and the target; or a difference‑of‑means for two groups).
        8. *Plot* scatterplots or bar charts to illustrate each relationship.
        9. *Return*:
          - The Python code implementing steps 1–8.
          - A brief text summary stating the main numeric results (e.g. “Correlation between X and Y is 0.12, indicating no strong linear relationship.”).

        Use only the two provided CSV filepaths; do not load any other data.

    7. Implement Code
       - Write idiomatic, well-commented Python.
       - Use pandas for data handling; validate column existence via ⁠ if col in df.columns: ⁠ checks when reading these datasets.
       - Organize into functions or a runnable script as appropriate.

    8. Validate and Test
       - Ensure the generated code runs without errors.
       - Confirm that any dataset and column references are guarded by existence checks.
       - Include brief test snippets or assertions to verify correctness where applicable.

    9. Deliver the Code
       - Return the full Python source as a single string, ready to execute.
    """

import io
import contextlib

@function_tool
def execute_code_text(code: str) -> dict:
    """
    Executes Python code and returns:
      - ⁠ output ⁠: everything printed during execution
    """
    buf_out = io.StringIO()
    try:
        ns = {
            "__builtins__": __builtins__,
            'pd': pd,
            'plt': plt,
            'os': os,
            'sns': sns,
            'display': lambda obj: print(obj),
            'df_accesso': df_accesso,
            'df_pendolarismo': df_pendolarismo,
            'df_reddito': df_reddito,
            'df_stipendi': df_stipendi,
        }

        with contextlib.redirect_stdout(buf_out):
            exec(code, ns)

        return {
            'output': buf_out.getvalue().strip(),
        }

    except Exception as e:
        import traceback
        return {
            'output': f"❌ Error: {e}\n{traceback.format_exc()}",
        }



# Memory buffer setup
memory_buffer = []
max_memory_length = 6

def update_memory_buffer(memory_buffer, user_prompt: str, agent_response: str):
    memory_buffer.append({
        "user": user_prompt,
        "agent": agent_response
    })
    if len(memory_buffer) > max_memory_length:
        memory_buffer.pop(0)

async def triage_with_memory(user_query: str, agent, memory_buffer: list, max_memory_length=6):
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



Reporter_agent = Agent(
    name="Reporter",
    instructions=(
        "You are the Reporter (visualization/code-gen agent). You will receive a prompt *only* from Data Processor. "
        "Treat that prompt as authoritative. "
        "1. First, use the ⁠ generate_python_code ⁠ tool exactly once to produce the final Python code or report.  \n"
        "2. Then, immediately pass that generated code to the ⁠ execute_code_text ⁠ tool to run it and capture its output.  \n"
        "3. Return a single response containing both:  \n"
        "   • The execution results (stdout or error) from ⁠ execute_code_text ⁠.  \n"
        "   • The key Insight of the obtained result.  \n"
        "Do *not* re‑interpret the user’s original query or call any other tools."
    ),
    model="gpt-4.1",
    tools=[generate_python_code, execute_code_text]
)
data_processor = Agent(
    name="Data Processor",
    instructions=(
        "You are the Data Processor. Always start by calling the ⁠ processo_data ⁠ tool with the user’s query. "
        "Wait for its structured JSON response. "
        "Then, transform that JSON into a clear, concise natural-language prompt for the Reporter agent "
        "(mention dataset name, chosen column, group-by if any, and what analysis to perform). "
        "Hand off that new prompt to Reporter. Do *not* answer the user directly."
    ),
    model="gpt-4.1",
    handoffs=[Reporter_agent],
    tools=[processo_data]
)


triage_agent = Agent(
    name="ConversationAgent",
    instructions=(
        "You are the orchestrator. For every incoming user query, do *not* answer directly. "
        "Instead, immediately call the ⁠ processo_data ⁠ tool (via the Data Processor agent) with the raw user query. "
        "Do *not* attempt any processing yourself, and do *not* call any other tool or return any text. "
        "Simply package the user’s exact query as input to Data Processor."
    ),
    model="gpt-4.1",
    handoffs=[data_processor],
    tools=[]
)

#draw_graph(triage_agent)
