# %%

import re
import os
import ast
import pandas as pd
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from datasets import load_dataset
from databench_eval import Evaluator
from databench_eval.utils import load_table, load_sample
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)

import pickle

# %%

# os.environ["AIML_API_KEY"] = "xxx"

aiml_key = os.getenv("AIML_API_KEY")

# API Configuration
API_KEY = aiml_key
BASE_URL = "https://api.aimlapi.com/v1"

api = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# %%

semeval_dev_qa   = load_dataset("cardiffnlp/databench", name="semeval", split="dev")
semeval_train_qa = load_dataset("cardiffnlp/databench", name="qa", split="train")
semeval_comp_qa = pd.read_csv("competition/test_qa.csv")

df_QA = pd.DataFrame(semeval_train_qa)
df_QA.head()

df_COMP = pd.DataFrame(semeval_comp_qa)
df_COMP.head()

# %%

# Lists to track processed columns
excluded_columns = []
replaced_columns = []

def call_openai_api(model, system_prompt, user_prompt):
    """Function to make API calls to OpenAI."""
    completion = api.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=256
    )
    response = completion.choices[0].message.content
    return response

def clean_headers(headers):
    """
    Clean and transform headers containing specific patterns.
    """
    cleaned_headers = []
    for header in headers:
        try:
            # Detect and transform headers with {key:value} patterns
            if re.search(r"\{[^:]+:[^}]+\}", header):
                # Replace {key:value} with key_value
                new_header = re.sub(
                    r"\{([^:]+):([^}]+)\}",
                    lambda m: f"{m.group(1).strip()}_{m.group(2).strip()}",
                    header
                )
                replaced_columns.append(header)  # Track replaced headers
                cleaned_headers.append(new_header)
            elif re.search(r"<gx:[^>]+>", header):
                # Replace {key:value} with key_value
                new_header = re.sub(r"<gx:[^>]+>", "", header).strip()
                
                replaced_columns.append(header)  # Track replaced headers
                cleaned_headers.append(new_header)
            else:
                cleaned_headers.append(header)
                
        except Exception as e:
            print(f"Error cleaning header '{header}': {e}")
            cleaned_headers.append(header)
    return cleaned_headers

def estimate_token_count(text, token_to_char_ratio=4):
    """Estimate the number of tokens from the string length."""
    return len(text) // token_to_char_ratio

# %%

def generate_information_prompt(query, max_tokens=7850):
    """Generates the prompt for OpenAI API."""

    global df  # Access the global `df`

    # Step 1: Clean headers
    df.columns = clean_headers(df.columns)

    df_cleaned = df.copy()

    # Step 2: Limit the length of cell values to 50 characters
    df_cleaned = df_cleaned.map(lambda x: x[:50] if isinstance(x, str) and len(x) > 50 else x)

    # Step 3: Dynamically adjust row count
    rows_to_include = 2  # Start with 2 rows
    columns_info = df.dtypes.to_dict()
    columns_info_str = str(columns_info)
    df_str = ""
    total_estimated_tokens = 0

    while rows_to_include > 0:
        df_str = df_cleaned.head(rows_to_include).to_string()
        estimated_tokens = estimate_token_count(df_str)
        total_estimated_tokens = estimated_tokens + estimate_token_count(columns_info_str)

        if total_estimated_tokens <= max_tokens:
            break  # Fits within the limit, exit the loop
        rows_to_include -= 1  # Reduce the number of rows

    # Step 5: Remove columns_info only if necessary
    if total_estimated_tokens > max_tokens:
        columns_info_str = ""  # Exclude columns_info

    information_retrieval_prompt_str = (
        f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert in Python and Pandas (version 2.2.2). Analyze a DataFrame and a query to infer key metadata required to process the query.

        Task:
        Based on the information provided:
        1. Columns Used: Identify the relevant column(s) for answering the query.
        2. Column Types: Provide the data types of the relevant columns.
        3. Response Type: Choose one of: `boolean`, `number`, `category`, `list[category]`, or `list[number]`. No other formats are allowed.
        4. Sample Answer: Generate a plausible sample answer based on the query and DataFrame preview, aligned with the Response Type.

        Requirements:
        - Pay close attention to what the query is asking for, it can be tricky.
        - Only include columns explicitly needed for the query.
        - Base the sample answer on plausible values from the provided DataFrame preview.
        - Ensure the response is concise and well-structured.
        - Do not provide any extra explanation or context beyond the requested metadata.

        Output Format:
        Columns Used: (columns_used)
        Column Types: (column_types)
        Response Type: (response_type)
        Sample Answer: (sample_answer)
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        DataFrame Preview:
        `{df_str}`

        Columns Information:
        `{columns_info_str}`

        Query:`{query}`
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|> Response: """
    )
    return information_retrieval_prompt_str

def generate_pandas_expression_prompt(query, generated_information, max_tokens=7850):
    """Generates the prompt for OpenAI API."""
    
    global df  # Access the global `df`

    # Step 1: Clean headers
    df.columns = clean_headers(df.columns)

    df_cleaned = df.copy()

    # Step 2: Limit the length of cell values to 50 characters
    df_cleaned = df_cleaned.map(lambda x: x[:50] if isinstance(x, str) and len(x) > 50 else x)

    # Step 3: Dynamically adjust row count
    rows_to_include = 2  # Start with 2 rows
    while True:
        df_str = df_cleaned.head(rows_to_include).to_string()
        estimated_tokens = estimate_token_count(df_str)
        if estimated_tokens <= max_tokens:
            break
        rows_to_include -= 1

    # Instruction for Pandas code generation
    instruction_str = (
        "1. You are tasked to convert the query into **a SINGLE expression** using Pandas (version 2.2.2).\n"
        "2. The result of the expression MUST be one of the following types: `boolean`, `number`, `category`, `list[category]`, or `list[number]`.\n"
        "3. You MUST NOT return a DataFrame or any type not listed above.\n"
        "4. **STRICTLY FORBIDDEN**: Writing multi-line code, defining variables, or using statements like `import`, `print`, or assignments (e.g., `x = ...`).\n"
        "5. The Python expression MUST have only ONE line of code that can be executed directly using the `eval()` function.\n"
        "6. **DO NOT USE NEWLINES** in the expression. Only return the single expression directly.\n"
        "7. **DO NOT QUOTE THE EXPRESSION**. The output must ONLY be the raw code of the single expression.\n"
    )

    pandas_prompt_str = (
        f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are working with a pandas DataFrame in Python (version 2.2.2), named `df`.
        Result of `print(df.head(2))`:
        `{df_str}`
        The following information was inferred from the DataFrame and query, you MUST use it to generate the expression:
        `{generated_information}`
        Instructions:
        {instruction_str}

        *** Pay CLOSE attention to what the query is asking for, it can be tricky. ***
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Query: `{query}`
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>Expression: """
    )
    return pandas_prompt_str

def generate_verify_response_prompt(sample_question, final_response, max_tokens=7850) -> None:
    
    global df  # Access the global `df`

    # Step 1: Clean headers
    df.columns = clean_headers(df.columns)

    df_cleaned = df.copy()

    # Step 2: Limit the length of cell values to 50 characters
    df_cleaned = df_cleaned.map(lambda x: x[:50] if isinstance(x, str) and len(x) > 50 else x)

    # Step 3: Dynamically adjust row count
    rows_to_include = 2  # Start with 2 rows
    while True:
        df_str = df_cleaned.head(rows_to_include).to_string()
        estimated_tokens = estimate_token_count(df_str)
        if estimated_tokens <= max_tokens:
            break
        rows_to_include -= 1

    # Instruction for Pandas code generation
    instruction_str = (
        "1. The result MUST be one of the following types: `boolean`, `number`, `category`, `list[category]`, or `list[number]`. No other types are allowed.\n"
        "2. You MUST NOT return a DataFrame, dictionary, or any other unsupported type.\n"
        "3. The Python expression MUST consist of ONLY ONE line of code that can be directly executed using the `eval()` function.\n"
        "4. **STRICTLY FORBIDDEN**: Writing multi-line code, defining variables, or using statements like `import`, `print`, or assignments (e.g., `x = ...`).\n"
        "5. The expression MUST solve the query in a concise, single line, with no additional context or helper functions.\n"
        "6. **DO NOT INCLUDE NEWLINES** in the expression. The output must be a SINGLE LINE ONLY.\n"
        "7. **PRINT ONLY THE RAW EXPRESSION**: Do not include explanations, comments, or quote the expression. The output must be directly evaluable.\n"
        "8. Failure to adhere to these rules will result in execution failure.\n"
    )

    # Prompt for Pandas code generation
    verify_response_str = (
        f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are working with a Pandas DataFrame in Python (version 2.2.2) named `df`. Your task is to analyze the response to a question a see if it makes sense.
        Result of `print(df.head(2))`:
        `{df_str}`
        Does the response make sense for the query? 
        - If yes, simply reply "yes".
        - If not, follow the instructions and generate a new expresion.
        Instructions: (Only use if the response does not make sense)
        {instruction_str}

        *** Pay CLOSE attention to what the query is asking for, it can be tricky. ***
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Query: `{sample_question}`
        Response: `{final_response}`
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>Response: """
    )

    return verify_response_str

def generate_error_fix_prompt(sample_question, error_encountered, generated_information, previous_expression, max_tokens=7850) -> None:
    
    global df  # Access the global `df`

    # Step 1: Clean headers
    df.columns = clean_headers(df.columns)

    df_cleaned = df.copy()

    # Step 2: Limit the length of cell values to 50 characters
    df_cleaned = df_cleaned.map(lambda x: x[:50] if isinstance(x, str) and len(x) > 50 else x)

    # Step 3: Dynamically adjust row count
    rows_to_include = 2  # Start with 2 rows
    while True:
        df_str = df_cleaned.head(rows_to_include).to_string()
        estimated_tokens = estimate_token_count(df_str)
        if estimated_tokens <= max_tokens:
            break
        rows_to_include -= 1

    # Instruction for Pandas code generation
    instruction_str = (
        "1. The new expression MUST resolve the query and fix the error encountered previously.\n"
        "2. The result of the expression MUST be one of the following types: `boolean`, `number`, `category`, `list[category]`, or `list[number]`. No other types are allowed.\n"
        "3. You MUST NOT return a DataFrame, dictionary, or any type not explicitly listed above.\n"
        "4. The Python expression MUST consist of ONLY ONE line of code and MUST be directly executable using the `eval()` function.\n"
        "5. **STRICTLY FORBIDDEN**: Writing multi-line code, defining variables, or using statements like `import`, `print`, or assignments (e.g., `x = ...`).\n"
        "6. **DO NOT INCLUDE NEWLINES** in the expression. Only provide a SINGLE LINE of code.\n"
        "7. **PRINT ONLY THE RAW EXPRESSION**: Do not include explanations, comments, or quote the expression. The output must be directly evaluable.\n"
        "8. Ensure the new expression fixes the error while strictly adhering to these rules. Failure to comply will result in failure of execution.\n"
    )

    # Prompt for Pandas code generation
    error_fix_prompt_str = (
        f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are working with a Pandas DataFrame in Python (version 2.2.2) named `df`. Your task is to fix a failed Pandas expression by generating a new one that avoids the same error.
        Below is a preview of the DataFrame (result of `print(df.head())`):
        `{df_str}`

        Task:
        1. Analyze the provided DataFrame, query, expected expression reponse information, previous expression, and encountered error.
        2. Generate a new expression that solves the query and prevents the error.

        Previous Attempt:
        Expression: `{previous_expression}`
        Error: `{error_encountered}`

        Expected Expression Reponse Information:
        {generated_information}

        Instructions:
        {instruction_str}
        
        *** Pay CLOSE attention to what the query is asking for, it can be tricky. ***
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Query: `{sample_question}`
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>Expression: """
    )

    return error_fix_prompt_str


# %%

def generate_information(sample_question, text_model='meta-llama/Llama-3-8b-chat-hf'):
    """Processes a query and returns the Pandas code generated by the API."""
    system_prompt = generate_information_prompt(sample_question)

    information_response = call_openai_api(
        model=text_model,
        system_prompt=system_prompt,
        user_prompt=sample_question
    )

    return information_response.strip()

def process_query(sample_question, generated_information, code_model='Qwen/Qwen2.5-Coder-32B-Instruct'):

    system_prompt = generate_pandas_expression_prompt(sample_question, generated_information)

    expression_response = call_openai_api(
        model=code_model,
        system_prompt=system_prompt,
        user_prompt=sample_question
    )

    return expression_response.strip()

def process_error(sample_question, error_encountered, generated_information, previous_expression, code_model='Qwen/Qwen2.5-Coder-32B-Instruct'):

    system_prompt = generate_error_fix_prompt(sample_question, error_encountered, generated_information, previous_expression)

    error_response = call_openai_api(
        model=code_model,
        system_prompt=system_prompt,
        user_prompt=sample_question
    )

    return error_response.strip()

def pandas_pipeline(generated_expression):
    import traceback

    pandas_output_parser = PandasInstructionParser(df=df)

    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_output_parser": pandas_output_parser,
        },
        verbose=False,
    )

    # Chain configuration
    qp.add_chain(["input", "pandas_output_parser"])

    # Add link from input to pandas_output_parser
    qp.add_link("input", "pandas_output_parser")

    try:
        # Run the pipeline
        response, intermediates = qp.run_with_intermediates(query_str=generated_expression)

        # Debug outputs
        print("Pipeline intermediates:", intermediates)
        print("Pipeline response:", response)

        return response, intermediates
    except ValueError as ve:
        print(f"Pipeline ValueError: {ve}")
        traceback.print_exc()
    except Exception as e:
        print(f"Pipeline Exception: {e}")
        traceback.print_exc()
    
    return None, None

# %%

# Main processing loop
if __name__ == "__main__":
    use_parquet = True  # Set to False to use load methods (for df_QA)
    time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    file_type = "parquet" if use_parquet else "qa"
    file_path = Path() / 'results' / f"predictions_{file_type}_{time}.txt"

    # Ensure the file is cleared at the beginning of the run
    with open(file_path, "w") as file:
        file.write("")

    df_COMP = pd.read_csv("competition/test_qa.csv")

    results_data = {
        "response": [],
        "question": [],
        "input": [],
        "dataset": [],
        "information_generation": [],
        "pandas_expression": [],
        "pandas_output_parser": []
    }

    df_name = ""

    for ix, row in (df_COMP if use_parquet else df_QA).iterrows():
        try:
            #if ix not in error_debug:
            #     continue
            if ix % 20 == 0:
                total_rows = len(df_COMP) if use_parquet else len(df_QA)
                print(f"Processing {ix} out of {total_rows}")

            if df_name != row["dataset"]:
                df_name = row["dataset"]
                if use_parquet:
                    # Load Parquet file
                    df = pd.read_parquet(f"competition/{df_name}/sample.parquet")
                else:
                    # Load QA datasets using appropriate methods
                    df = load_table(df_name)  # Replace with actual method as needed

            question = row["question"]
            
            generated_information_response = generate_information(question)

            pandas_expression_response = process_query(question, generated_information_response)
        
            final_reponse, intermediates = pandas_pipeline(pandas_expression_response)

            final_reponse = final_reponse.replace('\r\n', '').replace('\n', '').strip()
            
            results_data["response"].append(final_reponse)
            results_data["dataset"].append(df_name)
            results_data["question"].append(question)
            results_data["information_generation"].append(generated_information_response)
            results_data["pandas_expression"].append(pandas_expression_response)

            # Store intermediate results
            for k, v in intermediates.items():
                results_data[k].append(v)

            # Handle errors in the response
            if 'There was an error running the output as Python code' in final_reponse:
                print("\n\nError Pipeline called:\n")

                # print(f"Generated Information (llm1): {intermediates['llm1'].outputs['output'].message.content}")
                
                error_fix_expression_response = process_error(question, final_reponse, generated_information_response, pandas_expression_response)

                # Run the second pipeline to fix errors
                response_eh, intermediates_eh = pandas_pipeline(error_fix_expression_response)
                
                # Print all intermediary data after running the error-fix pipeline
                print("\nIntermediary Values After Error Fix Pipeline:")
                for k, v in intermediates_eh.items():
                    # Convert intermediary values to string if necessary
                    value_str = str(v) if not isinstance(v, str) else v
                    print(f"{k}: {value_str}")
                
                response_eh = response_eh.replace('\r\n', '').replace('\n', '').strip()
                results_data['response'][-1] = response_eh  # Update the last response
                
                final_reponse = response_eh  # Update the response for writing to file

            with open(file_path, "a") as file:
                file.write(f"{final_reponse}\n")

        except Exception as e:
            print(f"Error processing row {ix}: {e}")
            with open(file_path, "a") as file:
                file.write(f"Error: {e}\n")

# %%

f = 'results/predictions_qa_append_27_2256.txt'
# f = file_path

acc = Evaluator().eval(f)
#acc_lite = Evaluator().eval(..., save="predictions_lite.txt")

print("Predictions accuracy: " + str(acc))
#print("Predictions_lite accuracy: " + acc_lite)

# %%

file_path = Path() / f"results_data_{file_type}_{time}.pkl"

with open(file_path, 'wb') as f:
    pickle.dump(results_data, f)
    
# with open(file_path, "rb") as file:
#     loaded_dict = pickle.load(file)

# %%

df_results = pd.DataFrame(results_data)

# %%

df_results['input'] =\
    df_results['input'].apply(lambda x: x.outputs['query_str'])
df_results['pandas_output_parser'] = \
    df_results['pandas_output_parser'].apply(lambda x: x.outputs['output'])

# %%

# Filter rows containing errors and create a copy to avoid warnings
errors_df = df_results[df_results['pandas_output_parser'].str.startswith('There was an error', na=False)].copy()

# Extract the error type (assuming it's the remaining part of the string after the prefix)
errors_df['error_type'] = errors_df['pandas_output_parser'].str.replace('There was an error:', '').str.strip()

# Count the occurrences of each error type
error_counts = errors_df['error_type']\
    .value_counts()\
    .reset_index()\
    .rename(columns={'index': 'Error Type', 'error_type': 'Count'})

# Save the DataFrame to a CSV file
df_suf = 'comp' if use_parquet else 'qa'
csv_path = f'error_counts_{df_suf}_lite_open.csv'
error_counts.to_csv(csv_path, index=False, sep=';', encoding='utf-8-sig')

# %% 

df_results.to_csv(f'detailed_results_{df_suf}_lite__open.csv', index=False, sep=';', encoding='utf-8-sig')
len(df_results)
# %%
