import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools.bigquery import BigQueryCredentialsConfig, BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from google.cloud import bigquery
from google.genai import types
import google.auth
from google.auth.transport.requests import Request
import google.cloud.logging
load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env")
cloud_logging_client = google.cloud.logging.Client(
    project=os.getenv("GOOGLE_CLOUD_PROJECT")
)
cloud_logging_client.setup_logging()
from .callback_logging import log_model_response, log_query_to_model
RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=2, attempts=10)
# Uses externally-managed Application Default Credentials (ADC) by default.
# This decouples authentication from the agent / tool lifecycle.
# https://cloud.google.com/docs/authentication/provide-credentials-adc
application_default_credentials, _ = google.auth.default()
if not application_default_credentials.valid:
    application_default_credentials.refresh(Request())
credentials_config = BigQueryCredentialsConfig(
    credentials=application_default_credentials
)
# Define a tool configuration to block any write operations
tool_config = BigQueryToolConfig(write_mode=WriteMode.ALLOWED)
# Instantiate a BigQuery toolset
bigquery_toolset = BigQueryToolset(
    credentials_config=credentials_config, bigquery_tool_config=tool_config
)
def _serialize_datetime_in_dict(data):
    """Recursively converts datetime objects in a dictionary or list to ISO format strings."""
    if isinstance(data, dict):
        return {k: _serialize_datetime_in_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_serialize_datetime_in_dict(elem) for elem in data]
    elif isinstance(data, (datetime.date, datetime.datetime)):
        return data.isoformat()
    return data
def read_table_all(table_name: str):
    """Reads all rows from the specified table.
    Args:
        table_name: The name of the table to read from.
    Returns:
        list of dict: A list of all row dictionaries in the table.
    """
    client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    query = f"""
        SELECT * FROM `pool_data.{table_name}`
    """
    query_job = client.query(query)
    results = query_job.result()
    # Always returns a list (will be [] if the table is empty), which is safe to iterate
    return [_serialize_datetime_in_dict(dict(row)) for row in results]
def read_table(table_name: str, email: str):
    """Reads a single row matching the customer's email from the specified table.
    Args:
        table_name: The name of the table to read from.
        email: The customer's email address to filter by.
    Returns:
        dict: The row data if found, otherwise None.
    """
    client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    query = f"""
        SELECT * FROM `pool_data.{table_name}`
        WHERE customer_email = @email
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("email", "STRING", email)]
    )
    query_job = client.query(query, job_config=job_config)
    results = query_job.result()
    # Safely get the first row, or None if no rows exist
    row = next(results, None)
    if row:
        return _serialize_datetime_in_dict(dict(row))
    return None
def write_to_table(table_name: str, row: dict):
    """Writes a row to the specified table, automatically formatting date fields.
    Args:
        table_name: The name of the table to write to.
        row: A dictionary representing the row data to insert.
    Returns:
        str: "Success" on success, empty string if row is empty.
    """
    if not row:
        return ""
    client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    # We use a standard DML parameterized INSERT statement to avoid BigQuery's
    # streaming buffer limitation. This ensures that any rows written can be
    # deleted or modified immediately afterwards without causing BadRequest errors.
    columns = ", ".join(row.keys())
    param_placeholders = ", ".join(f"@{k}" for k in row.keys())
    query = f"""
        INSERT INTO `pool_data.{table_name}` ({columns})
        VALUES ({param_placeholders})
    """
    query_parameters = []
    for k, v in row.items():
        if v is None:
            param_type = "STRING"
        elif isinstance(v, bool):
            param_type = "BOOL"
        elif isinstance(v, int):
            param_type = "INT64"
        elif isinstance(v, float):
            param_type = "FLOAT64"
        elif isinstance(v, (datetime.date, datetime.datetime)):
            param_type = "DATE" if isinstance(v, datetime.date) else "DATETIME"
            v = v.isoformat()
        else:
            param_type = "STRING"
        query_parameters.append(bigquery.ScalarQueryParameter(k, param_type, v))
    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    query_job = client.query(query, job_config=job_config)
    query_job.result()
    return "Success"
def delete_from_table(table_name: str, email: str):
    """Deletes rows from the specified table that match the customer's email.
    Args:
        table_name: The name of the table to delete from.
        email: The customer's email to match for deletion.
    Returns:
        str: "Success" on success.
    """
    client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    query = f"""
        DELETE FROM `pool_data.{table_name}`
        WHERE customer_email = @email
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("email", "STRING", email)]
    )
    delete_job = client.query(query, job_config=job_config)
    delete_job.result()
    return "Success"
# MODIFICATION 1:
# Move the customer record using the existing helper functions.
def perform_consistent_transaction(
    from_table: str, to_table: str, customer_email: str
) -> bool:
    """
    Find a customer record, insert it into the destination table,
    and delete it from the source table.
    Return True when the operations succeed, or False when no
    matching source record exists or an operation reports failure.
    """
    # Step 1: Find the customer's record in the source table.
    row = read_table(from_table, customer_email)
    # Do not modify either table if no record exists.
    if row is None:
        return False
    # Step 2: Insert the record into the destination table.
    write_status = write_to_table(to_table, row)
    if write_status != "Success":
        return False
    # Step 3: Delete the record from the source table.
    delete_status = delete_from_table(from_table, customer_email)
    if delete_status != "Success":
        return False
    # Step 4: Return True if all operations succeed.
    return True
# MODIFICATION 2:
# Validate only the permitted table transitions.
def check_transaction(from_table: str, to_table: str) -> bool:
    """
    Return True only for permitted state transitions.
    Return False for invalid transitions, including terminal tables.
    """
    valid_transitions = {
        "pool_estimates": ["accepted_with_deposit", "denied_estimates"],
        "accepted_with_deposit": ["scheduled_installations"],
        "scheduled_installations": ["completed_pools"],
        "completed_pools": ["paid_and_closed"],
    }
    # .get() safely handles terminal tables such as:
    # paid_and_closed and denied_estimates.
    return to_table in valid_transitions.get(from_table, [])
# Agent Definition
root_agent = Agent(
    model=Gemini(model=os.getenv("MODEL"), retry_options=RETRY_OPTIONS),
    name="bigquery_agent",
    description=(
        "Agent to answer questions about BigQuery data and models and execute"
        " SQL queries."
    ),
    # MODIFICATION 3: Replaced the agent instructions.
    instruction="""
	    You are a data science agent with access to several BigQuery tools.
	    Make use of those tools to answer the user's questions.
	
	    Do not create new tables.
	    Before deleting a record to move it, confirm it exists and can be moved.
	    Before adding a record, confirm it is not already present.
	
	    The tables you have available are:
	        - pool_estimates: Contains all pool estimates
	        - accepted_with_deposit: Contains all pool estimates that have been accepted and have a deposit
	        - denied_estimates: Estimates that have been denied by the customer and will not proceed.
	        - scheduled_installations: Contains all pool installations that have been scheduled
	        - completed_pools: Contains all pool installations that have been completed
	        - paid_and_closed: Contains all pool installations that have been paid and closed
	
	    Use read_table_all to read the data from the tables.
	    Use check_transaction to check if a transaction is valid before performing any transactions. If not valid, tell the user so.
	    Use perform_consistent_transaction when you need to read a table, insert a row into another table and delete the original row.
	    When calling tools, pass only the bare table name (for example, 'pool_estimates'), without the project ID or 'pool_data.' dataset prefix.
	    """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    
    # MODIFICATION 4:
    # Removed bigquery toolset from the agent and exposed only
    # the verified Python functions.
    tools=[
        read_table,
        read_table_all,
        check_transaction,
        perform_consistent_transaction,
    ],
)
app = App(
    root_agent=root_agent,
    name="bigquery_agent",
)
