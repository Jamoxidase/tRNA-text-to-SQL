import gradio as gr
import regex as re
import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

# Import the query_api function from the tRNA script
# Assuming the script is saved as trna_retriever.py
from trna_retriever import query_api, AVAILABLE_MODELS

# Configure logging to stdout only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console only
    ]
)
logger = logging.getLogger(__name__)

# Define the messages database file path
MESSAGES_DB_PATH = "messages.db"

# Function to initialize the messages database
def initialize_messages_db():
    """Create or ensure the messages database exists with the correct schema."""
    try:
        conn = sqlite3.connect(MESSAGES_DB_PATH)
        cursor = conn.cursor()
        
        # Create the messages table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query_text TEXT NOT NULL,
            models TEXT NOT NULL,
            include_ontology INTEGER NOT NULL,
            trial_name TEXT NOT NULL,
            model_input TEXT,
            model_output TEXT,
            sql_query TEXT,
            success INTEGER,
            row_count INTEGER,
            execution_time REAL
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Messages database initialized at {MESSAGES_DB_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error initializing messages database: {str(e)}")
        return False

# Initialize the messages database when the module is loaded
initialize_messages_db()

# Create a ThreadPoolExecutor for running async functions
executor = ThreadPoolExecutor()

def run_async(coro):
    """Helper function to run async functions in Gradio."""
    try:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"Error in run_async: {str(e)}", exc_info=True)
        raise

def log_model_interaction(
    query_text: str,
    models: List[str],
    include_ontology: bool,
    trial_name: str,
    model_input: Optional[str] = None,
    model_output: Optional[str] = None,
    sql_query: Optional[str] = None,
    success: Optional[bool] = None,
    row_count: Optional[int] = None,
    execution_time: Optional[float] = None
) -> bool:
    """
    Log model interaction data to the SQLite database.
    
    Args:
        query_text: The user's query
        models: List of model names used
        include_ontology: Whether ontology was included
        trial_name: Name of the trial
        model_input: The full prompt sent to the model
        model_output: The model's response
        sql_query: Generated SQL query (if any)
        success: Whether the query was successful
        row_count: Number of rows returned (if query executed)
        execution_time: Time taken to execute the query
        
    Returns:
        True if logging was successful, False otherwise
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(MESSAGES_DB_PATH)
        cursor = conn.cursor()
        
        # Current timestamp
        timestamp = datetime.now().isoformat()
        
        # Convert models list to JSON string
        models_json = json.dumps(models)
        
        # Convert boolean to integer for SQLite
        include_ontology_int = 1 if include_ontology else 0
        success_int = 1 if success else 0 if success is not None else None
        
        # Insert the data
        cursor.execute(
            '''
            INSERT INTO messages (
                timestamp, query_text, models, include_ontology, 
                trial_name, model_input, model_output, sql_query,
                success, row_count, execution_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                timestamp, query_text, models_json, include_ontology_int,
                trial_name, model_input, model_output, sql_query,
                success_int, row_count, execution_time
            )
        )
        
        conn.commit()
        conn.close()
        logger.info(f"Logged interaction for query: '{query_text[:50]}...' with models: {models}")
        return True
        
    except Exception as e:
        logger.error(f"Error logging model interaction: {str(e)}")
        return False

class TRNARetrieverUI:
    """Class to handle the tRNA Ontology Retriever UI logic."""
    
    def __init__(self):
        """Initialize the UI handler."""
        self.last_results = None
    
    def process_query(
        self,
        query_text: str,
        models: List[str],
        include_ontology: bool,
        compare_with_without_ontology: bool,
        execute_query: bool,
        top_k: int,
        embedding_file: str,
        ontology_file: str,
        db_path: str,
        llm_provider: str = "ollama",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "cas/ministral-8b-instruct-2410_q4km"
    ) -> Tuple[str, Dict, list, str]:
        """
        Process a query and return formatted results.
        
        Returns:
            Tuple of:
            - Summary output (markdown)
            - Raw results (JSON)
            - List of tab data for results (each item contains name and content)
            - Model results HTML
        """
        
        # Validate inputs
        if not query_text.strip():
            return "Please enter a query.", None, [], ""
        
        if not models:
            return "Please select at least one model.", None, [], ""
        
        # Configure trials based on UI options
        if compare_with_without_ontology:
            trials = [
                {"name": "with_ontology", "include_ontology": True},
                {"name": "without_ontology", "include_ontology": False}
            ]
        else:
            trials = [{"name": "default", "include_ontology": include_ontology}]
        
        # Handle file paths
        if not embedding_file:
            embedding_file = "trna_index_v01.json"  # Default
        
        if not ontology_file:
            ontology_file = "trnadb_ontology_v01.json"  # Default
        
        # Set environment variables for LLM provider
        if llm_provider:
            os.environ["LLM_PROVIDER"] = llm_provider
        if ollama_url:
            os.environ["OLLAMA_BASE_URL"] = ollama_url

        # If using ollama:custom, replace it with the actual model name
        if "ollama:custom" in models and ollama_model:
            # Find and replace "ollama:custom" with the actual ollama model name
            for i, model in enumerate(models):
                if model == "ollama:custom":
                    models[i] = f"ollama:{ollama_model}"
                    logging.info(f"Using Ollama model: {ollama_model}")

        # Run the query through the API
        try:
            results = run_async(query_api(
                query_text=query_text,
                models=models,
                trials=trials,
                execute_query=execute_query,
                k=top_k,
                embedding_file=embedding_file,
                ontology_file=ontology_file,
                db_path=db_path if db_path else None
            ))
            
            # Store results for later use
            self.last_results = results
            
            # Log the interaction to the messages database
            # For single model case
            if 'model_response' in results:
                model_name = results['model_response'].get('model', 'Unknown Model')
                trial_name = "default"
                
                # Extract model input/output
                model_input = None
                if 'formatted_subgraph' in results:
                    model_input = f"Query: {query_text}\n\n{results.get('formatted_subgraph', '')}"
                
                model_output = results['model_response'].get('full_response', '')
                
                # Extract SQL and results info
                sql_query = None
                success = None
                row_count = None
                execution_time = None
                
                if 'model_response' in results and results['model_response'].get('sql_blocks'):
                    sql_query = results['model_response']['sql_blocks'][0]
                
                if 'query_results' in results:
                    query_results = results['query_results']
                    success = query_results.get('success', False)
                    row_count = query_results.get('row_count', 0)
                    execution_time = query_results.get('execution_time', 0)
                
                # Log the interaction
                log_model_interaction(
                    query_text=query_text,
                    models=[model_name],
                    include_ontology=include_ontology,
                    trial_name=trial_name,
                    model_input=model_input,
                    model_output=model_output,
                    sql_query=sql_query,
                    success=success,
                    row_count=row_count,
                    execution_time=execution_time
                )
                
            # For multiple model case
            elif 'model_results' in results:
                for model_name, trials in results['model_results'].items():
                    for trial_name, response in trials.items():
                        # Extract model input/output
                        model_input = None
                        if 'formatted_subgraph' in results:
                            # Only include subgraph if trial includes it
                            include_ontology_for_trial = 'with_ontology' in trial_name or \
                                                     (trial_name == 'default' and include_ontology)
                            if include_ontology_for_trial:
                                model_input = f"Query: {query_text}\n\n{results.get('formatted_subgraph', '')}"
                            else:
                                model_input = f"Query: {query_text}"
                        
                        model_output = response.get('full_response', '')
                        
                        # Extract SQL and results info
                        sql_query = None
                        success = None
                        row_count = None
                        execution_time = None
                        
                        if response.get('sql_blocks'):
                            sql_query = response['sql_blocks'][0]
                            
                        if response.get('query_results'):
                            query_results = response['query_results']
                            success = query_results.get('success', False)
                            row_count = query_results.get('row_count', 0)
                            execution_time = query_results.get('execution_time', 0)
                        
                        # Log the interaction
                        include_ontology_for_log = 'with_ontology' in trial_name or \
                                               (trial_name == 'default' and include_ontology)
                        
                        log_model_interaction(
                            query_text=query_text,
                            models=[model_name],
                            include_ontology=include_ontology_for_log,
                            trial_name=trial_name,
                            model_input=model_input,
                            model_output=model_output,
                            sql_query=sql_query,
                            success=success,
                            row_count=row_count,
                            execution_time=execution_time
                        )
            
            # Process the results for different display components
            summary_output = self.generate_summary_output(results)
            
            # Create a list of tab contents for query results
            results_tabs = self.generate_results_tabs(results)
            
            # Generate model results HTML
            model_results_html = self.generate_model_results_html(results)
            
            return summary_output, results, results_tabs, model_results_html
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            
            # Log the error
            log_model_interaction(
                query_text=query_text,
                models=models if models else ["unknown"],
                include_ontology=include_ontology,
                trial_name="error",
                model_input=f"Error occurred before model input could be generated",
                model_output=f"Error: {str(e)}",
                success=False
            )
            
            return error_msg, {"error": str(e)}, [], ""
            
    def generate_results_tabs(self, results: Dict) -> list:
        """
        Generate a list of tabs for query results.
        Each tab contains the model name, trial name, and HTML content.
        
        Returns:
            List of tuples (tab_name, html_content) for Gradio Tabs
        """
        # Collect all available model/trial combinations with valid results
        tab_contents = []
        
        # Single model case
        if 'model_response' in results:
            model_name = results['model_response'].get('model', 'Unknown Model')
            trial_name = "default"
            
            # Get query results directly from the results object, if any
            query_results = results.get('query_results') or results['model_response'].get('query_results', {})
            
            # Get our visual indicators ready
            has_error = False
            success_indicator = "✅"
            
            if query_results:
                # Check if query was successful and returned results
                if query_results.get('success') and query_results.get('row_count', 0) > 0:
                    # We have successful results with rows - let's display them
                    results_data = query_results.get('results', [])
                    column_names = query_results.get('column_names', [])
                    
                    if results_data and column_names:
                        tab_name = f"{success_indicator} {model_name} ({trial_name})"
                        content = self._generate_table_html(results_data, column_names, len(results_data))
                        tab_contents.append((tab_name, content))
                        return tab_contents
                else:
                    # Query failed or returned no rows
                    has_error = True
                    error_msg = query_results.get('error', 'No results available')
                    sql = query_results.get('sql', '')
            else:
                # No query results at all
                has_error = True
                error_msg = "No query results available"
                sql = ""
            
            # If we got here, there was an error or no results
            tab_name = f"❌ {model_name} ({trial_name})"
            content = self._generate_error_html(error_msg, sql)
            tab_contents.append((tab_name, content))
        
        # Multiple model case
        elif 'model_results' in results:
            for model_name, trials in results['model_results'].items():
                for trial_name, response in trials.items():
                    # Generate the tab name with status indicator
                    has_error = False
                    if response.get('query_results'):
                        if not response['query_results'].get('success'):
                            has_error = True
                    
                    # Tab name with status
                    status_icon = "❌" if has_error else "✅"
                    tab_name = f"{status_icon} {model_name} ({trial_name})"
                    
                    # Generate content based on success/error
                    if not has_error and response.get('query_results') and response['query_results'].get('row_count', 0) > 0:
                        # Success case
                        data, headers = self.extract_table_data(results, model_name, trial_name)
                        if data and headers:
                            content = self._generate_table_html(data, headers, len(data))
                        else:
                            content = "<div><p>No data found. This should not happen.</p></div>"
                    else:
                        # Error case
                        error_msg = "No results available"
                        sql = ""
                        if response.get('query_results'):
                            error_msg = response['query_results'].get('error', error_msg)
                            sql = response['query_results'].get('sql', '')
                        
                        content = self._generate_error_html(error_msg, sql)
                    
                    tab_contents.append((tab_name, content))
        
        # If no results at all, show an error
        if not tab_contents:
            tab_contents.append(("No Results", "<div>No query results available from any model.</div>"))
            
        return tab_contents
        
    def _generate_table_html(self, data: List[Dict], headers: List[str], row_count: int) -> str:
        """Generate HTML table from data and headers."""
        html = f"""
        <div style="margin-bottom: 10px; font-style: italic; color: #ffffff;">Total rows: {row_count}</div>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; border: 1px solid #34495e;">
                <thead>
                    <tr style="background-color: #2c3e50; color: white;">
        """
        
        # Add headers
        for header in headers:
            html += f'<th style="padding: 8px; text-align: left; border: 1px solid #34495e;">{header}</th>'
        
        html += """
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add rows
        for i, row in enumerate(data):
            # Alternate row colors using different dark shades
            bg_color = "#1f2c38" if i % 2 == 1 else "#2c3e50"
            html += f'<tr style="background-color: {bg_color};">'
            
            for header in headers:
                # Handle potential missing values
                value = row.get(header, "")
                # Sanitize the value for HTML display
                if isinstance(value, str):
                    # Escape HTML characters
                    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html += f'<td style="padding: 8px; border: 1px solid #34495e; color: #ffffff;">{value}</td>'
            
            html += "</tr>"
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def _generate_error_html(self, error_msg: str, sql: str = "") -> str:
        """Generate HTML for error messages."""
        html = """<div style="padding: 15px; background-color: #2c3e50; border: 1px solid #34495e; border-radius: 4px; margin-bottom: 15px;">
            <p style="color: #ffffff; margin: 0 0 10px 0;"><strong>Query execution failed.</strong></p>
        """
        
        if sql:
            html += f"""<p style="color: #ffffff; margin: 10px 0;">The executed SQL query was:</p>
            <pre style="background-color: #1f2c38; padding: 10px; border: 1px solid #34495e; border-radius: 4px; margin: 10px 0; color: #ffffff;">{sql}</pre>
            """
        
        html += f"""<p style="color: #ff6b6b; margin: 10px 0;"><strong>Error:</strong> {error_msg}</p>"""
        html += "</div>"
        
        return html
    
    def generate_results_table(self, results: Dict) -> str:
        """Generate HTML table from query results with tabs for multiple models/trials."""
        # First, collect all available model/trial combinations with valid results
        model_trial_data = []
        
        # Single model case
        if 'model_response' in results:
            model_name = results['model_response'].get('model', 'Unknown Model')
            trial_name = "default"
            
            # Try to get query_results from either top-level or inside model_response
            query_results = results.get('query_results') or results['model_response'].get('query_results', {})
            
            # Check if query_results exists and has the data we need
            if query_results and query_results.get('success') and query_results.get('row_count', 0) > 0:
                # Success case - get the data
                if 'results' in query_results and 'column_names' in query_results:
                    data = query_results['results']
                    headers = query_results['column_names']
                    model_trial_data.append({
                        'model_name': model_name,
                        'trial_name': trial_name,
                        'data': data,
                        'headers': headers,
                        'row_count': len(data),
                        'success': True
                    })
                else:
                    # Missing data structure
                    model_trial_data.append({
                        'model_name': model_name,
                        'trial_name': trial_name,
                        'success': False,
                        'error': 'Results structure incomplete',
                        'sql': query_results.get('sql', '')
                    })
            else:
                # Query failed or no results
                error_msg = 'No results available'
                sql = ''
                if query_results:
                    error_msg = query_results.get('error', error_msg)
                    sql = query_results.get('sql', '')
                
                model_trial_data.append({
                    'model_name': model_name,
                    'trial_name': trial_name,
                    'success': False,
                    'error': error_msg,
                    'sql': sql
                })
        
        # Multiple model case
        elif 'model_results' in results:
            for model_name, trials in results['model_results'].items():
                for trial_name, response in trials.items():
                    if response.get('query_results'):
                        query_results = response['query_results']
                        
                        if query_results.get('success') and query_results.get('row_count', 0) > 0:
                            # Success case
                            data, headers = self.extract_table_data(results, model_name, trial_name)
                            model_trial_data.append({
                                'model_name': model_name,
                                'trial_name': trial_name,
                                'data': data,
                                'headers': headers,
                                'row_count': len(data),
                                'success': True
                            })
                        else:
                            # Error case
                            model_trial_data.append({
                                'model_name': model_name,
                                'trial_name': trial_name,
                                'success': False,
                                'error': query_results.get('error', 'No results available'),
                                'sql': query_results.get('sql', '')
                            })
        
        # If no results at all, show an error
        if not model_trial_data:
            return "<div style='padding: 15px; background-color: #2c3e50; border: 1px solid #34495e; border-radius: 4px; margin-bottom: 15px;'><p style='color: #ffffff; margin: 0;'><strong>No query results available from any model.</strong></p></div>"
            
        # Generate the tabbed interface
        html = """
        <style>
            .tabs {
                display: flex;
                flex-wrap: wrap;
                margin-bottom: 10px;
            }
            .tab {
                padding: 8px 16px;
                cursor: pointer;
                background-color: #242427;
                color: #ffffff;
                border: 1px solid #34495e;
                border-radius: 4px 4px 0 0;
                margin-right: 5px;
                margin-bottom: -1px;
            }
            .tab.active {
                background-color: #34495e;
                border-bottom: 1px solid #34495e;
            }
            .tab-content {
                display: none;
                padding: 15px;
                border: 1px solid #34495e;
                border-radius: 0 4px 4px 4px;
                background-color: #242427;
            }
            .tab-content.active {
                display: block;
            }
        </style>
        
        <div class="tabs" id="query-result-tabs">
        """
        
        # Generate tabs
        for i, item in enumerate(model_trial_data):
            active_class = " active" if i == 0 else ""
            status_indicator = "✅" if item.get('success') else "❌"
            tab_label = f"{item['model_name']} ({item['trial_name']})"
            html += f"""<div class="tab{active_class}" onclick="activateTab('tab-{i}')">{status_indicator} {tab_label}</div>"""
        
        html += "</div>"  # End tabs
        
        # Generate tab content
        for i, item in enumerate(model_trial_data):
            active_class = " active" if i == 0 else ""
            html += f"""<div id="tab-{i}" class="tab-content{active_class}">"""
            
            if item.get('success'):
                # Success case - show table
                html += f"""<div style="margin-bottom: 10px; font-style: italic; color: #ffffff;">Total rows: {item['row_count']}</div>"""
                html += """<div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; border: 1px solid #34495e;">
                    <thead>
                        <tr style="background-color: #2c3e50; color: white;">
                """
                
                # Add headers
                for header in item['headers']:
                    html += f'<th style="padding: 8px; text-align: left; border: 1px solid #34495e;">{header}</th>'
                
                html += """
                        </tr>
                    </thead>
                    <tbody>
                """
                
                # Add rows
                for j, row in enumerate(item['data']):
                    # Alternate row colors
                    bg_color = "#1f2c38" if j % 2 == 1 else "#2c3e50"
                    html += f'<tr style="background-color: {bg_color};">'
                    
                    for header in item['headers']:
                        # Handle missing values
                        value = row.get(header, "")
                        # Sanitize for HTML
                        if isinstance(value, str):
                            value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        html += f'<td style="padding: 8px; border: 1px solid #34495e; color: #ffffff;">{value}</td>'
                    
                    html += "</tr>"
                
                html += """
                    </tbody>
                    </table>
                </div>
                """
            else:
                # Error case - show error message
                html += """<div style="padding: 15px; background-color: #2c3e50; border: 1px solid #34495e; border-radius: 4px; margin-bottom: 15px;">
                    <p style="color: #ffffff; margin: 0 0 10px 0;"><strong>Query execution failed.</strong></p>
                """
                
                if item.get('sql'):
                    html += f"""<p style="color: #ffffff; margin: 10px 0;">The executed SQL query was:</p>
                    <pre style="background-color: #1f2c38; padding: 10px; border: 1px solid #34495e; border-radius: 4px; margin: 10px 0; color: #ffffff;">{item['sql']}</pre>
                    """
                
                if item.get('error'):
                    html += f"""<p style="color: #ff6b6b; margin: 10px 0;"><strong>Error:</strong> {item['error']}</p>"""
                
                html += "</div>"
            
            html += "</div>"  # End tab content
        
        # Add JavaScript for tab functionality
        html += """
        <script>
            function activateTab(tabId) {
                // Hide all tabs
                var tabContents = document.getElementsByClassName('tab-content');
                for (var i = 0; i < tabContents.length; i++) {
                    tabContents[i].classList.remove('active');
                }
                
                // Deactivate all tab buttons
                var tabs = document.getElementsByClassName('tab');
                for (var i = 0; i < tabs.length; i++) {
                    tabs[i].classList.remove('active');
                }
                
                // Activate the selected tab
                document.getElementById(tabId).classList.add('active');
                
                // Activate the corresponding tab button
                var index = tabId.split('-')[1];
                var tabs = document.getElementsByClassName('tab');
                tabs[index].classList.add('active');
            }
        </script>
        """
        
        return html
    
    def extract_table_data(self, results: Dict, model_name: str = None, trial_name: str = None) -> Tuple[List[Dict], List[str]]:
        """
        Extract table data and headers from results.
        
        Args:
            results: The full results dictionary
            model_name: Specific model to extract data for (optional)
            trial_name: Specific trial to extract data for (optional)
            
        Returns:
            Tuple of (data, headers) or (None, None) if no valid data found
        """
        if not results:
            logger.warning("No results object provided to extract_table_data")
            return None, None
        
        # Log the structure of the results for debugging
        logger.info(f"Extract table data from results with keys: {list(results.keys())}")
        
        # Check if we have query results to display
        query_results = None
        
        # Single model case
        if model_name is None and trial_name is None:
            if 'model_response' in results and results['model_response'].get('query_results'):
                query_results = results['model_response']['query_results']
                logger.info(f"Found query results in model_response: {query_results.get('success', False)}")
                # Debug more information about the query_results
                if query_results:
                    logger.info(f"Query results success: {query_results.get('success')}")
                    logger.info(f"Query results row_count: {query_results.get('row_count', 0)}")
                    logger.info(f"Query results error: {query_results.get('error', 'None')}")
                    if 'results' in query_results:
                        logger.info(f"Query results contains data: {len(query_results['results'])} rows")
                    else:
                        logger.info("Query results does not contain 'results' key")
            
            # Multiple model case - use first model's results if no specific model requested
            elif 'model_results' in results:
                logger.info("Looking for query results in model_results")
                for m_name, trials in results['model_results'].items():
                    for t_name, response in trials.items():
                        if response.get('query_results') and response['query_results'].get('success'):
                            query_results = response['query_results']
                            logger.info(f"Found successful query results in {m_name} ({t_name})")
                            break
                    if query_results:
                        break
        # Extract specific model/trial results
        elif model_name and 'model_results' in results:
            if model_name in results['model_results']:
                trials = results['model_results'][model_name]
                
                # If trial name is specified, get that specific trial
                if trial_name and trial_name in trials:
                    response = trials[trial_name]
                    if response.get('query_results'):
                        query_results = response['query_results']
                        logger.info(f"Found specific results for {model_name} ({trial_name})")
                # Otherwise use the first trial
                else:
                    for t_name, response in trials.items():
                        if response.get('query_results'):
                            query_results = response['query_results']
                            logger.info(f"Found first trial results for {model_name} ({t_name})")
                            break
        
        if not query_results:
            logger.warning(f"No query_results found in response for model={model_name}, trial={trial_name}")
            return None, None
            
        if not query_results.get('success'):
            logger.warning(f"Query not successful: {query_results.get('error')}")
            return None, None
            
        if query_results.get('row_count', 0) == 0:
            logger.warning("Query returned 0 rows")
            return None, None
        
        # Extract data and headers
        if 'results' not in query_results or 'column_names' not in query_results:
            logger.warning("Query results doesn't contain results or column_names")
            return None, None
            
        data = query_results['results']
        headers = query_results['column_names']
        
        logger.info(f"Successfully extracted data: {len(data)} rows, {len(headers)} columns")
        return data, headers
    
    def generate_summary_output(self, results: Dict) -> str:
        """Generate summary output from results."""
        if not results:
            return "No results returned."
        
        # Basic info
        output = f"## Query Results\n\n"
        output += f"**Query:** {results['query']}\n\n"
        output += f"**Time Statistics:**\n"
        output += f"- Preparation Time: {results['latency']['preparation']:.2f}s\n"
        output += f"- Total Time: {results['latency']['total']:.2f}s\n\n"
        
        # Field retrieval info
        output += "### Semantic Field Retrieval\n\n"
        output += "The system identified these database fields as most relevant to your query:\n\n"
        
        for field in results['field_retrieval']['top_fields']:
            field_id = field['field_id']
            similarity = field['similarity']
            
            # Find field details
            field_detail = None
            for detail in results['field_retrieval']['field_details']:
                if detail.get('field_id') == field_id:
                    field_detail = detail
                    break
            
            output += f"- **{field_id}** (similarity: {similarity:.4f})\n"
            if field_detail and 'description' in field_detail:
                output += f"  - *{field_detail['description']}*\n"
        
        output += "\n### SQL Hint\n"
        output += "The system generated this SQL template to help query the data:\n\n"
        output += f"```sql\n{results['sql_hint']}\n```\n\n"
        
        # Add the full subgraph to the summary
        if 'formatted_subgraph' in results:
            output += "### Full Ontology Subgraph\n\n"
            output += "This is the complete ontology subgraph sent to the model:\n\n"
            output += f"```\n{results['formatted_subgraph']}\n```\n\n"
        
        return output
    
    def generate_model_results_html(self, results: Dict) -> str:
        """Generate HTML for model results with appropriate layout."""
        if not results or not ('model_results' in results or 'model_response' in results):
            return ""
        
        html = """
        <style>
            /* Main container for model results */
            .model-results-container {
                margin-top: 20px;
                background-color: #2c3e50;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #34495e;
                color: #ffffff;
            }
            
            /* Model grid layout */
            .model-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            /* Model card */
            .model-card {
                border: 1px solid #34495e;
                border-radius: 5px;
                overflow: hidden;
                background-color: #34495e;
                flex: 1 1 350px;
                max-width: 100%;
            }
            
            /* Model header with status */
            .model-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background-color: #2c3e50;
                color: white;
                padding: 10px 15px;
                font-weight: bold;
            }
            
            .status-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: normal;
                margin-left: 10px;
            }
            
            .status-success {
                background-color: #27ae60;
            }
            
            .status-error {
                background-color: #e74c3c;
            }
            
            /* Trial label */
            .trial-label {
                background-color: #1f2c38;
                color: white;
                padding: 6px 15px;
                font-size: 0.9em;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            
            /* Card content */
            .card-content {
                padding: 15px;
                background-color: #34495e;
            }
            
            /* Metrics grid */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
                gap: 10px;
                margin-bottom: 15px;
            }
            
            /* Metric box */
            .metric {
                background-color: #242427;
                padding: 8px;
                border-radius: 3px;
                text-align: center;
                border: 1px solid #34495e;
            }
            
            /* SQL block */
            .sql-block {
                background-color: #1f2c38;
                padding: 10px;
                border: 1px solid #34495e;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
                overflow-x: auto;
                max-height: 150px;
                overflow-y: auto;
                margin-bottom: 15px;
                color: #ffffff;
            }
            
            /* Error message */
            .error-message {
                color: #ff6b6b; 
                margin: 10px 0;
                padding: 10px;
                background-color: rgba(231, 76, 60, 0.1);
                border-radius: 4px;
                border-left: 3px solid #e74c3c;
            }
        </style>
        <div class="model-results-container">
            <h3 style="margin-top: 0; margin-bottom: 15px; color: #ffffff;">Model Results</h3>
            <div class="model-grid">
        """
        
        # Single model case
        if 'model_response' in results:
            response = results['model_response']
            model_name = response.get('model', 'Unknown Model')
            
            # Combine response with query results if they exist at the top level
            combined_response = dict(response)
            if 'query_results' in results and 'query_results' not in combined_response:
                combined_response['query_results'] = results['query_results']
            
            # Determine success/error status
            has_error = False
            error_msg = ""
            if combined_response.get('query_results'):
                if not combined_response['query_results'].get('success'):
                    has_error = True
                    error_msg = combined_response['query_results'].get('error', 'Query execution failed')
            
            # Create model card with status
            html += f"""
            <div class="model-card">
                <div class="model-header">
                    <span>{model_name}</span>
                    <span class="status-badge {('status-error' if has_error else 'status-success')}">
                        {('Error' if has_error else 'Success')}
                    </span>
                </div>
                <div class="card-content">
            """
            
            # Metrics
            html += self._generate_metrics_html(combined_response)
            
            # Error message (if any)
            if has_error:
                html += f"""<div class="error-message"><strong>Error:</strong> {error_msg}</div>"""
            
            # SQL
            if combined_response.get('sql_blocks'):
                html += """<div style="font-weight: bold; margin-bottom: 8px; color: #ffffff;">Generated SQL</div>"""
                sql = combined_response['sql_blocks'][0].replace('<', '&lt;').replace('>', '&gt;')
                html += f"""<div class="sql-block">{sql}</div>"""
            
            # Comments - use our improved 'comments' field if available, otherwise extract from full_response
            if combined_response.get('comments'):
                html += """<div style="font-weight: bold; margin-bottom: 8px; color: #ffffff;">Model Comments</div>"""
                # Handle either string or list of comments
                comment_text = combined_response['comments'][0] if isinstance(combined_response['comments'], list) else combined_response['comments']
                html += f"<p style='color: #ecf0f1;'>{comment_text}</p>"
            elif combined_response.get('full_response'):
                html += """<div style="font-weight: bold; margin-bottom: 8px; color: #ffffff;">Model Comments</div>"""
                full_response = combined_response.get('full_response')
                non_sql_text = re.sub(r'```sql.*?```', '', full_response, flags=re.DOTALL)
                # Clean up and add to HTML
                non_sql_text = non_sql_text.strip()
                if non_sql_text:
                    html += f"<p style='color: #ecf0f1;'>{non_sql_text}</p>"
            html += """
                </div>
            </div>
            """
        
        # Multi-model case
        elif 'model_results' in results:
            # Process each model and its trials
            for model_name, trials in results['model_results'].items():
                for trial_name, response in trials.items():
                    # Determine success/error status
                    has_error = False
                    error_msg = ""
                    if response.get('query_results'):
                        if not response['query_results'].get('success'):
                            has_error = True
                            error_msg = response['query_results'].get('error', 'Query execution failed')
                    
                    # Create model card with status
                    html += f"""
                    <div class="model-card">
                        <div class="model-header">
                            <span>{model_name}</span>
                            <span class="status-badge {('status-error' if has_error else 'status-success')}">
                                {('Error' if has_error else 'Success')}
                            </span>
                        </div>
                        <div class="trial-label">Trial: {trial_name}</div>
                        <div class="card-content">
                    """
                    
                    # Metrics
                    html += self._generate_metrics_html(response)
                    
                    # Error message (if any)
                    if has_error:
                        html += f"""<div class="error-message"><strong>Error:</strong> {error_msg}</div>"""
                    
                    # SQL
                    if response.get('sql_blocks'):
                        html += """<div style="font-weight: bold; margin-bottom: 8px; color: #ffffff;">Generated SQL</div>"""
                        sql = response['sql_blocks'][0].replace('<', '&lt;').replace('>', '&gt;')
                        html += f"""<div class="sql-block">{sql}</div>"""
                    
                    # Comments (brief summary)
                    if response.get('comments'):
                        html += """<div style="font-weight: bold; margin-bottom: 8px; color: #ffffff;">Model Comments</div>"""
                        # Handle either string or list of comments
                        comment_text = response['comments'][0] if isinstance(response['comments'], list) else response['comments']
                        html += f"<p style='color: #ecf0f1;'>{comment_text}</p>"
                    
                    html += """
                        </div>
                    </div>
                    """
            
        html += """
            </div>
        </div>
        """  # End model grid and container
        
        return html
    
    def _generate_metrics_html(self, response: Dict) -> str:
        """Generate HTML for metrics grid."""
        html = """<div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 10px;
            margin-bottom: 15px;">"""
        
        # Latency
        html += f"""
        <div style="
            background-color: #242427;
            padding: 8px;
            border-radius: 3px;
            text-align: center;
            border: 1px solid #34495e;">
            <div style="font-size: 12px; color: #ecf0f1; margin-bottom: 3px;">Latency</div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff;">{response['latency']:.2f}s</div>
        </div>
        """
        
        # Ontology
        html += f"""
        <div style="
            background-color: #242427;
            padding: 8px;
            border-radius: 3px;
            text-align: center;
            border: 1px solid #34495e;">
            <div style="font-size: 12px; color: #ecf0f1; margin-bottom: 3px;">Ontology</div>
            <div style="font-size: 16px; font-weight: bold; color: #ffffff;">{'Yes' if response.get('include_ontology', True) else 'No'}</div>
        </div>
        """
        
        # Find query results data - could be directly in response or in query_results field
        query_results = None
        if 'query_results' in response:
            query_results = response['query_results']
        
        # Query results - display even if error, but with different styling
        if query_results:
            # Row count with conditional styling
            row_color = "#27ae60" if query_results.get('success') else "#e74c3c"
            row_value = query_results.get('row_count', 0) if query_results.get('success') else "Error"
            
            html += f"""
            <div style="
                background-color: #242427;
                padding: 8px;
                border-radius: 3px;
                text-align: center;
                border: 1px solid #34495e;">
                <div style="font-size: 12px; color: #ecf0f1; margin-bottom: 3px;">Rows</div>
                <div style="font-size: 16px; font-weight: bold; color: {row_color};">{row_value}</div>
            </div>
            """
            
            # Query time
            html += f"""
            <div style="
                background-color: #242427;
                padding: 8px;
                border-radius: 3px;
                text-align: center;
                border: 1px solid #34495e;">
                <div style="font-size: 12px; color: #ecf0f1; margin-bottom: 3px;">Query Time</div>
                <div style="font-size: 16px; font-weight: bold; color: #ffffff;">{query_results.get('execution_time', 0):.2f}s</div>
            </div>
            """
        
        html += """</div>"""  # End metrics grid
        
        return html

# Define custom CSS for the UI
css = """
.header {
    margin-bottom: 20px;
}

.main-title {
    margin-bottom: 5px;
}

.input-section {
    margin-bottom: 20px;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #34495e;
    background-color: #242427;
    color: #ffffff;
}

.results-section {
    margin-top: 20px;
}

.summary-section {
    margin-bottom: 20px;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #34495e;
    background-color: #242427;

    color: #ffffff;
}

.table-section {
    margin-top: 20px;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #34495e;
    background-color: #242427;

    color: #ffffff;
}

/* Custom form layouts */
.form-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 10px;
}

/* Section headers */
.section-header {
    font-weight: bold;
    margin-bottom: 10px;
    color: #ffffff;
}

/* Target the Markdown components inside the sections */
.summary-section > div, .table-section > div {
    background-color: transparent !important; /* Make markdown background transparent */
}

/* Target any elements within the Markdown components */
.summary-section > div > div, .table-section > div > div {
    background-color: transparent !important;
}

.gradio-container .summary-section > div,
.gradio-container .table-section > div {
    background-color: transparent !important;
}
"""

# Create the UI
ui_handler = TRNARetrieverUI()

with gr.Blocks(css=css, title="tRNA Ontology Retriever") as app:
    # Header section
    with gr.Row(elem_classes=["header"]):
        with gr.Column():
            gr.Markdown("# tRNA Ontology Retriever", elem_classes=["main-title"])
            gr.Markdown("Query a tRNA database using natural language queries processed through LLMs. You may need to be very specific with some queries, as the system is currently free-form. \n Note: success status pertains to the query execution, not the accuracy of the results. Interactions are logged to a SQLite database for future reference, and an evaluation UI is coming soon. .")
    
    # Input section - horizontal layout
    with gr.Row(elem_classes=["input-section"]):
        with gr.Column():
            # First row - Query and model selection
            with gr.Row(elem_classes=["form-row"]):
                query_input = gr.Textbox(
                    label="Natural Language Query", 
                    placeholder="e.g., Find all tRNAs with methionine anticodons",
                    lines=2,
                    scale=3
                )
                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    label="Select Models",
                    value=["ollama:custom"],  # Using Ollama as default model
                    multiselect=True,
                    scale=1
                )
            
            # Second row - Options
            with gr.Row(elem_classes=["form-row"]):
                include_ontology = gr.Checkbox(
                    label="Include Ontology", 
                    value=True
                )
                compare_with_without_ontology = gr.Checkbox(
                    label="Compare With & Without Ontology", 
                    value=False
                )
                execute_query = gr.Checkbox(
                    label="Execute SQL", 
                    value=True
                )
                top_k = gr.Slider(
                    label="Top-K Fields", 
                    minimum=1, 
                    maximum=20, 
                    value=5, 
                    step=1
                )
            
            # Third row - Advanced options and submit button
            with gr.Row(elem_classes=["form-row"]):
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        embedding_file = gr.Textbox(
                            label="Embedding File",
                            value="trna_index_v01.json"
                        )
                        ontology_file = gr.Textbox(
                            label="Ontology File",
                            value="trnadb_ontology_v01.json"
                        )
                        db_path = gr.Textbox(
                            label="Database Path",
                            value="trna_db_v01.db",
                        )

                    with gr.Row():
                        llm_provider = gr.Radio(
                            label="LLM Provider",
                            choices=["litellm", "ollama"],
                            value="ollama",
                            info="Select which LLM provider to use"
                        )
                        ollama_url = gr.Textbox(
                            label="Ollama URL",
                            value="http://localhost:11434",
                            info="URL for Ollama server (if using Ollama provider)"
                        )

                    with gr.Row():
                        ollama_model = gr.Textbox(
                            label="Ollama Model Name",
                            value="cas/ministral-8b-instruct-2410_q4km",
                            info="Specify which Ollama model to use (must be installed via 'ollama pull <model>')"
                        )

                submit_btn = gr.Button(
                    "Process Query", 
                    variant="primary",
                    elem_classes=["process-btn"]
                )
    
    # Results section
    with gr.Column(elem_classes=["results-section"]) as results_container:
        # Model results - dynamic layout
        model_results_html = gr.HTML()

        # Query results table
        with gr.Accordion("Query Results", elem_classes=["table-section"], open=True):
            table_container = gr.HTML()
        
        # Summary section 
        with gr.Accordion("Query Summary", elem_classes=["summary-section"], open=True):
            results_output = gr.Markdown()

        
        # Raw JSON (hidden by default)
        with gr.Accordion("Raw JSON Data", open=False):
            json_output = gr.JSON()
    
    # Connect the button to the processing function
    submit_btn.click(
        fn=ui_handler.process_query,
        inputs=[
            query_input, model_dropdown, include_ontology, compare_with_without_ontology,
            execute_query, top_k, embedding_file, ontology_file, db_path,
            llm_provider, ollama_url, ollama_model
        ],
        outputs=[
            results_output, json_output, table_container, model_results_html
        ]
    )
    
# Utility function to check message log entries
def check_message_log(limit: int = 5):
    """Print the most recent entries in the message log."""
    try:
        conn = sqlite3.connect(MESSAGES_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            SELECT id, timestamp, query_text, models, trial_name, success, row_count 
            FROM messages 
            ORDER BY id DESC 
            LIMIT ?
            ''', 
            (limit,)
        )
        
        rows = cursor.fetchall()
        
        if not rows:
            print("No messages in the log yet.")
            return
            
        print(f"Recent message log entries (last {len(rows)}):")
        print("=" * 80)
        
        for row in rows:
            id, timestamp, query, models, trial, success, row_count = row
            models_list = json.loads(models)
            success_text = "✓" if success == 1 else "✗" if success == 0 else "?"
            row_text = str(row_count) if row_count is not None else "N/A"
            
            print(f"ID: {id} | Time: {timestamp} | Models: {', '.join(models_list)}")
            print(f"Query: {query[:50]}...")
            print(f"Trial: {trial} | Success: {success_text} | Rows: {row_text}")
            print("-" * 80)
            
        conn.close()
        
    except Exception as e:
        print(f"Error checking message log: {str(e)}")

# Add a CLI argument parser for various operations
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='tRNA Retriever UI')
    parser.add_argument('--check-logs', action='store_true', help='Check recent message logs')
    parser.add_argument('--log-limit', type=int, default=5, help='Number of log entries to show')
    parser.add_argument('--clear-logs', action='store_true', help='Clear all message logs')
    
    args = parser.parse_args()
    
    if args.clear_logs:
        try:
            conn = sqlite3.connect(MESSAGES_DB_PATH)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM messages')
            conn.commit()
            conn.close()
            print("Message logs cleared successfully.")
        except Exception as e:
            print(f"Error clearing logs: {str(e)}")
    
    if args.check_logs:
        check_message_log(args.log_limit)
    else:
        # Launch the Gradio app
        app.launch(server_name="0.0.0.0")#share=True)