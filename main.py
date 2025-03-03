import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from smolagents import LiteLLMModel, Tool, CodeAgent
import serpapi
import requests
import json
from cors_config import add_cors

# Initialize FastAPI app
app = FastAPI()
add_cors(app)
# Load API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")


# Validate API keys
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
if not SERPAPI_KEY:
    raise ValueError("SERPAPI_KEY environment variable is not set")


# Define SerpApiSearchTool
class SerpApiSearchTool(Tool):
    name = "serpapi_search"
    description = "Performs a search using SerpApi (e.g., Google Jobs) with the specified query and parameters."
    inputs = {
        "query": {"type": "string", "description": "The search query (e.g., 'software developer')."},
        "location": {"type": "string", "description": "The geographic location (e.g., 'India').", "default": None, "nullable": True}
    }
    output_type = "string"

    def __init__(self, api_key, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

    def forward(self, query: str, location: str = None):
        params = {
            "api_key": self.api_key,
            "engine": "google_jobs",
            "q": query,
            "google_domain": "google.co.in",
            "hl": "en",
            "gl": "in"
        }
        if location:
            params["location"] = location
        try:
            search = serpapi.search(params)
            results = search
            if 'error' in results:
                return f"API error: {results['error']}"
            elif 'jobs_results' in results:
                jobs = results['jobs_results']
                job_list = []
                for job in jobs:
                    job_link = job.get('apply_options', [{'link': 'N/A'}])[0]['link']
                    job_str = f"{job.get('title', 'N/A')} at {job.get('company_name', 'N/A')} in {job.get('location', 'N/A')}  apply_link: {job_link}"
                    job_list.append(job_str)
                return "\n".join(job_list)
            else:
                return "No job results found."
        except Exception as e:
            return f"Error performing search: {str(e)}"



# Initialize model, tools, and agent
model = LiteLLMModel("gemini/gemini-1.5-flash", temperature=0.2, max_tokens=8010)
search_tool = SerpApiSearchTool(api_key=SERPAPI_KEY)

agent = CodeAgent(
    tools=[search_tool],
    model=model,
    additional_authorized_imports=['json','stat', 're', 'collections', 'random', 'datetime', 'itertools', 'math', 'time', 'queue', 'unicodedata', 'statistics']
)

# Define the request body model using Pydantic
class JobSearchRequest(BaseModel):
    role: str
    location: str


# Define the API endpoint for POST requests
@app.post("/jobs")
def get_jobs(request: JobSearchRequest):
    # Validate required fields
    if not request.role or not request.location:
        raise HTTPException(status_code=400, detail="Role and location are required fields")

    # Construct the prompt dynamically based on the request body
    example_json = [{
        "name": "Job Name",
        "position": "Job Position",
        "link": "Job link",
    }]
    prompt = (
        f"find a good job for me having role '{request.role}' and in {request.location} "
        f"with the following example json {json.dumps(example_json, indent=4)}"
    )



    try:
        # Run the agent with the constructed prompt
        response = agent.run(prompt)

        # Try parsing the response as JSON (assuming agent.run() may return a string)
        try:
            print(response)
            parsed_response = response
            return parsed_response
        except json.JSONDecodeError:
            # If parsing fails, assume it's already a list of dicts or return as-is
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# To run the app: use `uvicorn filename:app --reload` in the terminal
