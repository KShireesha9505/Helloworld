
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from git import Repo
from github import Github
import os
import shutil
from github.GithubException import GithubException
from git import Repo, GitCommandError
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import datetime

class CodeSolution(BaseModel):
    """Schema for code solutions."""
    description: str = Field(description="Description of the solution approach")
    code: str = Field(description="Complete code including imports and docstring")

class GraphState(TypedDict):
    """State for the task processing workflow."""
    status: str  # Using status instead of error to track state
    task_content: str
    repo_dir: str
    generation: CodeSolution | None
    iterations: int


def clone_repository(github_token: str, repo_name: str) -> tuple[str, str]:
    """Clone the repository and return the local directory."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.join(script_dir, 'agent-task')
        
        print(f"Target directory: {repo_dir}")

        # Remove existing directory
        if os.path.exists(repo_dir):
            print("Removing existing directory...")
            try:
                shutil.rmtree(repo_dir, onerror=lambda func, path, exc_info: os.chmod(path, 0o777))
            except Exception as e:
                return "", f"Failed to clean directory: {str(e)}"

        # Clone with authentication
        try:
            print(f"Cloning {repo_name}...")
            # Use authenticated URL
            auth_url = f"https://{github_token}@github.com/{repo_name}.git"
            # print(auth_url)
            Repo.clone_from(
                url=auth_url,
                to_path=repo_dir,
                progress=print
            )
            return repo_dir, ""
        except GitCommandError as e:
            # print(e+"10")
            return "", f"Git clone failed (code {e.status}): {str(e)}"
        except Exception as e:
            # print(e+"20")
            return "", f"Clone error: {str(e)}"

    except Exception as e:
        # print(e+"30")
        return "", f"Repository setup failed: {str(e)}"
    
def read_task_file(repo_dir: str) -> tuple[str, str]:
    """Read the task markdown file."""
    try:
        task_path = os.path.join(repo_dir, 'tasks', 'task.md.txt')
        print(f"Attempting to read task from: {task_path}")
        
        # Verify path exists
        if not os.path.exists(task_path):
            return "", f"Task file not found at: {task_path}"
            
        with open(task_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return "", "Task file is empty"
            return content, ""
    except Exception as e:
        return "", f"Failed to read task: {str(e)}"

def initialize_state(github_token: str, repo_name: str) -> dict:
    """Initialize the workflow state."""
    repo_dir, error = clone_repository(github_token, repo_name)
    if error:
        # print("t1")
        return {
            "status": "failed",
            "task_content": "",
            "repo_dir": "",
            "generation": None,
            "iterations": 0
        }

    task_content, error = read_task_file(repo_dir)
    # print(task_content)
    if error:
        # print("hello")
        return {
            "status": "failed",
            "task_content": "",
            "repo_dir": repo_dir,
            "generation": None,
            "iterations": 0
        }

    return {
        "status": "ready",
        "task_content": task_content,
        "repo_dir": repo_dir,
        "generation": None,
        "iterations": 0
    }

from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Configure Gemini (add to your initialization code)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_solution(state: GraphState):
    """Generate code solution using Gemini"""
    if state["status"] == "failed":
        return state

    task_content = state["task_content"]
    iterations = state["iterations"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python developer. Generate a solution based on the task requirements.
        Include complete code with imports, type hints, docstring, and examples.
        The function must be named exactly 'calculate_products'."""),
        ("human", "Task description:\n{task}"),
    ])

    try:
        # Using Gemini Pro
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        chain = prompt | llm.with_structured_output(CodeSolution)

        print(f"\nGenerating solution - Attempt #{iterations + 1}")
        solution = chain.invoke({"task": task_content})
        
        print(f"\nGenerated solution:\n{solution.code}")
        return {
            "status": "generated",
            "task_content": task_content,
            "repo_dir": state["repo_dir"],
            "generation": solution,
            "iterations": iterations + 1
        }
    except Exception as e:
        print(f"Generation failed: {str(e)}")
        return {**state, "status": "failed"}

def test_solution(state: GraphState):
    """Test the generated code solution."""
    if state["status"] != "generated" or not state["generation"]:
        return {**state, "status": "failed"}

    try:
        namespace = {}
        exec(state["generation"].code, namespace)

        result = namespace['calculate_products']([1, 2, 3, 4])
        if result != [24, 12, 8, 6]:
            return {**state, "status": "failed"}

        return {**state, "status": "tested"}

    except Exception:
        return {**state, "status": "failed"}

def create_pr(state: GraphState):
    """Create a pull request with the solution."""
    if state["status"] != "tested" or not state["generation"]:
        return {**state, "status": "failed"}

    try:
        solution = state["generation"]
        repo = Repo(state["repo_dir"])

        # Local git operations
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"solution/array-products-{timestamp}"
        current = repo.create_head(branch_name)
        current.checkout()

        solution_path = os.path.join(state["repo_dir"], "array_products.py")
        with open(solution_path, "w") as f:
            f.write(solution.code)

        repo.index.add(["array_products.py"])
        repo.index.commit("feat: add array products calculator")
        origin = repo.remote("origin")
        origin.push(branch_name)

        # Create PR using GitHub API
        g = Github(os.getenv("GITHUB_TOKEN"))
        repo_name = repo.remotes.origin.url.split('.git')[0].split('/')[-2:]
        repo_name = '/'.join(repo_name)
        gh_repo = g.get_repo(repo_name)

        pr = gh_repo.create_pull(
            title="Add Array Products Calculator",
            body=f"Implements array products calculator with the following approach:\n\n{solution.description}",
            base="main",
            head=branch_name
        )

        print(f"Created PR: {pr.html_url}")
        return {**state, "status": "completed", "pr_url": pr.html_url}

    except Exception as e:
        print(f"Failed to create PR: {str(e)}")
        return {**state, "status": "failed"}

def should_continue(state: GraphState) -> str:
    """Determine next step based on status."""
    if state["status"] == "failed":
        if state["iterations"] < 3:
            return "generate"
        return "end"
    return "continue"

def create_agent(github_token: str, repo_name: str):
    workflow = StateGraph(GraphState)
    # print(github_token)
    workflow.add_node("generate", generate_solution)
    workflow.add_node("test", test_solution)
    workflow.add_node("create_pr", create_pr)

    # Define core workflow
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "test")
    workflow.add_edge("create_pr", END)

    # Define conditional transitions from test node
    workflow.add_conditional_edges(
        "test", should_continue,
        {"generate": "generate", "continue": "create_pr", "end": END}
    )
    return workflow.compile()

def run_agent(github_token: str, repo_name: str):
    """Run the agent to generate and submit a solution."""
    try:
        agent = create_agent(github_token, repo_name)
        # print(agent)
        initial_state = initialize_state(github_token, repo_name)

        if initial_state["status"] == "failed":
            print("Failed to initialize agent")
            return {"status": "failed"}

        result = agent.invoke(initial_state)
        print(result)
        if result["status"] == "completed":
            print("Successfully created PR with solution!")
        else:
            print("Failed to create solution")

        return {
            "status": result["status"],
            "generation": result["generation"].code if result["generation"] else None,
            "pr_url": result.get("pr_url")
        }

    except Exception as e:
        print("Agent execution failed")
        return {"status": "failed"}

import os
from dotenv import load_dotenv
load_dotenv()


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# print(GITHUB_TOKEN)
if not GITHUB_TOKEN:
    raise ValueError("Please set GITHUB_TOKEN environment variable")

REPO_NAME = "KShireesha9505/agent-task"
result = run_agent(GITHUB_TOKEN, REPO_NAME)

if result["status"] == "completed":
    print("\nSolution generated successfully!")
    if result.get("pr_url"):
        print(f"\nPull Request created at: {result['pr_url']}")
    if result.get("generation"):
        print("\nGenerated Code:")
        print(result["generation"])
else:
    print("\nFailed to generate solution")


