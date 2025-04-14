# github_agent.py
import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

class GitHubIssueResolver:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
            "Accept": "application/vnd.github.v3+json" #"Hey, I want my response in the format you use for version 3 of your API, and I want it as JSON."
        }
        self.owner = "techwithtim"  # Default, can be overridden
        self.repo = "Flask-Web-App-Tutorial"

    def _make_github_request(self, endpoint: str) -> Dict:
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/{endpoint}"
        response = requests.get(url, headers=self.headers)
        # print("--------")
        # print(response.json())
        return response.json() if response.status_code == 200 else {}

    def find_similar_issues(self, issue_title: str) -> List[Dict]:
        """Find issues with similar titles using GitHub search API"""
        print("hello")
        query = f"repo:{self.owner}/{self.repo} {issue_title} in:title state:all"
        print(query)
        search_url = f"https://api.github.com/search/issues?q={query}"
        response = requests.get(search_url, headers=self.headers)
        print(response.json())
        return response.json().get("items", [])

    def get_issue_state(self, issue_number: int) -> Dict:
        """Get detailed issue state including timeline events"""
        issue_data = self._make_github_request(f"issues/{issue_number}")
        # print(issue_data)
        # print("----")
        timeline_data = self._make_github_request(f"issues/{issue_number}/timeline")
        # print(timeline_data)
        # print("-----------_____")
        return {
            "basic": issue_data,
            "timeline": timeline_data,
            "is_resolved": self._check_if_resolved(issue_data, timeline_data)
        }

    def _check_if_resolved(self, issue_data: Dict, timeline_data: List) -> bool:
        """Determine if issue was properly resolved"""
        if issue_data.get("state") != "closed":
            return False
        # if timeline_data.get("event") == "closed":
        #     return True
        
        # Check for PRs that mention closing this issue
        for event in timeline_data:
            if event.get("event") == "cross-referenced":
                source = event.get("source", {})
                # print("-----------_____")
                # print(source)
                # print("-----------_____")
                if "pull_request" in source.get("html_url", ""):
                    if any(keyword in source.get("body", "").lower() 
                          for keyword in ["closes", "fixes", "resolves"]):
                        return True
        return False

    def contribute_to_issue(self, issue_number: int, comment: str) -> str:
        """Add context to existing issue"""
        print(issue_number)
        print(f"https://api.github.com/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments")
        response = requests.post(
            f"https://api.github.com/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments",
            headers=self.headers,
            json={"body": comment}
        )
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)
        return "Comment added successfully" if response.status_code == 201 else "Failed to add comment"

    def handle_new_issue(self, title: str, body: str) -> Dict:
        """Full workflow for issue handling"""
        similar_issues = self.find_similar_issues(title)
        
        for issue in similar_issues:
            issue_state = self.get_issue_state(issue["number"])
            
            # Case 1: Issue already resolved
            if issue_state["is_resolved"]:
                return {
                    "action": "duplicate",
                    "status": "resolved",
                    "message": f"This appears to be a duplicate of #{issue['number']} which was already resolved",
                    "reference": issue["html_url"]
                }
            
            # Case 2: Open issue exists
            return {
                "action": "contribute",
                "status": "open",
                "message": self.contribute_to_issue(
                    issue_number=issue["number"],
                    comment=f"Additional context from similar report:\n\n**Title**: {title}\n**Description**: {body}"
                ),
                "reference": issue["html_url"]
            }
        
        # Case 3: New issue
        return {
            "action": "new",
            "status": "unresolved",
            "message": "No similar issues found - this appears to be a new report"
        }

    def fetch_issues_as_documents(self) -> List[Document]:
        """Fetch all issues as LangChain Documents for vector store"""
        issues = self._make_github_request("issues?state=all")
        docs = []
        # print("__________________")
       # print(issues[0])
        for issue in issues:
            
            
            metadata = {
                "number": issue["number"],
                "title": issue["title"],

                "url": issue["html_url"],
                "state": issue["state"],
                "created_at": issue["created_at"]
            }
            content = f"Issue #{issue['number']}: {issue['title']}\nState: {issue['state']}\n"
            content += f"Description:\n{issue['body']}\n" if issue.get("body") else ""
            
            docs.append(Document(page_content=content, metadata=metadata))
            # print("----------------n")
        # print(docs)
        return docs





from dotenv import load_dotenv
import os
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_core.documents import Document
from langchain.tools import tool
from github2 import GitHubIssueResolver
from note import note_tool

load_dotenv()

# New solution generation tool
@tool

def generate_solution(problem_description: str) -> str:
    """Generates technical solutions for coding problems"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY")  # Explicitly pass the key
    )
    prompt = f"""Provide a detailed solution for:
    {problem_description}
    
    Include:
    1. Root cause analysis
    2. Fixes with code examples
    3. Prevention tips"""
    
    return llm.invoke(prompt).content

def initialize_vector_store() -> AstraDBVectorStore:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return AstraDBVectorStore(
        embedding=embeddings,
        collection_name="github_issues",
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    )



def format_issue_response(issues: List[Document], resolution_context: Dict = None) -> str:
    """Enhanced formatting with resolution status and solutions"""
    response = ""
    
    if resolution_context:
        response += f"RESOLUTION STATUS: {resolution_context.get('status', 'unknown').upper()}\n"
        if resolution_context.get("reference"):
            response += f"Reference: {resolution_context['reference']}\n\n"
    
    for i, issue in enumerate(issues, 1):
        meta = issue.metadata
        response += (
            f"{i}. [{'OPEN' if meta.get('state') == 'open' else 'CLOSED'}] {meta.get('title', 'Untitled')}\n"
            f"   #{meta.get('number')} | Created: {meta.get('created_at')}\n"
            f"   URL: {meta.get('url')}\n"
        )
        
        # Improved description extraction
        if issue.page_content:
            # Split into lines and find the Description line
            lines = issue.page_content.split('\n')
            desc_lines = []
            found_desc = False
            for line in lines:
                if line.startswith("Description:"):
                    found_desc = True
                    continue
                if found_desc and line.strip():
                    desc_lines.append(line)
            
            description = ' '.join(desc_lines) if desc_lines else "No description"
            response += f"   Description: {description[:200]}{'...' if len(description) > 200 else ''}\n"
        
        # Add solution if available in metadata
        if meta.get('solution'):
            response += f"   🔧 Solution Preview: {meta['solution'][:150]}...\n"
        
        response += "\n"
    
    return response
# Initialize components
resolver = GitHubIssueResolver()
vstore = initialize_vector_store()

# Update vector store with solutions if available
if input("Update issues database? (y/N): ").lower() == "y":
    issues = resolver.fetch_issues_as_documents()
    try:
        vstore.delete_collection()
    except:
        pass
    vstore = initialize_vector_store()
    
    # Enhance issues with solutions from PRs/comments
    enhanced_issues = []
    for issue in issues:
        # print(issue)
        
        issue_number = issue.metadata['number']
        resolution_data = resolver.get_issue_state(issue_number)
        # print("-------")
        # print(resolution_data)
        # print("-------")
    
        if resolution_data['is_resolved']:
            # Try to extract solution from closing PR or comments
            solution = f"Fixed in PR: {resolution_data.get('pr_url', 'N/A')}"
            # print(resolution_data)
            # print(solution)
            
            issue.metadata['solution'] = solution
        enhanced_issues.append(issue)
    
    vstore.add_documents(enhanced_issues)
    # print(f"Loaded {len(enhanced_issues)} issues")
    print(f"Loaded issues")

# Configure retriever
retriever = vstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

retriever_tool = create_retriever_tool(
    retriever,
    name="github_issues",
    description="Search for GitHub issues including solutions. Use before reporting new issues."
)

# Agent setup with solution-focused prompt
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

tools = [retriever_tool, note_tool, generate_solution]  # Added solution tool

prompt = hub.pull("hwchase17/react").partial(
    instructions="""You are a technical support assistant. Follow these steps:
1. First check for existing similar issues
2. If found, show their solutions
3. For new issues, analyze and provide:
   - Root cause
   - Step-by-step fix
   - Code examples
   - Prevention tips
4. Format responses clearly with markdown"""
)

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Main loop
while True:
    user_input = input("\nDescribe your GitHub issue (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    
    # Check for existing issues
    similar_docs = retriever.invoke(user_input)
    # print(similar_docs)
    
    if similar_docs:
        docs = []
        similar_issues = []
        
        for doc in similar_docs:
            metadata = doc.metadata
            docs.append(doc)
            similar_issues.append({
                "number": metadata.get("number"),
                "title": metadata.get("title"),
                "html_url": metadata.get("url"),
                "state": metadata.get("state"),
                "body": doc.page_content,
                "solution": metadata.get("solution", "")
            })
        
        resolution_status = resolver.get_issue_state(similar_issues[0]["number"])
        print("\n" + format_issue_response(docs, {
            "status": "resolved" if resolution_status["is_resolved"] else "open",
            "reference": similar_issues[0]["html_url"]
        }))
        
        action = input("\nChoose action: [1] Comment on existing  [2] Continue as new: ")
        if action == "1":
            comment = input("Enter your comment: ")
            print(resolver.contribute_to_issue(similar_issues[0]["number"], comment))
            continue
       
    
    # For new issues or when user chooses to proceed
    print("\nAnalyzing your issue and generating solution...")
    result = executor.invoke({
        "input": f"Help solve this issue: {user_input}",
        "existing_issues": format_issue_response(docs) if similar_docs else "No similar issues found"
    })
    print(result["output"])
