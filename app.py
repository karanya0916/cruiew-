import os
from dotenv import load_dotenv

from crewai import Agent, LLM, Task, Crew, Process


# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/openrouter/free")
MAX_OUTPUT_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "1800"))
if MAX_OUTPUT_TOKENS < 1:
    raise ValueError("OPENROUTER_MAX_TOKENS must be a positive integer.")

# ==========================================================
# OpenRouter LLM
# ==========================================================

llm = LLM(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
    max_tokens=MAX_OUTPUT_TOKENS,
)

print("\nOpenRouter model configured.")


# ==========================================================
# 1. Project Manager
# ==========================================================

project_manager = Agent(
    role="Project Manager",
    goal="Manage the software project, gather requirements, create a plan, and prepare final documentation.",
    backstory="""
    You are an experienced IT Project Manager with 15+ years of experience.
    You convert client ideas into clear requirements, plan the work,
    and produce professional project documentation.
    """,
    llm=llm,
    allow_delegation=False,
    memory=False,
    verbose=False
)


# ==========================================================
# 2. Software Architect
# ==========================================================

software_architect = Agent(
    role="Software Architect",
    goal="Design the complete software architecture, select technologies, and define system structure.",
    backstory="""
    You are a Senior Software Architect.
    You choose the right tech stack including frontend, backend,
    database, authentication, and deployment.
    You design scalable and secure systems following industry best practices.
    """,
    llm=llm,
    allow_delegation=False,
    memory=False,
    verbose=False
)


# ==========================================================
# 3. Full-Stack Developer
# ==========================================================

developer = Agent(
    role="Full-Stack Developer",
    goal="Build both the backend and frontend of the application.",
    backstory="""
    You are a skilled Full-Stack Developer.
    You design REST APIs, business logic, database schemas,
    and responsive frontend applications.
    You follow modern development standards.
    """,
    llm=llm,
    allow_delegation=False,
    memory=False,
    verbose=False
)


# ==========================================================
# 4. QA Engineer
# ==========================================================

qa_engineer = Agent(
    role="QA Engineer",
    goal="Review the entire solution, identify issues, and suggest improvements.",
    backstory="""
    You are a Quality Assurance expert.
    You review code, architecture, security, performance,
    and testing strategies.
    You provide actionable feedback to ensure a robust,
    high-quality product.
    """,
    llm=llm,
    allow_delegation=False,
    memory=False,
    verbose=False
)


print("\nAll 4 AI Agents Created Successfully!")


# ==========================================================
# User Input
# ==========================================================

problem_statement = input("\nEnter Project Idea:\n\n")
if not problem_statement.strip():
    raise ValueError("Project idea cannot be empty.")


# ==========================================================
# 1. Planning Task
# ==========================================================

planning_task = Task(
    description=f"""
    The client wants: {problem_statement}

    Create a concise project plan (max 300 words) covering:

    - Project goal and scope
    - Main modules / features
    - Functional and non-functional requirements
    - High-level workflow
    - Deliverables
    """,

    expected_output="""
    A concise project plan with clear requirements (maximum 300 words).
    """,

    agent=project_manager
)


# ==========================================================
# 2. Architecture Task
# ==========================================================

architecture_task = Task(
    description="""
    Based on the project plan provided, design the software architecture.

    Include:

    - Technology stack
    - Frontend
    - Backend
    - Database
    - Authentication
    - Deployment
    - Folder structure
    - Explanation of technology choices
    """,

    expected_output="""
    A concise architecture document covering the requested sections.
    """,

    agent=software_architect,

    context=[planning_task]
)


# ==========================================================
# 3. Development Task
# ==========================================================

development_task = Task(
    description="""
    Based on the architecture document, create a detailed development plan
    covering both backend and frontend.

    Include:

    - Database tables / schema
    - REST API endpoints
    - Business logic
    - Authentication mechanism
    - Frontend pages
    - Components
    - Routing
    - UI/UX
    - Development steps
    """,

    expected_output="""
    A concise development plan covering the requested backend and frontend sections.
    """,

    agent=developer,

    context=[architecture_task]
)


# ==========================================================
# 4. QA Task
# ==========================================================

qa_task = Task(
    description="""
    Review the entire solution including:

    - Project plan
    - Architecture
    - Development plan

    Provide:

    - Strengths and weaknesses
    - Security improvements
    - Performance improvements
    - Testing strategy
    - Deployment suggestions
    - Overall quality assessment
    """,

    expected_output="""
    A concise QA review with actionable feedback.
    """,

    agent=qa_engineer,

    context=[
        planning_task,
        architecture_task,
        development_task
    ]
)


print("\nAll 4 Tasks Created Successfully!")


# ==========================================================
# Create Crew
# ==========================================================

software_company = Crew(
    agents=[
        project_manager,
        software_architect,
        developer,
        qa_engineer
    ],

    tasks=[
        planning_task,
        architecture_task,
        development_task,
        qa_task
    ],

    process=Process.sequential,

    verbose=True,

    memory=False
)


print("\nCrew Created Successfully!")


# ==========================================================
# Execute Crew
# ==========================================================

print("\nExecuting Crew...\n")

result = software_company.kickoff()


# ==========================================================
# Final Output
# ==========================================================

print("\n" + "=" * 80)
print("           AI SOFTWARE COMPANY (4 AGENTS)")
print("=" * 80)

print(result.raw)

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 80)