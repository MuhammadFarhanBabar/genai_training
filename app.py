import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Pinecone as pc
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from sqlalchemy import text, inspect
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse

# API Keys - Load from environment variables for security
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_ENV'] = os.getenv("PINECONE_ENV", "us-east-1")
os.environ['INDEX_NAME'] = os.getenv("INDEX_NAME")

# Database
Database_url = "sqlite:///finance.db"
engine = create_engine(Database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Embeddings and Vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = pc.from_existing_index(
    index_name="pinecorntesindx",
    embedding=embeddings,
)

groq_llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. "
               "Use ONLY the following context to answer the question. "
               "Do not use any external knowledge or information not provided in the context. "
               "If the answer is not in the context, say 'I cannot find the answer in the provided context.'"
               "\n\nContext: {context}"),
    ("human", "{input}"),
])

chain = create_retrieval_chain(
    vectorstore.as_retriever(),
    create_stuff_documents_chain(groq_llm, prompt)
)

# Agent State and Functions (copy from notebook)
class AgentState(TypedDict):
    question: str
    sql_query: str
    query_result: str
    query_rows: list
    attempts: int
    relevance: str
    sql_error: bool

description = """
    "Year": "The fiscal year for which the financial data is reported.",
    "Company": "The name of the company like [AAPL,MSFT,GOOG,PYPL,AIG,PCG,SHLDQ,,MCD,BCS,NVDA,INTC,AMZN].",
    "Category": "The industry or sector the company operates in.",
    "Market Cap(in B USD)": "Market capitalization in billions of US dollars.",
    "Revenue": "Total revenue generated in the fiscal year (in USD).",
    "GrossProfit": "Gross profit earned in the fiscal year (in USD).",
    "NetIncome": "Net income after all operating expenses, taxes, and interest (in USD).",
    "Earning Per Share": "Earnings per share (EPS), indicating profitability per share.",
    "EBITDA": "Earnings before interest, taxes, depreciation, and amortization (in USD).",
    "Share Holder Equity": "Total equity held by shareholders at the end of the fiscal year (in USD).",
    "Cash Flow from Operating": "Net cash generated from core operating activities (in USD).",
    "Cash Flow from Investing": "Net cash used or generated through investing activities (in USD).",
    "Cash Flow from Financial Activities": "Net cash inflow or outflow from financing activities (in USD).",
    "Current Ratio": "Liquidity ratio calculated as current assets divided by current liabilities.",
    "Debt/Equity Ratio": "Financial leverage ratio measuring the proportion of debt to equity.",
    "ROE": "Return on Equity – percentage return on shareholders’ equity.",
    "ROA": "Return on Assets – percentage return relative to total assets.",
    "ROI": "Return on Investment – a measure of profitability relative to invested capital.",
    "Net Profit Margin": "Percentage of revenue that remains as profit after all expenses.",
    "Free Cash Flow per Share": "Cash available to equity holders on a per-share basis (in USD).",
    "Return on Tangible Equity": "Return based on equity excluding intangible assets.",
    "Number of Employees": "Total number of full-time employees in the company during the fiscal year.",
    "Inflation Rate(in US)": "Annual inflation rate in the United States for the corresponding year."
"""

def get_database_schema(engine):
    inspector = inspect(engine)
    schema = ""
    for table_name in inspector.get_table_names():
        schema += f"Table: {table_name}\n"
        for column in inspector.get_columns(table_name):
            col_name = column["name"]
            col_type = str(column["type"])
            if column.get("primary_key"):
                col_type += ", Primary Key"
            if column.get("foreign_keys"):
                fk = list(column["foreign_keys"])[0]
                col_type += f", Foreign Key to {fk.column.table.name}.{fk.column.name}"
            schema += f"- {col_name}: {col_type}\n"
        schema += "\n"
    return schema + description

class CheckRelevance(BaseModel):
    relevance: str = Field(description="Indicates whether the question is related to the database schema. 'relevant' or 'not_relevant'.")

def check_relevance(state: AgentState, config: RunnableConfig):
    question = state["question"]
    schema = get_database_schema(engine)
    system = """You are an assistant that determines whether a given question is related to the following database schema.
Schema:
{schema}
Respond with only "relevant" or "not_relevant".
""".format(schema=schema)
    human = f"Question: {question}"
    check_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    structured_llm = groq_llm.with_structured_output(CheckRelevance)
    relevance_checker = check_prompt | structured_llm
    relevance = relevance_checker.invoke({})
    state["relevance"] = relevance.relevance
    return state

class ConvertToSQL(BaseModel):
    sql_query: str = Field(description="The SQL query corresponding to the user's natural language question.")

def convert_nl_to_sql(state: AgentState, config: RunnableConfig):
    question = state["question"]
    system = """You are an assistant that converts natural language questions into SQL queries based.Use the exact DataBase Name,Table and Columns Name:
            Table Name = ['finance_report']
            Column Name and Data Type:
                - 'Year' bigint
                - 'Company' text
                - Category text
                - 'Market Cap(in B USD)' double
                - 'Revenue' double
                - 'GrossProfit' double
                - 'NetIncome' double
                - 'Earning Per Share' double
                - 'EBITDA' double
                - 'Share Holder Equity' double
                - 'Cash Flow from Operating' double
                - 'Cash Flow from Investing' double
                - 'Cash Flow from Financial Activities' double
                - 'Current Ratio' double
                - 'DebtEquityRatio' double
                - 'ROE' double
                - 'ROA' double
                - 'ROI' double
                - 'Net Profit Margin' double
                - 'Free Cash Flow per Share' double
                - 'Return on Tangible Equity' double
                - 'Number of Employees' bigint
                - 'Inflation Rate(in US)' double
            Columns Description:
                "Year": "The fiscal year for which the financial data is reported.",
                "Company": "The name of the company like [AAPL,MSFT,GOOG,PYPL,AIG,PCG,SHLDQ,,MCD,BCS,NVDA,INTC,AMZN].",
                "Category": "The industry or sector the company operates in.",
                "Market Cap(in B USD)": "Market capitalization in billions of US dollars.",
                "Revenue": "Total revenue generated in the fiscal year (in USD).",
                "Gross Profit": "Gross profit earned in the fiscal year (in USD).",
                "NetIncome": "Net income after all operating expenses, taxes, and interest (in USD).",
                "Earning Per Share": "Earnings per share (EPS), indicating profitability per share.",
                "EBITDA": "Earnings before interest, taxes, depreciation, and amortization (in USD).",
                "Share Holder Equity": "Total equity held by shareholders at the end of the fiscal year (in USD).",
                "Cash Flow from Operating": "Net cash generated from core operating activities (in USD).",
                "Cash Flow from Investing": "Net cash used or generated through investing activities (in USD).",
                "Cash Flow from Financial Activities": "Net cash inflow or outflow from financing activities (in USD).",
                "Current Ratio": "Liquidity ratio calculated as current assets divided by current liabilities.",
                "DebtEquityRatio": "Financial leverage ratio measuring the proportion of debt to equity.",
                "ROE": "Return on Equity – percentage return on shareholders’ equity.",
                "ROA": "Return on Assets – percentage return relative to total assets.",
                "ROI": "Return on Investment – a measure of profitability relative to invested capital.",
                "Net Profit Margin": "Percentage of revenue that remains as profit after all expenses.",
                "Free Cash Flow per Share": "Cash available to equity holders on a per-share basis (in USD).",
                "Return on Tangible Equity": "Return based on equity excluding intangible assets.",
                "Number of Employees": "Total number of full-time employees in the company during the fiscal year.",
                "Inflation Rate(in US)": "Annual inflation rate in the United States for the corresponding year."
                
                Ensure that all query-related data is scoped. 
                Provide only the SQL query without any explanations. Alias columns appropriately to match the expected keys in the result.
                Note Do not use '_' in columns. Use the same column name that are mentioned above
            """
    convert_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Question: {question}")])
    structured_llm = groq_llm.with_structured_output(ConvertToSQL)
    sql_generator = convert_prompt | structured_llm
    result = sql_generator.invoke({"question": question})
    state["sql_query"] = result.sql_query
    return state

def execute_sql(state: AgentState):
    sql_query = state["sql_query"].strip()
    session = SessionLocal()
    try:
        result = session.execute(text(sql_query))
        if sql_query.lower().startswith("select"):
            rows = result.fetchall()
            columns = result.keys()
            if rows:
                header = ", ".join(columns)
                state["query_rows"] = [dict(zip(columns, row)) for row in rows]
                data = "; ".join([", ".join([f"{key}: {value}" for key, value in row.items()]) for row in state["query_rows"]])
                formatted_result = f"{header}\n{data}"
            else:
                state["query_rows"] = []
                formatted_result = "No results found."
            state["query_result"] = formatted_result
            state["sql_error"] = False
        else:
            session.commit()
            state["query_result"] = "The action has been successfully completed."
            state["sql_error"] = False
    except Exception as e:
        state["query_result"] = f"Error executing SQL query: {str(e)}"
        state["sql_error"] = True
    finally:
        session.close()
    return state

def generate_human_readable_answer(state: AgentState):
    sql = state["sql_query"]
    result = state["query_result"]
    query_rows = state.get("query_rows", [])
    sql_error = state.get("sql_error", False)
    system = """
        You are a professional financial data analyst assistant. Your role is to interpret SQL query results and convert them into clear, executive-level summaries that are ready for dashboards, reports, or stakeholder communication.
        Column Definitions:
            "Year": "The fiscal year for which the financial data is reported.",
            "Company": "The name of the company like [AAPL,MSFT,GOOG,PYPL,AIG,PCG,SHLDQ,,MCD,BCS,NVDA,INTC,AMZN].",
            "Category": "The industry or sector the company operates in.",
            "Market Cap(in B USD)": "Market capitalization in billions of US dollars.",
            "Revenue": "Total revenue generated in the fiscal year (in USD).",
            "Gross Profit": "Gross profit earned in the fiscal year (in USD).",
            "NetIncome": "Net income after all operating expenses, taxes, and interest (in USD).",
            "Earning Per Share": "Earnings per share (EPS), indicating profitability per share.",
            "EBITDA": "Earnings before interest, taxes, depreciation, and amortization (in USD).",
            "Share Holder Equity": "Total equity held by shareholders at the end of the fiscal year (in USD).",
            "Cash Flow from Operating": "Net cash generated from core operating activities (in USD).",
            "Cash Flow from Investing": "Net cash used or generated through investing activities (in USD).",
            "Cash Flow from Financial Activities": "Net cash inflow or outflow from financing activities (in USD).",
            "Current Ratio": "Liquidity ratio calculated as current assets divided by current liabilities.",
            "DebtEquityRatio": "Financial leverage ratio measuring the proportion of debt to equity.",
            "ROE": "Return on Equity – percentage return on shareholders’ equity.",
            "ROA": "Return on Assets – percentage return relative to total assets.",
            "ROI": "Return on Investment – a measure of profitability relative to invested capital.",
            "Net Profit Margin": "Percentage of revenue that remains as profit after all expenses.",
            "Free Cash Flow per Share": "Cash available to equity holders on a per-share basis (in USD).",
            "Return on Tangible Equity": "Return based on equity excluding intangible assets.",
            "Number of Employees": "Total number of full-time employees in the company during the fiscal year.",
            "Inflation Rate(in US)": "Annual inflation rate in the United States for the corresponding year."
        Instructions for the LLM:
            Analyze dynamically: Use only the columns provided in the result set. Do not refer to missing data.
            Avoid filler language: Never use phrases like "more data is needed", "limited data", or "could not determine". Always write with confidence.
            Always generate a clean, executive-friendly summary, as if it were written by a senior analyst for business stakeholders.
            Highlight key financial metrics, year-over-year trends, and noteworthy performance changes.
            Focus on clarity, business value, and actionable insight — not technical jargon.  
         Output Style Examples:
        
        DO:
        - "In FY2022, Apple reported $394B in revenue and $99.8B in net income, delivering an EPS of $6.11. Despite a lower market cap compared to 2021, Apple maintained a strong ROE of 197% and added 10K new employees."
        - "Over a 5-year period, Apple’s net profit margin remained above 20%, with the highest EBITDA recorded in 2022 at $130.5B, signaling efficient operations despite inflation pressures."
        
        DO NOT:
        - "Based on the limited data available…"
        - "It would be helpful to have more information…"
        - "The company seems profitable but cannot confirm without more metrics…"
"""
    if sql_error:
        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", f"""SQL Query:\n{sql}\n\nResult:\n{result}\nFormulate a clear and understandable error message in a single sentence, informing them about the issue."""),
        ])
    elif sql.lower().startswith("select"):
        if not query_rows:
            generate_prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", f"""SQL Query:\n{sql}\nResult:\n{result}"""),
            ])
        else:
            generate_prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", f"""SQL Query:\n{sql}\nResult:\n{result}"""),
            ])
    else:
        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", f"""SQL Query:\n{sql}\nResult:\n{result}"""),
        ])
    human_response = generate_prompt | groq_llm | StrOutputParser()
    answer = human_response.invoke({})
    state["query_result"] = answer
    return state

class RewrittenQuestion(BaseModel):
    question: str = Field(description="The rewritten question.")

def regenerate_query(state: AgentState):
    question = state["question"]
    system = """You are an assistant that reformulates an original question to enable more precise SQL queries. Ensure that all necessary details, such as table joins, are preserved to retrieve complete and accurate data."""
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", f"Original Question: {question}\nReformulate the question to enable more precise SQL queries, ensuring all necessary details are preserved."),
    ])
    structured_llm = groq_llm.with_structured_output(RewrittenQuestion)
    rewriter = rewrite_prompt | structured_llm
    rewritten = rewriter.invoke({})
    state["question"] = rewritten.question
    state["attempts"] += 1
    return state

def generate_fallback_response(state: AgentState):
    user_query = state.get("question", "")
    try:
        response = chain.invoke({"input": user_query})
        answer = response["answer"]
        state["query_result"] = answer or "I cannot find the answer in the provided context."
    except Exception as ex:
        state["query_result"] = f"Sorry, I don't know the answer to that. ({str(ex)})"
    return state

def end_max_iterations(state: AgentState):
    state["query_result"] = "Please try again."
    return state

def relevance_router(state: AgentState):
    if state["relevance"].lower() == "relevant":
        return "convert_to_sql"
    else:
        return "generate_fallback_response"

def check_attempts_router(state: AgentState):
    if state["attempts"] < 3:
        return "convert_to_sql"
    else:
        return "end_max_iterations"

def execute_sql_router(state: AgentState):
    if not state.get("sql_error", False):
        return "generate_human_readable_answer"
    else:
        return "regenerate_query"

workflow = StateGraph(AgentState)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("convert_to_sql", convert_nl_to_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("generate_human_readable_answer", generate_human_readable_answer)
workflow.add_node("regenerate_query", regenerate_query)
workflow.add_node("generate_fallback_response", generate_fallback_response)
workflow.add_node("end_max_iterations", end_max_iterations)

workflow.add_conditional_edges("check_relevance", relevance_router, {
    "convert_to_sql": "convert_to_sql",
    "generate_fallback_response": "generate_fallback_response"
})
workflow.add_edge("convert_to_sql", "execute_sql")
workflow.add_conditional_edges("execute_sql", execute_sql_router, {
    "generate_human_readable_answer": "generate_human_readable_answer",
    "regenerate_query": "regenerate_query",
})
workflow.add_conditional_edges("regenerate_query", check_attempts_router, {
    "convert_to_sql": "convert_to_sql",
    "end_max_iterations": "end_max_iterations",
})
workflow.add_edge("generate_human_readable_answer", END)
workflow.add_edge("generate_fallback_response", END)
workflow.add_edge("end_max_iterations", END)
workflow.set_entry_point("check_relevance")
workflow_app = workflow.compile()

app = FastAPI()

@app.post("/query")
async def query_endpoint(question: str = Form(...)):
    try:
        result = workflow_app.invoke({"question": question, "attempts": 0})
        return JSONResponse(content={"query": question, "answer": result.get("query_result", "")})
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))