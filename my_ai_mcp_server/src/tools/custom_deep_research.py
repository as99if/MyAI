from datetime import datetime
from src.tools.google_search_tool import google_custom_search
from src.tools.groq import groq_inference


def custom_deep_research_by_planning(query: str, task_memory_messages: list = [], api_key: str = None, search_engine_id: str = None) -> dict:
    """Implementation of deep research with task-based memory and multi-stage reasoning"""
    
    # Step 1: Initial Planning
    planning_system_prompt = """You are a research planning assistant.
    Create a detailed research plan with specific sub-questions and approaches.
    Focus on breaking down complex topics into manageable parts."""

    planning_prompt = f"""Given the research topic: '{query}'
    1. Break down this topic into 3-5 key aspects to investigate
    2. Generate specific sub-questions for each aspect
    3. Suggest search queries for each sub-question
    Format your response in Markdown with clear sections."""

    research_plan, model = groq_inference(
        message=planning_prompt,
        system_message=planning_system_prompt,
        model="deepseek-coder",
        task_memory_messages=task_memory_messages
    )
    # TODO: change it to mistral inference

    # Step 2: Execute Search Queries
    search_results = []
    for sub_query in research_plan["sub_queries"]:
        results = google_custom_search(
            query=sub_query,
            google_custom_search_api_key=api_key,
            google_custom_search_engine_id=search_engine_id
        )
        search_results.append({"query": sub_query, "results": results})

    # Step 3: Analyze Search Results
    analysis_system_prompt = """You are a research analyst.
    Synthesize information from multiple sources.
    Identify key findings and patterns."""

    analysis_prompt = f"""Analyze these search results for the topic: '{query}'
    1. Identify main themes and findings
    2. Note any contradictions or gaps
    3. Highlight the most credible sources
    Format your response in Markdown."""

    analysis_results, model = groq_inference(
        message=analysis_prompt,
        system_message=analysis_system_prompt,
        model="deepseek-coder",
        task_memory_messages=[research_plan] + task_memory_messages
    )
    # TODO: change it to mistral inference


    # Step 4: Generate Conclusions
    conclusion_system_prompt = """You are a research synthesizer.
    Create comprehensive yet concise conclusions.
    Support claims with evidence from the research."""

    conclusion_prompt = f"""Based on all findings for '{query}':
    1. Summarize key conclusions
    2. Support with specific evidence
    3. Identify remaining questions
    Format in Markdown with citations."""

    final_conclusions, model = groq_inference(
        message=conclusion_prompt,
        system_message=conclusion_system_prompt,
        model="deepseek-coder",
        task_memory_messages=[research_plan, analysis_results] + task_memory_messages
    )

    return {
        "query": query,
        "research_plan": research_plan,
        "analysis": analysis_results,
        "conclusions": final_conclusions,
        "sources": search_results,
        "timestamp": datetime.now().isoformat()
    }