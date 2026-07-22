import os
import asyncio
import logging
import streamlit as st
from typing import Optional, Dict, List
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ BLOG GENERATOR ------------------
class BlogGenerator:
    def __init__(
        self,
        groq_api_key: str,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.tools = self._initialize_tools(google_api_key, google_cse_id)

    def _initialize_tools(self, google_api_key, google_cse_id) -> List[Tool]:
        tools = []

        wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=4000
        )
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

        tools.append(
            Tool(
                name="Wikipedia",
                func=wiki_tool.run,
                description="Search Wikipedia for factual information."
            )
        )

        if google_api_key and google_cse_id:
            search_wrapper = GoogleSearchAPIWrapper(
                google_api_key=google_api_key,
                google_cse_id=google_cse_id,
                k=5
            )
            tools.append(
                Tool(
                    name="GoogleSearch",
                    func=search_wrapper.run,
                    description="Search Google for real-time data."
                )
            )

        return tools

    async def _generate_section(self, prompt: str) -> str:
        template = ChatPromptTemplate.from_template(prompt)
        chain = template | self.llm | StrOutputParser()
        return await chain.ainvoke({})

    async def _generate_parallel(self, prompts: Dict[str, str]) -> Dict[str, str]:
        tasks = [self._generate_section(p) for p in prompts.values()]
        results = await asyncio.gather(*tasks)
        return dict(zip(prompts.keys(), results))

    def generate_blog(self, topic: str, word_count: int) -> Dict[str, str]:
        prompts = {
            "Introduction": f"Write a 150-word engaging introduction about '{topic}'.",
            "Main Content": f"Write a detailed explanation of '{topic}' (~{word_count - 300} words).",
            "Conclusion": f"Write a short conclusion summarizing '{topic}'."
        }

        sections = asyncio.run(self._generate_parallel(prompts))
        blog = self._assemble_blog(topic, sections, word_count)

        return {
            "blog": blog,
            "sections": sections,
            "word_count": len(blog.split())
        }

    def _assemble_blog(self, topic, sections, word_count) -> str:
        return f"""
# {topic}

*Target Word Count: {word_count}*

---

## Introduction
{sections['Introduction']}

---

## Main Content
{sections['Main Content']}

---

## Conclusion
{sections['Conclusion']}

---

*Generated using Groq + LangChain*
"""


# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="AI Blog Generator", layout="wide")
st.title("📝 AI Blog Generator (Groq + LangChain)")

with st.sidebar:
    st.header("🔐 API Keys")
    groq_api_key = st.text_input("Groq API Key", type="password")
    google_api_key = st.text_input("Google API Key (optional)", type="password")
    google_cse_id = st.text_input("Google CSE ID (optional)")

st.subheader("📌 Blog Settings")
topic = st.text_input("Blog Topic")
word_count = st.slider("Target Word Count", 500, 3000, 1500)

if st.button("🚀 Generate Blog"):
    if not groq_api_key or not topic:
        st.error("Groq API key and topic are required.")
    else:
        with st.spinner("Generating blog..."):
            generator = BlogGenerator(
                groq_api_key=groq_api_key,
                google_api_key=google_api_key,
                google_cse_id=google_cse_id
            )
            result = generator.generate_blog(topic, word_count)

        st.success("Blog generated successfully!")

        st.markdown(result["blog"])
        st.caption(f"Actual word count: {result['word_count']}")

        st.download_button(
            "⬇️ Download Markdown",
            result["blog"],
            file_name=f"{topic.replace(' ', '_')}.md"
        )
