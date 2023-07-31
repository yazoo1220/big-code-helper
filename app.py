import streamlit as st
from streamlit_ace import st_ace
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document


content = Document(page_content=st_ace(theme='terminal'), metadata={})

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=1000,
    chunk_overlap=0,
    language=Language.PYTHON
)

def _sanitize_output(text: str):
    try:
        _, after = text.split("```python")
        return after.split("```")[0]
    except ValueError:
        return text.split("```python")[1]

if st.button('submit'):
    with st.spinner(text='in progress...'):
        texts = text_splitter.split_documents([content])
        prompt=ChatPromptTemplate.from_template(
            "You are a helpful assistant that add comments to each meaningful block and return the code with those cmoments. The code should be in a code block starting with ```python\n\nCODE: {input}",
        )
        chain = prompt | ChatOpenAI() | StrOutputParser() | _sanitize_output
        result = []
        
        for text in texts:
            result.append(chain.invoke({"input":text.page_content}))
    
        result_pane = st_ace(value=''.join(result), theme='nord_dark')
    
