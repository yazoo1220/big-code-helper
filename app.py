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
    chunk_size=500,
    chunk_overlap=0,
    language=Language.PYTHON
)

import re

def _sanitize_output(text: str):
    # Check if input is a string
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
        return text
        
    # Find all code blocks
    code_blocks = re.findall(r'```.*?\n(.*?)```', text, re.DOTALL)
    
    # Check if any code blocks were found
    if not code_blocks:
        raise ValueError("No code blocks found in the input text")
        return text
        
    return '\n'.join(code_blocks)


if st.button('submit'):
    with st.spinner(text='in progress...'):
        texts = text_splitter.split_documents([content])
        prompt=ChatPromptTemplate.from_template(
            "You are a helpful assistant that add comments to each meaningful block and return the code with those cmoments. The code should be in a code block starting with ```python\n\nCODE: {input}",
        )
        chain = prompt | ChatOpenAI() | StrOutputParser() | _sanitize_output
        result = []
        with st.expander(label='texts'):
            st.write(texts)
        
        for text in texts:
            print(text)
            result.append(chain.invoke({"input":text.page_content}))

        st.code(''.join(result), line_numbers=True)
        # result_pane = st_ace(value=''.join(result), theme='nord_dark')
    
