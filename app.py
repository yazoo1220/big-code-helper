import streamlit as st
from streamlit_ace import st_ace
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document

model = st.sidebar.selectbox('model',['gpt-3.5-turbo','gpt-4'])
language = st.sidebar.selectbox('language',['python','javascript','typescript','markdown'])
format = st.sidebar.selectbox('output',['only code','markdown'])

with st.container():
    request = st.text_input(label='request', value='add comments to each meaningful block and return the code with those comments')

col1, col2 = st.columns(2)
with col1.container():
    content = Document(page_content=st_ace(theme='terminal'), metadata={})

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

if language == "python":
    language_type = Language.PYTHON
elif language in ["typescript","javascript"]:
    language_type = Language.JS
else:
    language_type = Language.PYTHON
    
text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=500,
    chunk_overlap=0,
    language=language_type
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


if col1.button('submit'):
    with col2.container():
        with st.spinner(text='in progress...'):
            texts = text_splitter.split_documents([content])
            prompt=ChatPromptTemplate.from_template(
                "You are a helpful assistant. Please {request}. Codes should be in code blocks starting with ```{language}\n\nCODE: {input}",
            )
            if format == 'only code':
                chain = prompt | ChatOpenAI(temperature=0,model_name=model) | StrOutputParser() | _sanitize_output
            else:
                chain = prompt | ChatOpenAI(temperature=0,model_name=model) | StrOutputParser()
            
            result = []
            with st.expander(label='texts'):
                st.write(texts)
            
            for text in texts:
                print(text)
                result.append(chain.invoke({"input":text.page_content, "request":request, "language": language_type}))
    
            if format == 'only code':
                st.code(''.join(result), line_numbers=True)
            else:
                st.markdown(''.join(result))
        
