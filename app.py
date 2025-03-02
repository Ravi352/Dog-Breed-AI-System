import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import Tool
from langchain_ollama.llms import OllamaLLM

# Streamlit UI
st.title("Dog Breed AI System")

# User ID Input (for tracking user interactions)
user_id = st.text_input("Enter User ID:", key="user_id")

# Pre_Proocessing the Dataframe
df = pd.read_csv("PreProcessed.csv").rename(columns={"Unnamed: 0": "breed"}).reset_index(drop=True)


## Model embeddning creation and loading for the description of the breed
model = SentenceTransformer("all-MiniLM-L6-v2")
description_embeddings = model.encode(df["description"].tolist(), convert_to_tensor=True)
def get_breed_description(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, description_embeddings)
    best_match_idx = similarities.argmax().item()
    return df.iloc[best_match_idx]["description"]


semantic_search_tool = Tool(
    name="Breed Description Search",
    func=get_breed_description,
    description="Provides breed descriptions based on user queries."
)


llm = OllamaLLM(model="llama3.2:1b")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


prompt_template = PromptTemplate(
    template="""You are an AI assistant helping users with dog breed information.

    User ID: {user_id}

    Conversation History:
    {chat_history}
    
    User Query: {query}
    
    Answer:
    """,
    input_variables=["user_id", "chat_history", "query"]
)


pandas_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    agent_type="zero-shot-react-description",
    allow_dangerous_code=True
)


agent_executor = AgentExecutor(
    agent=pandas_agent,
    tools=[semantic_search_tool],
    verbose=True,
    handle_parsing_errors=True
)


query = st.text_input("Ask a query about dog breeds:")

if st.button("Get Answer"):
    if query and user_id:
        with st.spinner("Thinking..."):
            try:
                chat_history = memory.load_memory_variables({})["chat_history"]

                response = agent_executor.invoke({
                    "input": query,
                    "user_id": user_id,
                    "chat_history": chat_history
                })

                # Save conversation for the user
                memory.save_context({"input": query, "user_id": user_id}, {"output": response.get("output", response)})

                st.success(f" Answer: {response.get('output', response)}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter both User ID and a query.")

# Show conversation history for the user
st.subheader("Conversation History")
chat_history = memory.load_memory_variables({})["chat_history"]

for message in chat_history:
    if "user_id" in message and message["user_id"] == user_id:
        st.write(f"**{'User' if message.type == 'human' else 'AI'} ({user_id}):** {message.content}")
