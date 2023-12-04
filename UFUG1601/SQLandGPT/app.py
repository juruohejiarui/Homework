import streamlit as st
import mygpt as mg
gpt = mg.GPT()
if gpt.load_GPT_DB_from_json_file() == False:
  gpt.build_GPT_DB_from_directory("data")
  gpt.store_GPT_DB_to_json_file()
  
st.title("My first GPT App")
st.write("Give me a first word")
word = st.text_input("Word")
n_tokens = st.slider("Number of tokens", 10, 300, 10)
button = st.button("Submit")
if button:
    st.write(gpt.infer(word, n_tokens))
