import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧠 Personal Chatbot with Flan-T5")
st.caption("🚀 Created by Rushikesh Jare | Data Science Portfolio Project")

user_input = st.text_input("You:", placeholder="e.g. What is the capital of India?")
submit = st.button("Ask")

if submit and user_input:
    # Flan-T5 expects input prompts like instructions
    prompt = f"Answer this question: {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.markdown("**Bot:** " + response)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "<small>© 2025 Rushikesh Jare · Made with ❤️ using Streamlit + Hugging Face</small>"
    "</div>",
    unsafe_allow_html=True
)
