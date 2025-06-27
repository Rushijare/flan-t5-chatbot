import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model (cached)
@st.cache_resource
def load_model():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model

# Load once
tokenizer, model = load_model()

# App Title & Branding
st.title("🧠 AI Chatbot using Flan-T5")
st.caption("🚀 Created by Rushikesh Jare | Data Science Portfolio Project")

# Ask user input
user_input = st.text_input("You:", placeholder="Ask me anything...")
submit = st.button("Ask")

if submit and user_input:
    with st.spinner("Thinking..."):
        # Instruction-style prompt
        prompt = f"Answer this question: {user_input}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.markdown("**Bot:** " + response)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "<small>© 2025 <b>Rushikesh Jare</b> · Made with ❤️ using Streamlit + Hugging Face</small>"
    "</div>",
    unsafe_allow_html=True
)
