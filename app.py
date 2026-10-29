import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="LegalMind - Assistant Juridique", page_icon="⚖️", layout="wide")
st.title("⚖️ LegalMind")
st.markdown("Votre assistant IA pour le droit marocain (Code du Travail, DOC, Droit des Sociétés).")

# ==========================================
# CHARGEMENT DE LA BASE ET DU MODÈLE (Caché)
# ==========================================
@st.cache_resource
def load_rag_system():
    # 1. Charger les embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 2. Charger la base ChromaDB existante
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 3. Configurer le LLM via Groq (Gratuit et rapide)
    # Vous devez créer une clé API gratuite sur console.groq.com
    llm = ChatGroq(
        model="mixtral-8x7b-32768", 
        temperature=0.1,
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    # 4. Créer le prompt strict
    template = """Tu es un assistant juridique spécialisé dans le droit marocain.
Réponds à la question en te basant UNIQUEMENT sur le contexte ci-dessous.
Si l'information n'est pas dans le contexte, dis : "Je ne trouve pas cette information dans les textes fournis."
Cite la source (ex: Code du Travail, Article X) si possible.

Contexte :
{context}

Question : {question}

Réponse en français :"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 5. Créer la chaîne RAG
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

try:
    rag_chain, retriever = load_rag_system()
except Exception as e:
    st.error(f"Erreur lors du chargement du système : {e}")
    st.stop()

# ==========================================
# INTERFACE UTILISATEUR
# ==========================================
st.subheader("Posez votre question juridique")

# Champ de texte
question = st.text_input("Exemple : Mon employé peut-il démissionner sans préavis ?", "")

if st.button("Rechercher", type="primary"):
    if question.strip() == "":
        st.warning("Veuillez poser une question.")
    else:
        with st.spinner("Analyse des textes juridiques en cours..."):
            try:
                # Obtenir la réponse
                response = rag_chain.invoke(question)
                
                # Afficher la réponse
                st.success("Réponse :")
                st.write(response)
                
                # Afficher les sources (chunks utilisés)
                with st.expander("📚 Voir les extraits juridiques utilisés (Sources)"):
                    docs = retriever.invoke(question)
                    for i, doc in enumerate(docs):
                        source = doc.metadata.get('source', 'Inconnu')
                        page = doc.metadata.get('page', '?')
                        st.markdown(f"**Source {i+1}** : `{source}` (page {page})")
                        st.info(doc.page_content)
                        st.markdown("---")
                        
            except Exception as e:
                st.error(f"Erreur lors de la génération : {e}")

st.markdown("---")
st.caption("Projet prototype - Ne remplace pas un avis juridique professionnel.")