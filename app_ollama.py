import streamlit as st
import requests
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import json
import time
import os
from dotenv import load_dotenv

# โหลดข้อมูลความลับจากไฟล์ .env
load_dotenv()

# --- 1. CONFIGURATION ---
S2_API_KEY = os.getenv("S2_API_KEY") 
BASE_URL = "https://api.semanticscholar.org/graph/v1"

st.set_page_config(layout="wide", page_title="Research Knowledge Graph")

# --- 2. UI HEADER ---
st.title("Research Paper Knowledge Graph & Recommender")
st.markdown("ระบบดึงข้อมูลจาก Semantic Scholar และใช้ Ollama (Local LLM) อ่าน Abstract เพื่อสกัด Method, Dataset, และ Metric")

# --- 3. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("การตั้งค่า")
    query = st.text_input("หัวข้อที่ต้องการค้นหา:", "Large Language Models")
    limit = st.slider("จำนวนเปเปอร์เริ่มต้น:", 5, 300, 10)

# --- 4. API FUNCTIONS (Semantic Scholar) ---
@st.cache_data(show_spinner=False)
def fetch_papers_with_limit(query, limit):
    headers = {"x-api-key": S2_API_KEY, "User-Agent": "Mozilla/5.0"}
    params = {"query": query, "limit": limit, "fields": "title,authors,year,references,abstract"}
    
    try:
        # หน่วงเวลา 1 วินาทีให้ Semantic Scholar
        time.sleep(1) 
        response = requests.get(f"{BASE_URL}/paper/search", params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            st.error(f"Semantic Scholar API Error: {response.status_code}")
            return []
    except Exception as e:
        return []

# --- 5. AI INFORMATION EXTRACTION (Ollama Local) ---
def extract_info_with_ollama(abstract):
    if not abstract:
        return {"methods": [], "datasets": [], "metrics": [], "topics": []}
    
    prompt = f"""
    You are an AI researcher assistant. Read the following abstract and extract information.
    Return ONLY a JSON object with this exact structure:
    {{
        "methods": ["list", "of", "methods or architectures used"],
        "datasets": ["list", "of", "datasets or benchmarks used"],
        "metrics": ["list", "of", "evaluation metrics used"]
        "topics": ["list", "of", "general research topics or keywords"]
    }}
    If a category is not mentioned, use an empty list []. Keep items very concise (1-3 words).
    
    Abstract: {abstract}
    """
    
    try:
        # ยิงคำขอไปที่ Ollama
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3", 
            "prompt": prompt,
            "format": "json",   
            "stream": False
        })
        
        if response.status_code == 200:
            result_text = response.json().get('response', '')
            
            # --- โค้ดทำความสะอาดข้อความจาก AI (กันบั๊กข้อมูลว่างเปล่า) ---
            # 1. ลบเครื่องหมาย Markdown ที่ AI อาจจะแถมมา
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            # 2. ตัดเอาเฉพาะข้อความที่อยู่ระหว่างวงเล็บปีกกา
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                clean_json_str = result_text[start_idx:end_idx+1]
                try:
                    return json.loads(clean_json_str)
                except json.JSONDecodeError:
                    return {"methods": [], "datasets": [], "metrics": [], "topics": []}
            else:
                return {"methods": [], "datasets": [], "metrics": [], "topics": []}
            
        else:
            return {"methods": [], "datasets": [], "metrics": [], "topics": []}
            
    except requests.exceptions.ConnectionError:
        st.error("ไม่สามารถเชื่อมต่อกับ Ollama ได้! โปรดตรวจสอบว่าโปรแกรม Ollama เปิดอยู่และทำงานปกติ")
        return {"methods": [], "datasets": [], "metrics": [], "topics": []}
    except Exception as e:
        return {"methods": [], "datasets": [], "metrics": [], "topics": []}

# --- 6. GRAPH CONSTRUCTION ---
def build_knowledge_graph(initial_papers):
    G = nx.DiGraph()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_papers = len(initial_papers)
    
    for i, p in enumerate(initial_papers):
        p_id = p.get('paperId')
        if not p_id: continue
        
        status_text.text(f"กำลังให้ Ollama อ่าน Abstract เปเปอร์ที่ {i+1}/{total_papers}...")
        progress_bar.progress((i + 1) / total_papers)
        
        # 6.1 โหนด Paper และ Author
        G.add_node(p_id, label=p['title'][:30], title=p['title'], type='Paper', color='#74b9ff', shape='box', size=25)
        
        authors = p.get('authors') or []
        for author in authors:
            if not author: continue
            a_id = author.get('authorId')
            if a_id:
                G.add_node(a_id, label=author['name'], title=f"Author: {author['name']}", type='Author', color='#55efc4', shape='ellipse', size=15)
                G.add_edge(a_id, p_id, label='writes', color='#b2bec3')
        
        # 6.2 โหนด References
        raw_refs = p.get('references') or []
        for ref in raw_refs[:2]:
            if not ref: continue
            ref_id = ref.get('paperId')
            if ref_id:
                if ref_id not in G:
                    G.add_node(ref_id, label=ref['title'][:30], title=ref['title'], type='Paper', color='#fab1a0', shape='box', size=20)
                G.add_edge(p_id, ref_id, label='cites', color='#ff7675')
                
        # 6.3 โหนดข้อมูลเชิงลึก (จาก Ollama)
        abstract = p.get('abstract')
        if abstract:
            extracted_data = extract_info_with_ollama(abstract)
            
            for method in extracted_data.get('methods', []):
                m_node = f"M_{method}"
                G.add_node(m_node, label=method, title=f"Method: {method}", type='Method', color='#a29bfe', shape='triangle', size=20)
                G.add_edge(p_id, m_node, label='uses_method', color='#a29bfe')
                
            for dataset in extracted_data.get('datasets', []):
                d_node = f"D_{dataset}"
                G.add_node(d_node, label=dataset, title=f"Dataset: {dataset}", type='Dataset', color='#ffeaa7', shape='database', size=20)
                G.add_edge(p_id, d_node, label='uses_dataset', color='#ffeaa7')
                
            for metric in extracted_data.get('metrics', []):
                mt_node = f"MT_{metric}"
                G.add_node(mt_node, label=metric, title=f"Metric: {metric}", type='Metric', color='#fd79a8', shape='diamond', size=20)
                G.add_edge(p_id, mt_node, label='evaluates_on', color='#fd79a8')
                
            # --- ส่วนที่เพิ่มใหม่: โหนด Topic ---
            for topic in extracted_data.get('topics', []):
                t_node = f"T_{topic}"
                # ใช้รูปดาว (star) และสีเขียวอมฟ้า (#81ecec) เพื่อแยกให้ชัดเจนจากโหนดอื่นๆ
                G.add_node(t_node, label=topic, title=f"Topic: {topic}", type='Topic', color='#81ecec', shape='star', size=25)
                G.add_edge(p_id, t_node, label='has_topic', color='#81ecec')

    status_text.success("วิเคราะห์เสร็จสมบูรณ์!")
    return G

# --- 7. RECOMMENDATION ALGORITHM ---
def get_recommendations(G, target_paper_id):
    try:
        pers = {n: 0.0 for n in G.nodes()}
        if target_paper_id in pers:
            pers[target_paper_id] = 1.0
        scores = nx.pagerank(G, personalization=pers)
        paper_scores = [(node, scores[node]) for node in G.nodes() if G.nodes[node].get('type') == 'Paper' and node != target_paper_id]
        return sorted(paper_scores, key=lambda x: x[1], reverse=True)[:5]
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
        return []

# --- 8. MAIN LOGIC (ส่วนสร้างกราฟ) ---
if st.button("ค้นหาและสร้าง Knowledge Graph"):
    with st.spinner("กำลังเริ่มกระบวนการ..."):
        data = fetch_papers_with_limit(query, limit)
        
        if data:
            G = build_knowledge_graph(data)
            
            # บันทึกกราฟลงระบบชั่วคราว เพื่อไม่ให้กราฟหายเวลาคลิกปุ่มอื่น
            st.session_state['graph'] = G
            st.session_state['graph_ready'] = True
            
            # วาดและเซฟ HTML
            net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white", directed=True, notebook=True)
            net.from_nx(G)
            net.save_graph("research_graph.html")
            
            st.success(f"สร้างกราฟสำเร็จ! พบทั้งหมด {len(G.nodes())} Nodes")
        else:
            st.warning("ไม่พบข้อมูล หรือ API ติดปัญหา")

# --- ส่วนแสดงผลกราฟ (แยกออกมาอยู่นอกปุ่มเพื่อให้แสดงค้างไว้ได้) ---
if st.session_state.get('graph_ready', False):
    st.subheader("Knowledge Graph Exploration")
    try:
        with open("research_graph.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=720)
    except FileNotFoundError:
        pass

# --- 9. RECOMMENDATION SECTION ---
if 'graph' in st.session_state:
    st.divider()
    st.subheader("ระบบแนะนำเปเปอร์ (Graph-based Recommendation)")
    current_graph = st.session_state['graph']
    paper_list = {current_graph.nodes[n]['title']: n for n in current_graph.nodes() if current_graph.nodes[n].get('type') == 'Paper'}
    
    selected_paper = st.selectbox("เลือกเปเปอร์ที่คุณสนใจ เพื่อดูรายการแนะนำที่เกี่ยวข้องกัน:", list(paper_list.keys()))
    
    if st.button("ดูคำแนะนำ"):
        target_id = paper_list[selected_paper]
        recs = get_recommendations(current_graph, target_id)
        
        if recs:
            st.write("### เปเปอร์ที่แนะนำสำหรับคุณ:")
            for i, (p_id, score) in enumerate(recs):
                p_title = current_graph.nodes[p_id].get('title', 'Unknown')
                st.write(f"{i+1}. **{p_title}** (คะแนนความเกี่ยวข้องกัน: {score:.4f})")