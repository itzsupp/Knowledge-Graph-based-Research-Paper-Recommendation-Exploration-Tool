# Research Paper Knowledge Graph & Recommender

Prototype เครื่องมือสำรวจและวิเคราะห์ความสัมพันธ์ของงานวิจัยด้วย Knowledge Graph

เว็บแอปพลิเคชันนี้ทำงานแบบ Interactive โดยใช้ Semantic Scholar API ร่วมกับ Large Language Model (Ollama) แบบ Local เพื่อสร้าง Knowledge Graph จากงานวิจัย ช่วยให้นักวิจัยสามารถสำรวจความเชื่อมโยง ค้นหาเปเปอร์ที่เกี่ยวข้องกันผ่านอัลกอริทึมกราฟ และทำการค้นหาเชิงความหมายหรือ Semantic Search ที่ซับซ้อน

## ฟีเจอร์หลัก

1. **การสร้างกราฟความรู้ (Knowledge Graph Construction)**
   - ดึงข้อมูลเปเปอร์อัตโนมัติสูงสุด 100 ฉบับผ่าน Semantic Scholar API
   - ใช้ Local LLM (`llama3` ผ่าน Ollama) อ่าน Abstract เพื่อสกัดข้อมูล เช่น `Method`, `Dataset`, `Topic` และ `Evaluation Metric`
   - แมปคะแนนผลการประเมิน เช่น Accuracy ลงบนเส้นเชื่อมของกราฟโดยตรงเพื่อแสดงบริบทเชิงลึก

2. **การแสดงผลกราฟแบบโต้ตอบได้ (Interactive Graph Visualization)**
   - สร้าง UI กราฟด้วย `NetworkX` และ `PyVis`
   - ผู้ใช้สามารถลากโหนด ซูมเข้าออก และปรับแต่งค่าฟิสิกส์ของกราฟ เช่น ความห่างของโหนดได้

3. **ระบบแนะนำเปเปอร์ด้วยกราฟ (Graph-based Paper Recommendation)**
   - ใช้อัลกอริทึม Personalized PageRank บนกราฟแบบไม่มีทิศทาง (Undirected Graph)
   - แนะนำเปเปอร์ที่เกี่ยวข้องโดยคำนวณความสัมพันธ์ผ่าน Method, Dataset, Topic และการอ้างอิง Citation ที่ใช้ร่วมกัน

4. **การค้นหาเชิงความหมาย (Semantic Search)**
   - เป็นการสืบค้นจากโครงสร้างความสัมพันธ์ในกราฟ
   - ค้นหาเปเปอร์ที่ตรงตามเงื่อนไขความสัมพันธ์ที่กำหนด

## เทคโนโลยีที่ใช้

- **ภาษา:** Python
- **Frontend Framework:** Streamlit
- **การประมวลผลกราฟ:** NetworkX, PyVis
- **AI/LLM:** Ollama (Local Llama 3)
- **แหล่งข้อมูล:** Semantic Scholar API

## การติดตั้งและการใช้งาน

ทำตามขั้นตอนเหล่านี้เพื่อรันโปรเจกต์ในเครื่องของคุณ:

### 1. สิ่งที่ต้องเตรียม
- ติดตั้ง Python 3.8 ขึ้นไป
- ติดตั้งและเปิดใช้งาน [Ollama](https://ollama.com/) บนเครื่องของคุณ

### 2. ติดตั้งไลบรารีที่จำเป็น
Clone repository นี้ และติดตั้งแพ็กเกจ Python ทั้งหมดที่ต้องใช้ผ่านไฟล์ `requirements.txt`

```bash
pip install -r requirements.txt
```

### 3. ตั้งค่าระบบและรันแอปพลิเคชัน
ก่อนเริ่มใช้งานเว็บแอปพลิเคชัน ให้ทำการตั้งค่าและรันระบบตามลำดับดังนี้:

- เปิดใช้งาน Ollama: ดาวน์โหลดและรันโมเดล Llama 3 ทิ้งไว้เป็น Background เบื้องหลัง

```bash
ollama run llama3
```

- ตั้งค่า Environment Variables: สร้างไฟล์ .env ไว้ในโฟลเดอร์หลักของโปรเจกต์ และเพิ่ม API Key ของ Semantic Scholar ของคุณลงไป สามารถขอ API ได้จาก [Semantic Scholar](https://www.semanticscholar.org/product/api)

```bash
S2_API_KEY=your_semantic_scholar_api_key_here
```

- รันแอปพลิเคชัน: เริ่มการทำงานของเซิร์ฟเวอร์ Streamlit เพื่อเปิดหน้าเว็บ

```bash
streamlit run app_ollama.py
```

# วิธีการใช้งาน
## 1. ค้นหาเปเปอร์
 พิมพ์หัวข้องานวิจัยที่สนใจ เช่น Large Language Models และตั้งค่าจำนวนเปเปอร์เริ่มต้นที่แถบด้านข้าง จากนั้นกด ค้นหาและสร้าง Knowledge Graph

## 2. สำรวจกราฟ
ใช้เมาส์ซูมและลากโหนดต่างๆ สามารถปรับระยะห่างของโหนดได้ที่เมนูฟิสิกส์ด้านล่างของกราฟ

## 3. ระบบแนะนำ 
เลื่อนลงมาที่ส่วน "ระบบแนะนำเปเปอร์" เลือกเปเปอร์ที่คุณสนใจ ระบบจะแสดงเปเปอร์ที่เกี่ยวข้องกันมากที่สุด 5 อันดับแรก โดยคำนวณจากโครงสร้างกราฟ

## 4. ค้นหาเชิงความหมาย 
ใช้เมนู Dropdown ในส่วน "Semantic Search" เพื่อกรองหาเปเปอร์ที่ใช้ Method เฉพาะเจาะจง ร่วมกับ Dataset ที่ต้องการ