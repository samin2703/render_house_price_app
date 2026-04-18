# Resume Project Descriptions

---

## 1. Context-Aware RAG AI Tutor · [samin2703/RAG_AI_Tutor](https://github.com/samin2703/RAG_AI_Tutor)

**Built a Retrieval-Augmented Generation (RAG) tutoring web app that answers questions grounded in user-uploaded academic PDFs using FAISS semantic search and an OpenRouter-hosted LLM.**

- Designed an end-to-end RAG pipeline with **LangChain**, **FAISS** vector indexing, and **Sentence-Transformers** embeddings; supports optional **cross-encoder reranking** to improve retrieval precision before answer generation.
- Implemented **level-adaptive prompt engineering** (Beginner / Intermediate / Advanced) so explanations are automatically tailored to learner proficiency while remaining citation-faithful to retrieved source chunks.
- Delivered a polished **Streamlit** UI featuring PDF upload/ingestion, source-cited expandable panels, and persistent JSON-backed conversation history.

---

## 2. Bangla License Plate Detection & Recognition · [samin2703/Bangla_license_plate_project](https://github.com/samin2703/Bangla_license_plate_project)

**Trained and deployed a custom YOLOv8 object-detection model that localises and reads Bangladeshi license plates — including Bangla script characters — directly from uploaded images.**

- Fine-tuned a **YOLOv8** model (`best.pt`) on a **102-class** dataset covering 10 Arabic numerals, Bangla character tokens, and all 64 Bangladeshi district names for complete plate parsing.
- Built a post-processing algorithm using **OpenCV** and **NumPy** that clusters detected bounding boxes into text rows (by y-centroid vs. box height) and reconstructs the full plate string in correct reading order.
- Packaged the pipeline into a **Streamlit** web app where users upload an image, receive an annotated prediction overlay, and see the parsed plate text in a structured multi-line layout.

---

## 3. Customer Support AI Assistant · [samin2703/customer_support_ai](https://github.com/samin2703/customer_support_ai)

**Built a production-ready automated customer-support system combining DistilBERT intent classification, FAISS-backed RAG, and sentiment-driven escalation into a FastAPI service with a Streamlit analytics dashboard.**

- Fine-tuned **DistilBERT** (Hugging Face Transformers) on synthetic data to classify queries into four intents (Billing, Technical Issue, Refund, Complaint), achieving **~92.3 % accuracy** at **~45 ms** average inference latency.
- Implemented a full RAG pipeline with **LangChain**, **FAISS** (384-dim `all-MiniLM-L6-v2` embeddings), and **OpenAI GPT-3.5-turbo**; retrieves top-3 document chunks and auto-escalates to a human agent when classifier confidence < 0.7 or negative sentiment is detected.
- Exposed a **FastAPI** REST API (`/chat`, `/create_ticket`, `/feedback`, `/analytics`) with Swagger UI documentation and built a real-time **Streamlit** monitoring dashboard for query volume, escalation rate, and response-time metrics.

---

## 4. House Price Prediction Web App · [samin2703/render_house_price_app](https://github.com/samin2703/render_house_price_app)

**Developed and deployed a house price prediction web application backed by an XGBoost regression model, served via a FastAPI API and publicly hosted on Render.**

- Trained an **XGBoost** regression model on the Boston Housing dataset, selecting the six highest-impact features (RM, LSTAT, DIS, PTRATIO, TAX, NOX) to keep the deployed model lightweight.
- Built a **FastAPI** backend with **Jinja2**-templated HTML frontend and a typed **Pydantic** request schema; the `/predict` endpoint returns real-time price estimates from user-supplied feature values.
- Deployed the full application to **Render** for public access, with `uvicorn` as the ASGI server and `pandas` / `XGBoost` handling inference on each request.
