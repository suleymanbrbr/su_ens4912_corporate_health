# SUT Agentic RAG — Technical Features Overview

This document provides a detailed technical breakdown of the SUT Corporate Health system's capabilities, from its agentic reasoning core to its interactive knowledge representation.

## 🧠 1. Agentic RAG Engine (Core)
The system has transitioned from a basic "Search & Answer" pipeline to an **autonomous agent** built on a ReAct (Reasoning + Acting) loop using Gemini 2.0 Flash.

- **Chain-of-Thought (CoT)**: The agent explicitly plans its investigation strategy using a `Thought:` block before calling any tools. It analyzes whether a single search is sufficient or if multi-hop investigation across SUT articles is required.
- **Specialized Tool-Calling Loop**:
  - `search_sut_chunks`: Semantic vector search using `pgvector` (cosine similarity) for high-level retrieval.
  - `search_sut_fulltext`: BM25 keyword search using PostgreSQL `tsvector('turkish')` for finding specific codes (ICD-10, ATC).
  - `lookup_kg_entity`: High-precision node lookup in the Knowledge Graph; returns the node's attributes and all 1-hop neighbors.
  - `explore_kg_path`: Graphs-based BFS traversal to verify connectivity between disparate SUT rules (e.g., finding the prescribed specialist for a restricted drug).
  - `read_user_report`: Contextual analysis of user-uploaded text, allowing the agent to "see" the patient's condition.
  - `calculate`: Numeric engine for dosage thresholds, age limits, and cost percentage calculations.
- **Contextual Awareness**: Dynamic persona shifting (Patient, Doctor, Admin) that adjusts terminology and level of technical detail in the output.
- **Self-Correction (Critic Loop)**: A secondary LLM pass audits the final response, specifically checking if `[Madde X.X.X]` citations exist in the retrieved text.

## 🕸️ 2. Knowledge Graph Infrastructure (Enhanced)
The KG serves as a multi-relational database of SUT rules, significantly reducing the "discovery hop" problem.

- **Schema-First Extraction**: Gemini-powered extraction with forced JSON schemas (Pydantic) to ensure every node has a `Type` and every edge has a `Relation`.
- **Node Taxonomy**:
  - `DRUG / ETIKEN MADDE`: Linked with ATC codes and dosage rules.
  - `DIAGNOSIS / TEŞHİS`: Linked with ICD-10 identifiers.
  - `RULE / SUT KURALI`: Canonical representation of specific articles.
  - `SPECIALIST / UZMAN`: Identifies required approval boards or physician types.
- **Relational Logic**: 
  - `TREATS`: Connects Drugs to Diagnoses.
  - `REQUIRES_CONDITION`: Connects Rules to specific patient criteria (e.g., fail-first prerequisites).
  - `HAS_LIMIT`: Connects Rules to duration or quantity constraints.
- **Search Optimization**: 
  - **Hybrid Search**: Combines string matching for exact names with `pgvector` embeddings for semantic similarity.
  - **Performance**: Uses `ivfflat` indexing for sub-second retrieval over thousands of nodes.

## 📄 3. Document Analysis & Persistence
Beyond searching existing rules, the system analyzes user-provided context.

- **PDF Extraction**: Backend integration with `pypdf` for reliable text extraction from digital prescriptions and medical reports.
- **Relational Chat History**: Conversation history is stored in Postgres with `file_metadata` support, allowing the system to remember and re-reference uploaded documents across sessions.
- **Session Context Management**: Automatic summarization of long chat histories (when turns > 10) to maintain high-quality context without hitting LLM token limits.

## 🖼️ 4. Frontend & User Experience
A modern, responsive React interface designed for both patients and healthcare administrators.

- **Real-Time Streaming**: character-by-character typewriter rendering via Server-Sent Events (SSE).
- **Interactive Graph Viewer**: Uses `react-force-graph-2d` for a high-performance visualization.
  - **Multi-Node Selection**: Supports Shift+Click for combined relationship analysis.
  - **Navigation Stability**: Sidebar includes `scrollbar-gutter: stable` and flexible height constraints for robust scrolling across all viewport sizes.
- **Agent Trace Timeline**: A collapsible transparency panel that shows users the agent's exact tool calls and intermediate thoughts.
- **Role-Based Personas**: Dynamic system prompt switching (Citizen, Doctor, Admin) to tailor terminology and technical depth.

## ⚡ 5. Performance & Scalability
- **Persistent Model Cache**: Dedicated Docker volume for HuggingFace models (`paraphrase-multilingual-MiniLM-L12-v2` and rerankers), ensuring near-instant startups.
- **Offline Mode**: Configured libraries to prioritize local files, eliminating network-induced query latency.
- **pgvector Optimization**: Indexed vector searches using `ivfflat` for sub-second retrieval over thousands of article chunks.
