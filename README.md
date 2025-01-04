
# AI-PDF-QA-with-LangChain

AI-PDF-QA-with-LangChain is a robust application designed to enable question-answering over PDF documents using the power of AI. It combines LangChain's document processing capabilities, Pinecone's vector store for efficient retrieval, and OpenAI's language models to provide precise answers to user queries.

## Features
- **PDF Upload**: Upload and process PDF documents seamlessly.
- **Text Chunking**: Split documents into manageable chunks for effective processing.
- **Vector Search**: Use Pinecone's vector store for efficient information retrieval.
- **Conversational AI**: Leverage OpenAI's language models for accurate answers.
- **History Tracking**: Maintain a history of user queries and responses.

## Technologies Used
- **Streamlit**: Frontend framework for building interactive web applications.
- **LangChain**: Document loaders and chains for conversational AI.
- **Pinecone**: Vector database for semantic search.
- **OpenAI**: Language models for answering user queries.
- **Python**: Backend implementation and application logic.

## Installation

### Prerequisites
- Python 3.7+
- API keys for OpenAI and Pinecone

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/NoorMuhammad106/ai-pdf-qa-with-langchain.git
   cd ai-pdf-qa-with-langchain
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     PINECONE_ENVIRONMENT=your_pinecone_environment
     PINECONE_INDEX_NAME=your_index_name
     ```

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Open the app in your browser (usually at `http://localhost:8501`).
3. Upload a PDF document.
4. Ask questions related to the document.



## License
This project is licensed under the MIT License. 

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

## Acknowledgments
- [LangChain](https://www.langchain.com/)
- [Pinecone](https://www.pinecone.io/)
- [OpenAI](https://openai.com/)
- [Streamlit](https://streamlit.io/)

---

Feel free to contact us for any questions or support!
