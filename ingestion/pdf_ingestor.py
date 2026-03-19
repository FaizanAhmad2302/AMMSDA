import os
import PyPDF2
from config import DATA_DIR

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"📄 Reading PDF: {os.path.basename(pdf_path)} ({total_pages} pages)")
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        print(f"✅ Successfully extracted text from {os.path.basename(pdf_path)}")
        return text
    
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return None

def load_all_pdfs(data_dir=DATA_DIR):
    """Load all PDFs from the data directory."""
    documents = []
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("⚠️ No PDF files found in data/ folder!")
        return documents
    
    print(f"📂 Found {len(pdf_files)} PDF file(s)")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        if text:
            documents.append({
                "filename": pdf_file,
                "content": text
            })
    
    print(f"\n✅ Total documents loaded: {len(documents)}")
    return documents

if __name__ == "__main__":
    docs = load_all_pdfs()
    if docs:
        print(f"\n📝 Preview of first document:")
        print(docs[0]["content"][:500])