import os
import PyPDF2
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def pinecone_setup():
    """Setup Pinecone client and ensure the index exists."""
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        embedding_dimension = 1024

        pc = Pinecone(api_key=pinecone_api_key)
        index_names = [index["name"] for index in pc.list_indexes()]
        print("Available indexes:", index_names)

        if index_name not in index_names:
            print(f"Creating index '{index_name}' with dimension {embedding_dimension}.")
            pc.create_index(
                name=index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            print(f"Using existing index '{index_name}'.")

        return pc, pc.Index(name=index_name)
    except Exception as err:
        print("Error while setting up Pinecone:", err)

def convert_chunks_into_records(pc, chunks):
    """Convert text chunks into embedding records."""
    try:
        if not chunks:
            print("No chunks provided.")
            return None

        records = []
        count = 1
        chunk_size = 96

        while chunks:
            splitted_chunk = chunks[:chunk_size]
            chunks = chunks[chunk_size:]

            # Create embeddings using the Pinecone inference API
            embeddings = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=splitted_chunk,
                parameters={"input_type": "passage", "truncate": "END"}
            )

            for text, embedding in zip(splitted_chunk, embeddings):
                records.append({
                    "id": str(count),
                    "values": embedding['values'],
                    "metadata": {'text': text}
                })
                count += 1

        return records
    except Exception as err:
        print("Error while creating embeddings from chunks:", err)

def extract_text_from_pdf_to_chunks(file_path):
    """Extract text from a PDF and split it into chunks."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text_chunks = []

            for page in reader.pages:
                page_text = page.extract_text()
                text_chunks.append(page_text)

            return text_chunks
    except Exception as err:
        print("Error while extracting text from PDF:", err)

def push_the_embeddings_into_pinecone(index, records):
    """Push embeddings into Pinecone in batches."""
    try:
        batch_size = 100
        total_records = len(records)
        print(f"Total records to insert: {total_records}")

        for i in range(0, total_records, batch_size):
            batch = records[i:i + batch_size]
            
            index.upsert(vectors=batch)
            print(f"Batch {i // batch_size + 1}: Successfully inserted {len(batch)} embeddings into Pinecone.")

        print("Successfully inserted all embeddings into Pinecone.")
    except Exception as err:
        print("Error while pushing embeddings into Pinecone:", err)

def main():
    """Main function to process the PDF and store embeddings."""
    pc, index = pinecone_setup()
    print("Pinecone setup complete.")

    pdf_path = "FILE_NAME"
    text_chunks = extract_text_from_pdf_to_chunks(pdf_path)

    if text_chunks:
        records = convert_chunks_into_records(pc, text_chunks)
        if records:
            push_the_embeddings_into_pinecone(index, records)

if __name__ == "__main__":
    main()