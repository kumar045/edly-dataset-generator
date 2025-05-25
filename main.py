import streamlit as st
from google import genai
from google.genai import types
import json
import pandas as pd
from typing import Dict, Any, List
import io
import base64
import PyPDF2
import docx

# Configure page
st.set_page_config(
    page_title="JSON Dataset Generator",
    page_icon="📊",
    layout="wide"
)

generate_content_config = types.GenerateContentConfig(
    response_mime_type="application/json",
)

def configure_gemini(api_key: str):
    """Configure Gemini API with the provided key"""
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {str(e)}")
        return False

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using PyPDF2"""
    try:
        # Ensure the file pointer is at the beginning for PyPDF2
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        pdf_file.seek(0) # Reset file pointer for potential reuse
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        # Ensure the file pointer is at the beginning for python-docx
        docx_file.seek(0)
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        docx_file.seek(0) # Reset file pointer for potential reuse
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def process_pdf_directly_with_gemini(pdf_file, json_example: str, client: genai.Client, num_samples: int = 1) -> List[Dict]:
    """Process PDF directly with Gemini using the new API"""
    try:
        model = "gemini-1.5-flash" # Using an available and capable model for PDF processing

        # Read PDF file as bytes
        pdf_file.seek(0) # Ensure pointer is at the beginning
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer for potential reuse

        prompt = f"""
        Based on the content of this PDF document and the JSON example format provided, generate {num_samples} similar JSON objects that extract relevant information from the document.

        JSON Example Format:
        {json_example}

        Instructions:
        1. Analyze the PDF document content thoroughly.
        2. Extract relevant information that matches the structure of the JSON example.
        3. Generate {num_samples} JSON objects with the same structure.
        4. Ensure the data is realistic and consistent with the document content.
        5. Return only valid JSON objects, one per line. If multiple JSON objects are generated, ensure each is a separate entry in a JSON array if possible, or one JSON per line.
        6. Do not include any explanations or additional text outside the JSON objects.

        Generated JSON objects:
        """

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_data(
                        mime_type="application/pdf",
                        data=pdf_bytes
                    ),
                    types.Part.from_text(text=prompt)
                ]
            )
        ]

        response = client.models.generate_content(
            model=model,
            contents=contents,
            generation_config=generate_content_config # Use generation_config
        )
        
        json_objects = []
        # The response should be a JSON string, potentially an array of objects or objects separated by newlines
        try:
            # Assuming the model returns a JSON string that might be a list or a single object
            # And the response.text directly contains the JSON string
            parsed_response = json.loads(response.text)
            if isinstance(parsed_response, list):
                json_objects.extend(parsed_response)
            else:
                json_objects.append(parsed_response)
        except json.JSONDecodeError:
            # Fallback for multiple JSON objects separated by newlines (as per prompt instructions)
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('{') or line.startswith('[')):
                    try:
                        json_obj = json.loads(line)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError:
                        # Log or handle lines that are not valid JSON
                        st.warning(f"Skipping invalid JSON line: {line[:100]}...")
                        continue
        
        # Ensure we return only the requested number of samples if more are generated
        return json_objects[:num_samples] if json_objects else []

    except Exception as e:
        st.error(f"Error processing PDF directly with Gemini: {str(e)}")
        st.info("Falling back to text extraction method if direct processing failed earlier in the main flow.")
        return []

def generate_json_from_document(document_text: str, json_example: str, client: genai.Client, num_samples: int = 1) -> List[Dict]:
    """Generate JSON data from document using Gemini API"""
    try:
        model = "gemini-1.5-flash"

        prompt = f"""
        Based on the following document and JSON example format, generate {num_samples} similar JSON objects that extract relevant information from the document.

        Document:
        {document_text}

        JSON Example Format:
        {json_example}

        Instructions:
        1. Extract relevant information from the document that matches the structure of the JSON example.
        2. Generate {num_samples} JSON objects with the same structure.
        3. Ensure the data is realistic and consistent with the document content.
        4. Return only valid JSON objects. If generating multiple, provide them as a JSON array or one JSON object per line.
        5. Do not include any explanations or additional text outside the JSON objects.

        Generated JSON objects:
        """

        request_contents = [types.Content(parts=[types.Part.from_text(text=prompt)])]
        
        response = client.models.generate_content(
            model=model, 
            contents=request_contents, 
            generation_config=generate_content_config # Use generation_config
        )

        json_objects = []
        try:
            parsed_response = json.loads(response.text)
            if isinstance(parsed_response, list):
                json_objects.extend(parsed_response)
            else:
                json_objects.append(parsed_response)
        except json.JSONDecodeError:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('{') or line.startswith('[')):
                    try:
                        json_obj = json.loads(line)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError:
                        st.warning(f"Skipping invalid JSON line: {line[:100]}...")
                        continue
        
        return json_objects[:num_samples] if json_objects else []

    except Exception as e:
        st.error(f"Error generating JSON from document: {str(e)}")
        return []

def generate_similar_json(json_example: str, client: genai.Client, num_samples: int = 1) -> List[Dict]:
    """Generate similar JSON structures based on example"""
    try:
        model = "gemini-1.5-flash" 

        prompt = f"""
        Based on the following JSON example, generate {num_samples} similar JSON objects with the same structure but different realistic data.

        JSON Example:
        {json_example}

        Instructions:
        1. Maintain the exact same JSON structure and field names.
        2. Generate {num_samples} new JSON objects with realistic, varied data.
        3. Ensure data types match the original example.
        4. Make the generated data diverse and realistic.
        5. Return only valid JSON objects. If generating multiple, provide them as a JSON array or one JSON object per line.
        6. Do not include any explanations or additional text outside the JSON objects.

        Generated JSON objects:
        """
        request_contents = [types.Content(parts=[types.Part.from_text(text=prompt)])]
        
        response = client.models.generate_content(
            model=model, 
            contents=request_contents, 
            generation_config=generate_content_config # Use generation_config
        )
        
        json_objects = []
        try:
            parsed_response = json.loads(response.text)
            if isinstance(parsed_response, list):
                json_objects.extend(parsed_response)
            else:
                json_objects.append(parsed_response)
        except json.JSONDecodeError:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('{') or line.startswith('[')):
                    try:
                        json_obj = json.loads(line)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError:
                        st.warning(f"Skipping invalid JSON line: {line[:100]}...")
                        continue
        
        return json_objects[:num_samples] if json_objects else []

    except Exception as e:
        st.error(f"Error generating similar JSON: {str(e)}")
        return []

def validate_json(json_string: str) -> tuple[bool, str]:
    """Validate JSON string"""
    try:
        json.loads(json_string)
        return True, "Valid JSON"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"

def main():
    st.title("📊 JSON Dataset Generator")
    st.markdown("Generate JSON datasets using Google Gemini API with direct PDF processing support")
    
    gemini_client = None # Initialize gemini_client

    # Sidebar for API configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        api_key = st.text_input("Google Gemini API Key", type="password", help="Enter your Google Gemini API key")
        
        if api_key:
            gemini_client = configure_gemini(api_key) # Returns genai.Client object or False
            # configure_gemini function already shows st.error if it fails
            
        st.markdown("---")
        st.markdown("### 📖 How to get API Key")
        st.markdown("""
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create a new API key
        3. Copy and paste it above
        """)
        
        st.markdown("---")
        st.markdown("### 🆕 PDF Processing")
        st.info("PDFs can be processed directly by Gemini for potentially better accuracy!")
    
    if not api_key:
        st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to continue.")
        st.stop() # Stop execution if no API key is provided
    
    # If execution reaches here, api_key was provided.
    # gemini_client is either a genai.Client instance or False.

    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📄 Document + JSON Example", "🔄 JSON Structure Generator", "📊 Dataset Manager"])
    
    with tab1:
        st.header("Generate JSON from Documents")
        st.markdown("Upload a document and provide a JSON example to extract structured data.")
        
        col1, col2 = st.columns([1, 1])
        
        document_text_content = "" # Initialize to hold text from file or text_area
        processing_method = "Direct Gemini Processing (Recommended)" # Default for PDF

        with col1:
            st.subheader("📄 Document Input")
            
            uploaded_file = st.file_uploader("Upload Document", type=['txt', 'pdf', 'docx'])
            
            manual_document_text = st.text_area("Or paste document text here:", height=300, key="manual_text_tab1")
            
            if uploaded_file and uploaded_file.type == "application/pdf":
                processing_method = st.radio(
                    "PDF Processing Method:",
                    ["Direct Gemini Processing (Recommended)", "Text Extraction + Processing"],
                    help="Direct processing can maintain document structure and formatting better. Text extraction is a fallback."
                )
            
            if uploaded_file:
                if uploaded_file.type == "text/plain":
                    document_text_content = str(uploaded_file.read(), "utf-8")
                    uploaded_file.seek(0)
                elif uploaded_file.type == "application/pdf":
                    if processing_method == "Text Extraction + Processing":
                        with st.spinner("Extracting text from PDF..."):
                            document_text_content = extract_text_from_pdf(uploaded_file)
                    else: # Direct Gemini Processing
                        st.info("PDF will be processed directly by Gemini when you click Generate.")
                        # document_text_content remains empty for now, direct processing uses the file
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    with st.spinner("Extracting text from DOCX..."):
                        document_text_content = extract_text_from_docx(uploaded_file)
            elif manual_document_text: # If no file uploaded, use manual text
                 document_text_content = manual_document_text
        
        with col2:
            st.subheader("📋 JSON Example")
            json_example = st.text_area("JSON Example Format:", height=200,
                                        value='{\n  "name": "John Doe",\n  "age": 30,\n  "occupation": "Software Engineer"\n}', key="json_example_tab1")
            
            is_valid, validation_msg = validate_json(json_example)
            if is_valid:
                st.success(f"✅ {validation_msg}")
            else:
                st.error(f"❌ {validation_msg}")
        
        col3, col4 = st.columns([1, 1])
        with col3:
            num_samples = st.number_input("Number of samples to generate:", min_value=1, max_value=20, value=3, key="num_samples_tab1")
        with col4:
            st.write("") # Spacer
            st.write("") # Spacer
            generate_btn1 = st.button("🚀 Generate JSON Dataset", key="gen1", use_container_width=True)
            
        if generate_btn1:
            if not gemini_client:
                st.error("🔴 Gemini client is not configured or initialization failed. Please check your API key in the sidebar and retry.")
                st.stop()

            if not json_example or not is_valid:
                st.error("❌ Please provide a valid JSON example format.")
                st.stop()

            generated_data = []
            
            if uploaded_file and uploaded_file.type == "application/pdf" and processing_method == "Direct Gemini Processing (Recommended)":
                with st.spinner("Processing PDF directly with Gemini..."):
                    generated_data = process_pdf_directly_with_gemini(uploaded_file, json_example, gemini_client, num_samples)
                
                if not generated_data:
                    st.warning("Direct PDF processing returned no data or failed. Attempting fallback with text extraction...")
                    uploaded_file.seek(0) # Reset for fallback extraction
                    fallback_text = extract_text_from_pdf(uploaded_file)
                    if fallback_text:
                        with st.spinner("Fallback: Processing extracted text with Gemini..."):
                            generated_data = generate_json_from_document(fallback_text, json_example, gemini_client, num_samples)
                    else:
                        st.error("Fallback failed: Could not extract text from PDF.")
            elif document_text_content: # Covers TXT, DOCX, PDF (text extraction), and manual paste
                with st.spinner("Generating JSON dataset from text..."):
                    generated_data = generate_json_from_document(document_text_content, json_example, gemini_client, num_samples)
            elif uploaded_file and uploaded_file.type == "application/pdf" and processing_method != "Direct Gemini Processing (Recommended)":
                 st.error("Please ensure PDF text is extracted if 'Text Extraction' method is chosen, or select direct processing.") # Should be covered by document_text_content
            else: # No input provided
                st.error("⚠️ Please upload a document or paste text into the text area.")
                st.stop()
                
            if generated_data:
                st.success(f"✅ Generated {len(generated_data)} JSON objects!")
                
                if 'generated_datasets' not in st.session_state:
                    st.session_state.generated_datasets = []
                
                dataset_name = f"Document_Dataset_{len(st.session_state.generated_datasets) + 1}"
                st.session_state.generated_datasets.append({
                    'name': dataset_name,
                    'data': generated_data,
                    'type': 'document_based'
                })
                
                st.subheader("📊 Generated JSON Objects (Preview)")
                for i, json_obj in enumerate(generated_data[:min(3, len(generated_data))]): # Preview up to 3
                    with st.expander(f"JSON Object {i+1}"):
                        st.json(json_obj)
                if len(generated_data) > 3:
                    st.info(f"Showing 3 of {len(generated_data)} generated objects. View all in 'Dataset Manager'.")
            elif generate_btn1 : # if button was pressed but no data
                st.error("Failed to generate JSON data. Please check your inputs, document content, and API key, then try again.")
    
    with tab2:
        st.header("Generate Similar JSON Structures")
        st.markdown("Provide a JSON example to generate similar structures with different data.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 JSON Example")
            json_template = st.text_area("JSON Template:", height=300,
                                        value='{\n  "product_name": "Laptop",\n  "price": 999.99,\n  "category": "Electronics",\n  "in_stock": true,\n  "specifications": {\n    "brand": "TechCorp",\n    "model": "TC-2024"\n  }\n}', key="json_template_tab2")
            
            is_valid2, validation_msg2 = validate_json(json_template)
            if is_valid2:
                st.success(f"✅ {validation_msg2}")
            else:
                st.error(f"❌ {validation_msg2}")
        
        with col2:
            st.subheader("⚙️ Generation Settings")
            num_samples2 = st.number_input("Number of samples:", min_value=1, max_value=50, value=5, key="samples2")
            
            st.write("") # Spacer
            st.write("") # Spacer
            generate_btn2 = st.button("🔄 Generate Similar JSON", key="gen2", use_container_width=True)
            
        if generate_btn2:
            if not gemini_client:
                st.error("🔴 Gemini client is not configured or initialization failed. Please check your API key in the sidebar and retry.")
                st.stop()

            if not json_template or not is_valid2:
                st.error("❌ Please provide a valid JSON template.")
                st.stop()

            with st.spinner("Generating similar JSON structures..."):
                generated_data2 = generate_similar_json(json_template, gemini_client, num_samples2)
                
            if generated_data2:
                st.success(f"✅ Generated {len(generated_data2)} JSON objects!")
                
                if 'generated_datasets' not in st.session_state:
                    st.session_state.generated_datasets = []
                
                dataset_name = f"Structure_Dataset_{len(st.session_state.generated_datasets) + 1}"
                st.session_state.generated_datasets.append({
                    'name': dataset_name,
                    'data': generated_data2,
                    'type': 'structure_based'
                })
                
                st.subheader("📊 Generated Data Preview")
                for i, json_obj in enumerate(generated_data2[:3]):
                    with st.expander(f"Sample {i+1}"):
                        st.json(json_obj)
                
                if len(generated_data2) > 3:
                    st.info(f"... and {len(generated_data2) - 3} more samples. Check the Dataset Manager tab to view all.")
            elif generate_btn2: # if button was pressed but no data
                 st.error("Failed to generate similar JSON data. Please check your template and API key, then try again.")

    with tab3:
        st.header("Dataset Manager")
        st.markdown("Manage and export your generated datasets.")
        
        if 'generated_datasets' not in st.session_state or not st.session_state.generated_datasets:
            st.info("📭 No datasets generated yet. Use the other tabs to create datasets.")
        else:
            dataset_names = [ds['name'] for ds in st.session_state.generated_datasets]
            selected_dataset_name = st.selectbox("Select Dataset:", dataset_names)
            
            if selected_dataset_name:
                dataset = next((ds for ds in st.session_state.generated_datasets if ds['name'] == selected_dataset_name), None)
                
                if dataset:
                    col1_dm, col2_dm = st.columns([2, 1])
                    
                    with col1_dm:
                        st.subheader(f"📊 {selected_dataset_name}")
                        st.write(f"**Type:** {dataset['type'].replace('_', ' ').title()}")
                        st.write(f"**Records:** {len(dataset['data'])}")
                        
                        with st.expander("View All Data", expanded=False):
                            st.json(dataset['data'])
                    
                    with col2_dm:
                        st.subheader("📤 Export Options")
                        
                        json_data_export = json.dumps(dataset['data'], indent=2)
                        st.download_button(
                            label="💾 Download as JSON",
                            data=json_data_export,
                            file_name=f"{selected_dataset_name}.json",
                            mime="application/json",
                            key=f"dl_json_{selected_dataset_name}"
                        )
                        
                        try:
                            df = pd.json_normalize(dataset['data'])
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📊 Download as CSV",
                                data=csv_data,
                                file_name=f"{selected_dataset_name}.csv",
                                mime="text/csv",
                                key=f"dl_csv_{selected_dataset_name}"
                            )
                        except Exception as e:
                            st.caption(f"CSV export not available for this dataset structure. ({e})")
                        
                        st.markdown("---")
                        if st.button("🗑️ Delete This Dataset", key=f"delete_{selected_dataset_name}", type="primary"):
                            st.session_state.generated_datasets = [
                                ds for ds in st.session_state.generated_datasets 
                                if ds['name'] != selected_dataset_name
                            ]
                            st.success(f"Dataset '{selected_dataset_name}' deleted.")
                            st.rerun()
            
            if st.session_state.generated_datasets: # Show clear all only if there are datasets
                st.markdown("---")
                if st.button("🗑️ Clear All Datasets", type="secondary"):
                    st.session_state.generated_datasets = []
                    st.success("All datasets cleared.")
                    st.rerun()

if __name__ == "__main__":
    main()       
