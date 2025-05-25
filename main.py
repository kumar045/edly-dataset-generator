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
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {str(e)}")
        return False

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using PyPDF2"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def process_pdf_directly_with_gemini(pdf_file, json_example: str, api_key: str, num_samples: int = 1) -> List[Dict]:
    """Process PDF directly with Gemini 2.5 Pro using the new API"""
    try:
        # Use the new Gemini client for direct PDF processing
        client = genai.Client(api_key=api_key)
        model = "gemini-2.0-flash"  # Using available model
        
        # Read PDF file as bytes
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer for potential reuse
        
        prompt = f"""
        Based on the content of this PDF document and the JSON example format provided, generate {num_samples} similar JSON objects that extract relevant information from the document.

        JSON Example Format:
        {json_example}

        Instructions:
        1. Analyze the PDF document content thoroughly
        2. Extract relevant information that matches the structure of the JSON example
        3. Generate {num_samples} JSON objects with the same structure
        4. Ensure the data is realistic and consistent with the document content
        5. Return only valid JSON objects, one per line
        6. Do not include any explanations or additional text

        Generated JSON objects:
        """
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
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
            config=generate_content_config
        )
        
        # Parse the response to extract JSON objects
        json_objects = []
        lines = response.text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('{') or line.startswith('[')):
                try:
                    json_obj = json.loads(line)
                    json_objects.append(json_obj)
                except json.JSONDecodeError:
                    continue
        
        return json_objects
        
    except Exception as e:
        st.error(f"Error processing PDF directly with Gemini: {str(e)}")
        # Fallback to text extraction method
        st.info("Falling back to text extraction method...")
        return []

def generate_json_from_document(document_text: str, json_example: str, api_key: str, num_samples: int = 1) -> List[Dict]:
    """Generate JSON data from document using Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        Based on the following document and JSON example format, generate {num_samples} similar JSON objects that extract relevant information from the document.

        Document:
        {document_text}

        JSON Example Format:
        {json_example}

        Instructions:
        1. Extract relevant information from the document that matches the structure of the JSON example
        2. Generate {num_samples} JSON objects with the same structure
        3. Ensure the data is realistic and consistent with the document content
        4. Return only valid JSON objects, one per line
        5. Do not include any explanations or additional text

        Generated JSON objects:
        """
        
        response = model.generate_content(prompt,generation_config=generate_content_config)
        
        # Parse the response to extract JSON objects
        json_objects = []
        lines = response.text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('{') or line.startswith('[')):
                try:
                    json_obj = json.loads(line)
                    json_objects.append(json_obj)
                except json.JSONDecodeError:
                    continue
        
        return json_objects
        
    except Exception as e:
        st.error(f"Error generating JSON from document: {str(e)}")
        return []

def generate_similar_json(json_example: str, api_key: str, num_samples: int = 1) -> List[Dict]:
    """Generate similar JSON structures based on example"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        Based on the following JSON example, generate {num_samples} similar JSON objects with the same structure but different realistic data.

        JSON Example:
        {json_example}

        Instructions:
        1. Maintain the exact same JSON structure and field names
        2. Generate {num_samples} new JSON objects with realistic, varied data
        3. Ensure data types match the original example
        4. Make the generated data diverse and realistic
        5. Return only valid JSON objects, one per line
        6. Do not include any explanations or additional text

        Generated JSON objects:
        """
        
        response = model.generate_content(prompt,generation_config=generate_content_config)
        
        # Parse the response to extract JSON objects
        json_objects = []
        lines = response.text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('{') or line.startswith('[')):
                try:
                    json_obj = json.loads(line)
                    json_objects.append(json_obj)
                except json.JSONDecodeError:
                    continue
        
        return json_objects
        
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
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        api_key = st.text_input("Google Gemini API Key", type="password", help="Enter your Google Gemini API key")
        
        if api_key:
            if configure_gemini(api_key):
                st.success("✅ API Key configured successfully!")
            else:
                st.error("❌ Failed to configure API Key")
        
        st.markdown("---")
        st.markdown("### 📖 How to get API Key")
        st.markdown("""
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create a new API key
        3. Copy and paste it above
        """)
        
        st.markdown("---")
        st.markdown("### 🆕 PDF Processing")
        st.info("PDFs are now processed directly by Gemini for better accuracy!")
    
    if not api_key:
        st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to continue.")
        return
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📄 Document + JSON Example", "🔄 JSON Structure Generator", "📊 Dataset Manager"])
    
    with tab1:
        st.header("Generate JSON from Documents")
        st.markdown("Upload a document and provide a JSON example to extract structured data.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Document Input")
            
            # File upload
            uploaded_file = st.file_uploader("Upload Document", type=['txt', 'pdf', 'docx'])
            
            # Text area for manual input
            document_text = st.text_area("Or paste document text here:", height=300)
            
            # Processing method selection for PDFs
            if uploaded_file and uploaded_file.type == "application/pdf":
                processing_method = st.radio(
                    "PDF Processing Method:",
                    ["Direct Gemini Processing (Recommended)", "Text Extraction + Processing"],
                    help="Direct processing maintains document structure and formatting better"
                )
            
            # Handle file processing
            if uploaded_file:
                if uploaded_file.type == "text/plain":
                    document_text = str(uploaded_file.read(), "utf-8")
                    uploaded_file.seek(0)  # Reset for potential reuse
                elif uploaded_file.type == "application/pdf":
                    if not hasattr(st.session_state, 'pdf_processed') or st.session_state.pdf_processed != uploaded_file.name:
                        if processing_method == "Text Extraction + Processing":
                            with st.spinner("Extracting text from PDF..."):
                                document_text = extract_text_from_pdf(uploaded_file)
                                st.session_state.pdf_processed = uploaded_file.name
                        else:
                            st.info("PDF will be processed directly by Gemini when you click Generate.")
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    with st.spinner("Extracting text from DOCX..."):
                        document_text = extract_text_from_docx(uploaded_file)
        
        with col2:
            st.subheader("📋 JSON Example")
            json_example = st.text_area("JSON Example Format:", height=200, 
                                      value='{\n  "name": "John Doe",\n  "age": 30,\n  "occupation": "Software Engineer"\n}')
            
            # Validate JSON
            is_valid, validation_msg = validate_json(json_example)
            if is_valid:
                st.success(f"✅ {validation_msg}")
            else:
                st.error(f"❌ {validation_msg}")
        
        # Generation controls
        col3, col4 = st.columns([1, 1])
        with col3:
            num_samples = st.number_input("Number of samples to generate:", min_value=1, max_value=20, value=3)
        with col4:
            generate_btn1 = st.button("🚀 Generate JSON Dataset", key="gen1")
        
        if generate_btn1 and json_example and is_valid:
            generated_data = []
            
            # Handle different processing methods
            if uploaded_file and uploaded_file.type == "application/pdf" and processing_method == "Direct Gemini Processing (Recommended)":
                with st.spinner("Processing PDF directly with Gemini..."):
                    generated_data = process_pdf_directly_with_gemini(uploaded_file, json_example, api_key, num_samples)
                    
                    # Fallback to text extraction if direct processing fails
                    if not generated_data:
                        st.warning("Direct PDF processing failed, trying text extraction method...")
                        with st.spinner("Extracting text and processing..."):
                            document_text = extract_text_from_pdf(uploaded_file)
                            if document_text:
                                generated_data = generate_json_from_document(document_text, json_example, api_key, num_samples)
            
            elif document_text:
                with st.spinner("Generating JSON dataset..."):
                    generated_data = generate_json_from_document(document_text, json_example, api_key, num_samples)
            
            else:
                st.error("Please provide a document or text input.")
                return
            
            if generated_data:
                st.success(f"✅ Generated {len(generated_data)} JSON objects!")
                
                # Store in session state
                if 'generated_datasets' not in st.session_state:
                    st.session_state.generated_datasets = []
                
                dataset_name = f"Document_Dataset_{len(st.session_state.generated_datasets) + 1}"
                st.session_state.generated_datasets.append({
                    'name': dataset_name,
                    'data': generated_data,
                    'type': 'document_based'
                })
                
                # Display results
                st.subheader("📊 Generated JSON Objects")
                for i, json_obj in enumerate(generated_data):
                    with st.expander(f"JSON Object {i+1}"):
                        st.json(json_obj)
            else:
                st.error("Failed to generate JSON data. Please check your inputs and try again.")
    
    with tab2:
        st.header("Generate Similar JSON Structures")
        st.markdown("Provide a JSON example to generate similar structures with different data.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 JSON Example")
            json_template = st.text_area("JSON Template:", height=300,
                                       value='{\n  "product_name": "Laptop",\n  "price": 999.99,\n  "category": "Electronics",\n  "in_stock": true,\n  "specifications": {\n    "brand": "TechCorp",\n    "model": "TC-2024"\n  }\n}')
            
            # Validate JSON
            is_valid2, validation_msg2 = validate_json(json_template)
            if is_valid2:
                st.success(f"✅ {validation_msg2}")
            else:
                st.error(f"❌ {validation_msg2}")
        
        with col2:
            st.subheader("⚙️ Generation Settings")
            num_samples2 = st.number_input("Number of samples:", min_value=1, max_value=50, value=5, key="samples2")
            
            generate_btn2 = st.button("🔄 Generate Similar JSON", key="gen2")
            
            if generate_btn2 and json_template and is_valid2:
                with st.spinner("Generating similar JSON structures..."):
                    generated_data2 = generate_similar_json(json_template, api_key, num_samples2)
                    
                    if generated_data2:
                        st.success(f"✅ Generated {len(generated_data2)} JSON objects!")
                        
                        # Store in session state
                        if 'generated_datasets' not in st.session_state:
                            st.session_state.generated_datasets = []
                        
                        dataset_name = f"Structure_Dataset_{len(st.session_state.generated_datasets) + 1}"
                        st.session_state.generated_datasets.append({
                            'name': dataset_name,
                            'data': generated_data2,
                            'type': 'structure_based'
                        })
                        
                        # Display preview
                        st.subheader("📊 Generated Data Preview")
                        for i, json_obj in enumerate(generated_data2[:3]):  # Show first 3
                            with st.expander(f"Sample {i+1}"):
                                st.json(json_obj)
                        
                        if len(generated_data2) > 3:
                            st.info(f"... and {len(generated_data2) - 3} more samples. Check the Dataset Manager tab to view all.")
    
    with tab3:
        st.header("Dataset Manager")
        st.markdown("Manage and export your generated datasets.")
        
        if 'generated_datasets' not in st.session_state or not st.session_state.generated_datasets:
            st.info("📭 No datasets generated yet. Use the other tabs to create datasets.")
            return
        
        # Dataset selection
        dataset_names = [ds['name'] for ds in st.session_state.generated_datasets]
        selected_dataset = st.selectbox("Select Dataset:", dataset_names)
        
        if selected_dataset:
            # Find selected dataset
            dataset = next(ds for ds in st.session_state.generated_datasets if ds['name'] == selected_dataset)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📊 {selected_dataset}")
                st.write(f"**Type:** {dataset['type'].replace('_', ' ').title()}")
                st.write(f"**Records:** {len(dataset['data'])}")
                
                # Display data
                with st.expander("View All Data", expanded=True):
                    for i, record in enumerate(dataset['data']):
                        st.json(record)
            
            with col2:
                st.subheader("📤 Export Options")
                
                # JSON export
                json_data = json.dumps(dataset['data'], indent=2)
                st.download_button(
                    label="💾 Download as JSON",
                    data=json_data,
                    file_name=f"{selected_dataset}.json",
                    mime="application/json"
                )
                
                # CSV export (if possible)
                try:
                    df = pd.json_normalize(dataset['data'])
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download as CSV",
                        data=csv_data,
                        file_name=f"{selected_dataset}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.info("CSV export not available for complex nested JSON structures.")
                
                # Delete dataset
                if st.button("🗑️ Delete Dataset", key=f"delete_{selected_dataset}"):
                    st.session_state.generated_datasets = [
                        ds for ds in st.session_state.generated_datasets 
                        if ds['name'] != selected_dataset
                    ]
                    st.rerun()
        
        # Clear all datasets
        if st.session_state.generated_datasets:
            st.markdown("---")
            if st.button("🗑️ Clear All Datasets"):
                st.session_state.generated_datasets = []
                st.rerun()

if __name__ == "__main__":
    main()
