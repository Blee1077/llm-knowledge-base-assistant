import os
import gradio as gr
from haystack.nodes import EmbeddingRetriever, PromptModel
from typing import Tuple, Dict
from utils import (
    INITIAL_TEMPLATE,
    load_pickle,
    load_json_file,
    load_json_from_s3,
    create_initial_pipe,
    create_lfqa_section_doc_pipe,
    create_lfqa_whole_doc_pipe,
    run_lfqa
)

DOC_STORE_FOLDER = 'doc_stores'
S3_CONFIG = load_json_file('s3_config.json')
OPENAI_API_KEY = load_json_from_s3(bucket=S3_CONFIG['S3_BUCKET'], key=S3_CONFIG["OPENAI_API_S3_KEY"])['Key']

# Load Document Stores
whole_doc_store = load_pickle(file_name=f'{DOC_STORE_FOLDER}/whole_doc_store.pkl')
split_doc_store = load_pickle(file_name=f'{DOC_STORE_FOLDER}/split_doc_store.pkl')

# Set up prompt model and embedding retrievers using OpenAI models
prompt_model = PromptModel(
    model_name_or_path="gpt-3.5-turbo",
    max_length=750,
    api_key=OPENAI_API_KEY
)
split_doc_retriever = EmbeddingRetriever(
    document_store=split_doc_store,
    batch_size=8,
    use_gpu=False,
    embedding_model="text-embedding-ada-002",
    api_key=OPENAI_API_KEY,
    max_seq_len=8192,
    top_k=3
)
whole_doc_retriever = EmbeddingRetriever(
    document_store=whole_doc_store,
    batch_size=8,
    use_gpu=False,
    embedding_model="text-embedding-ada-002",
    api_key=OPENAI_API_KEY,
    max_seq_len=8192,
    top_k=1
)

# Set up pipelines
initial_pipe = create_initial_pipe(prompt_model, INITIAL_TEMPLATE)
lfqa_section_pipe = create_lfqa_section_doc_pipe(prompt_model, split_doc_retriever)
lfqa_whole_pipe = create_lfqa_whole_doc_pipe(prompt_model, whole_doc_retriever)


def gradio_interface(query: str) -> Tuple[str, Dict[str, str]]:
    """Query the LFQA system and return the answer and relevant documents."""
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            answer, docs = run_lfqa(query, initial_pipe, lfqa_section_pipe, lfqa_whole_pipe)
            formatted_docs = "\n\n".join([f"{key} ({value['name']}):\n{value['content']}" for key, value in docs.items()])
            return answer, formatted_docs
        
        except Exception as e:
            if 'timed out' in str(e):
                retries += 1
                if retries == max_retries:
                    raise Exception(
                        "Error: The server took too long to respond. Please try again later.",
                    )
            else:
                raise Exception("Unexpected error occured.")

# Define example inputs
EXAMPLES = [
    "Summarise the IT acceptable use policy for me",
    "Can I work remotely?",
    "List out the employee benefits",
    "How are performance evaluations conducted?",
    "What is the process for employee recruitment?",
    "What are the weekly employee working hours?",
    "How do employees report business expenses for reimbursement?",
    "Summarise code of conduct policy",
    "List out all the employee leave types available",
    "How does the company handle employee termination?",
    "Can you tell me how to make an apple pie?"
]

APP_DESCRIPTION = """
Welcome to the Dunder Mifflin Paper Company, Inc. Knowledge Base Assistant. 
This tool is powered by OpenAI models for text embedding and answer synthesis and is designed to answer questions related to the company's Human Resource and IT policies. The policy documents are entirely generated by GPT-4.

Simply enter your question in the textbox and the Assistant will provide a synthesized answer along with any relevant documents from the knowledge base. 

The Assistant's responses are based on the following documents in our database:
1. Attendance and remote work policy - Guidelines for employee attendance, remote work eligibility, home office allowances, and communication expectations.
2. Benefits and compensation policy - Outlines payroll procedures, medical benefits, wellness programs, bonuses, allowances, retirement plans, and other benefits to support employee well-being and work-life balance.
3. Code of conduct policy - Standards of behavior emphasizing the company's core values. Details ethical conduct, professionalism, diversity, conflict of interest, confidentiality, health and safety, and anti-harassment policies.
4. Expense policy - Covers authorized business expense reimbursements such as travel, accommodation, meals, and conferences. Outlines the process for obtaining prior approval, submitting expense reports with documentation, and corporate credit card usage guidelines.
5. Health and safety policy - Shared responsibilities among employees to maintain a safe workplace. Includes hazard identification, risk assessment, accident reporting, emergency procedures, and training.
6. Employee leave policy - Covers various types of leave including vacation, sick leave, public holidays, parental, bereavement, jury duty, unpaid leave, work from home, leave donation, professional development, and military leave.
7. Performance evaluation and promotion policy - Annual evaluations, promotion processes, and criteria. Emphasizes fairness, transparency, and employee growth.
8. Recruitment policy - Principles and processes for attracting, selecting, and onboarding employees. Covers candidate selection criteria, recruitment process, employee referral program, and comprehensive onboarding.
9. Employee termination policy - Guidelines for voluntary and involuntary terminations. Covers resignation notice, performance-based termination, layoffs, termination for cause, notice periods, exit interviews, final pay, benefits continuation, and return of company property.
10. Employee working hours policy - Standard working hours, flexible schedules, breaks, and overtime. Promotes a healthy work-life balance and legal compliance. Covers timekeeping, overtime compensation, approval, recordkeeping.
11. IT acceptable use policy - Employee responsibilities for using company technology resources securely and efficiently. Includes authorization, personal use, security measures, password protection, email etiquette, data communication, software installation, remote access, and file storage.

Please note that the Assistant will classify your question and determine whether the answer lies in a whole document, a document's section, or if it cannot be answered using the available policies. If you have any questions not related to these documents, the Assistant may not be able to provide a precise answer. Also, Dunder Mifflin Paper Company, Inc is the fictional company name of the 2005 American Sitcom "The Office".
"""

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.inputs.Textbox(lines=2, label="Enter your query"),
    outputs=[
        gr.outputs.Textbox(label="Answer"),
        gr.outputs.Textbox(label="Relevant Documents")
    ],
    examples=EXAMPLES,
    cache_examples=False,
    allow_flagging='never',
    title="Dunder Mifflin Knowledge Base Assistant using ChatGPT",
    description=APP_DESCRIPTION,
)

# Launch Gradio interface
iface.launch(server_port=int(os.getenv('PORT')), enable_queue=False)