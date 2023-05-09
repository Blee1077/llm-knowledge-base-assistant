import pickle
import json
import boto3
from typing import Tuple, Dict
from haystack.pipelines import Pipeline
from haystack.nodes import (
    EmbeddingRetriever, PromptNode, PromptModel, PromptTemplate, AnswerParser
)
S3_RESOURCE = boto3.resource('s3')


POLICY_NAMES = [
        "Attendance and remote work policy: Guidelines for employee attendance, remote work eligibility, home office allowances, and communication expectations at Dunder Mifflin.",
        "Benefits and compensation policy: Outlines payroll procedures, medical benefits, wellness programs, bonuses, allowances, retirement plans, and other benefits to support employee well-being and work-life balance.",
        "Code of conduct policy: Outlines standards of behavior, emphasizing core values of the company. Employees must adhere to ethical conduct, professionalism, diversity, conflict of interest, confidentiality, health and safety, and anti-harassment policies.",
        "Expense policy: Covers reimbursement for authorized business expenses, such as travel, accommodation, meals, and conferences. Employees must obtain prior approval, submit expense reports with documentation, and adhere to guidelines for corporate credit card usage.",
        "Health and safety policy: Outlines shared responsibilities among employees to maintain a safe workplace. Includes hazard identification, risk assessment, accident reporting, emergency procedures, and training.",
        "Employee leave policy: Covers vacation, sick leave, public holidays, parental, bereavement, jury duty, unpaid leave, work from home, leave donation, professional development, and military leave.",
        "Performance evaluation and promotion policy: outlines annual evaluations, promotion processes, and criteria. The policy emphasizes fairness, transparency, employee growth.",
        "Recruitment policy: outlines principles and processes for attracting, selecting, and onboarding employees. The policy covers candidate selection criteria, recruitment process, employee referral program, and comprehensive onboarding.",
        "Employee termination policy: Outlines guidelines for voluntary and involuntary terminations, ensuring fairness and legal compliance. It covers resignation notice, performance-based termination, layoffs, termination for cause, notice periods, exit interviews, final pay, benefits continuation, and return of company property.",
        "Employee working hours policy: Outlines guidelines for standard working hours, flexible schedules, breaks, and overtime, promoting a healthy work-life balance and legal compliance. Covers timekeeping, overtime compensation, approval, recordkeeping.",
        "IT acceptable use policy: Outlines employee responsibilities for using company technology resources securely and efficiently, including authorization, personal use, security measures, password protection, email etiquette, data communication, software installation, remote access, and file storage."
    ]

INITIAL_TEMPLATE = PromptTemplate(
    name="initial-prompt",
    prompt_text=f"You are a knowledge base assistant for Dunder Mifflin Paper Company, Inc. and you have access to the following HR and IT policies: {' ,'.join(POLICY_NAMES)}"
    "Your task is to Determine if user query needs entire an policy (answer 'Whole'), a section (answer 'Section'), or cannot be answered using policies (answer 'N/A')."
    "Only use 'Whole', 'Section', or 'N/A' as response with no additional text."
    "User query: {query}; Answer:",
    output_parser=AnswerParser(),
)

def load_pickle(file_name: str):
    """Loads in a pickle file given its filepath
    
    Args:
        file_name (str): Path of pickle file
    
    Returns:
        unpickled object
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def load_json_file(file_path) -> dict:
    """Load a local JSON file into a dictionary in Python.

    Parameters:
        file_path (str): The path to the JSON file to load.

    Returns:
        dict: A dictionary containing the contents of the JSON file.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the file contents cannot be parsed as valid JSON.
    """
    try:
        # Open the JSON file and load its contents into a Python dictionary
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        raise ValueError(f"Invalid JSON in file: {file_path}")


def load_json_from_s3(bucket: str, key: str) -> dict:
    """Loads a JSON file from S3 bucket.
    
    Args:
        bucket (str): S3 bucket containing JSON file
        key (str): Path within bucket of JSON file
        
    Returns:
        dict
    """
    content_object = S3_RESOURCE.Object(bucket, key)
    file_content = content_object.get()["Body"].read().decode("utf-8")
    return json.loads(file_content)


def create_initial_pipe(prompt_model: PromptModel, initial_template: str) -> Pipeline:
    """Create the initial pipeline for query classification.

    Args:
        prompt_model (PromptModel): The prompt model to be used in the PromptNode.
        initial_template (str): The template for the initial prompt.

    Returns:
        Pipeline: The initial pipeline with a single PromptNode.
    """
    initial_node = PromptNode(prompt_model, default_prompt_template=initial_template)
    initial_pipe = Pipeline()
    initial_pipe.add_node(component=initial_node, name="initial_prompt_node", inputs=["Query"])
    
    return initial_pipe


def create_lfqa_section_doc_pipe(prompt_model: PromptModel, split_doc_retriever: EmbeddingRetriever) -> Pipeline:
    """Create the LFQA pipeline for answering questions using document sections.

    Args:
        prompt_model (PromptModel): The prompt model to be used in the PromptNode.
        split_doc_retriever (EmbeddingRetriever): The retriever for obtaining document sections.

    Returns:
        Pipeline: The LFQA pipeline with a Split Document Retriever and a PromptNode.
    """
    lfqa_section_node = PromptNode(prompt_model, default_prompt_template="question-answering")
    lfqa_section_pipe = Pipeline()
    lfqa_section_pipe.add_node(component=split_doc_retriever, name="split_doc_retriever", inputs=["Query"])
    lfqa_section_pipe.add_node(component=lfqa_section_node, name="lfqa_prompt_node", inputs=["split_doc_retriever"])
    
    return lfqa_section_pipe


def create_lfqa_whole_doc_pipe(prompt_model: PromptModel, whole_doc_retriever: EmbeddingRetriever) -> Pipeline:
    """Create the LFQA pipeline for answering questions using whole documents.

    Args:
        prompt_model (PromptModel): The prompt model to be used in the PromptNode.
        whole_doc_retriever (EmbeddingRetriever): The retriever for obtaining whole documents.

    Returns:
        Pipeline: The LFQA pipeline with a Whole Document Retriever and a PromptNode.
    """
    lfqa_whole_node = PromptNode(prompt_model, default_prompt_template="question-answering")
    lfqa_whole_pipe = Pipeline()
    lfqa_whole_pipe.add_node(component=whole_doc_retriever, name="whole_doc_retriever", inputs=["Query"])
    lfqa_whole_pipe.add_node(component=lfqa_whole_node, name="summariser_prompt_node", inputs=["whole_doc_retriever"])
    
    return lfqa_whole_pipe


def extract_outputs(output: Dict) -> Tuple[str, Dict[str, str]]:
    """Extract the answer and documents from the pipeline output.

    Args:
        output (dict): The output dictionary from the pipeline.

    Returns:
        tuple: A tuple containing the answer (str) and a dictionary of documents.
    """
    answer = output['answers'][0].answer
    docs = {f"Document {idx}": {'content': doc.content, 'name': doc.meta['name']} for idx, doc in enumerate(output['documents'], start=1)}
    return answer, docs


def process_query(query: str, pipeline: Pipeline) -> Tuple[str, Dict[str, str]]:
    """Processes a user's query using the specified pipeline and extracts the answer and related documents.

    Args:
        query (str): The user's query.
        pipeline (Pipeline): The Haystack pipeline used to process the query and generate an answer.

    Returns:
        tuple: A tuple containing the answer (str) and a dictionary of related documents.
    """
    answer_output = pipeline.run(query=query)
    return extract_outputs(answer_output)


def run_lfqa(query: str, initial_pipe: Pipeline, lfqa_section_pipe: Pipeline, lfqa_whole_pipe: Pipeline) -> Tuple[str, Dict[str, str]]:
    """Runs the initial pipeline to classify the query, then runs the appropriate LFQA
    pipeline based on the query classification result.

    Args:
        query (str): The user's query.
        initial_pipe (Pipeline): The initial pipeline for query classification.
        lfqa_section_pipe (Pipeline): The LFQA pipeline for answering questions using document sections.
        lfqa_whole_pipe (Pipeline): The LFQA pipeline for answering questions using whole documents.

    Returns:
        tuple: A tuple containing the answer (str) and a dictionary of documents.
    """
    initial_output = initial_pipe.run(query=query)
    query_class = initial_output['answers'][0].answer
    
    if 'section' in query_class.lower():
        return process_query(query, lfqa_section_pipe)
    elif 'whole' in query_class.lower():
        return process_query(query, lfqa_whole_pipe)
    else:
        return "Answer cannot be provided with internal knowledge base.", {}