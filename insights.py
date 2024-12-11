import logging
import os
import random
from collections import defaultdict
import json

from bson import ObjectId
from langchain.chat_models import AzureChatOpenAI
from src.GenerativeAI.DBOperation.DatasetReadWrite import datasetReadWrite
# Set environment variables (ensure these are valid for your use case)
os.environ["OPENAI_API_KEY"] = "0b1d7d099829418fb1293b97f2ae9c23"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
mongo_obj = datasetReadWrite()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_chat_model(temperature=0.0):
    """ Initializes the chat model for Azure OpenAI GPT-3.5/4.0 """
    return AzureChatOpenAI(
        deployment_name="gpt35exploration",
        azure_endpoint="https://exploregenaiworkspace.openai.azure.com",
        temperature=temperature
    )


def llm_BERT_analysis(chat_model, content):
    """ Processes the failed test case data and provides insights in JSON format """

    template = (
        "You are an intelligent agent capable of gathering insights from a collection of failed augmentations of a call conversation. "
        "You are analyzing failed test cases for a customer service system that interacts with travel agency customers. "
        "Each failure is associated with a specific augmentation type applied to the call conversations and a metric that failed, along with the reason for the failure. "
        "Your goal is to analyze this data and provide actionable insights for improving the test data, with specific recommendations for augmentations to target system performance gaps.\n\n"

        "Inputs: Failures in Test Data\n"
        "Each test case includes:\n"
        "- A call conversation with the customer.\n"
        "- The augmentation type that was applied to the conversation (e.g., formal vs informal language, multilingual conversation, background noise).\n"
        "- The metric that failed (e.g., coverage, relevancy, completeness).\n"
        "- The reason for failure (e.g., missing information, incorrect tone, improper language usage, etc.).\n"

        "Consider the examples only as a reference.\n"
        "YOUR TASK: Analyze the provided failures.\n"
        "For each failure entry, assess the augmentation type and the failure reason.\n"
        "Determine the potential gaps in the test data that may have caused the failure of the metric.\n"
        "Identify common patterns or recurring issues across multiple test cases (e.g., certain augmentation types leading to specific metric failures).\n"
        "Provide actionable insights: Based on the gaps identified, suggest specific augmentations or new combinations to improve system performance.\n"
        "Suggest additional data scenarios or augmentations that were not covered in the test data but are likely to improve system robustness.\n"
        "Ensure the insights are specific and actionable, highlighting both the failures and potential fixes.\n"
        "Prioritize the augmentations: Prioritize augmentations that directly address the failures encountered in the analysis.\n"
        "For each suggested augmentation, provide atlest 5 example scenarios where the augmentation would address a failure and improve system performance.\n"
        "Consider factors like real-world applicability, potential for system improvement, and ease of implementation when suggesting the priority.\n"
        "collection of failed augmentations: '''{}'''\n\n"
        
        "provide the output in json format "
    ).format(content)

    try:
        # Get the response from the model
        response1 = chat_model.call_as_llm(template)
        print(response1)
        response2= json.dumps(response1,indent=2)
        print(response2)
        response=json.loads(response2)

        print(response)
        return response

        # Split the response into individual insights (each insight is separated by a newline)
        insights = response.split('\n\n')

        # Loop through each insight and parse the relevant fields


    except Exception as e:
        # Handle errors if any occur
        result = {
            "status": "error",
            "message": str(e)
        }
        return json.dumps(result, indent=4)


def analyze_insights(content, temperature=0.0):
    """ Analyze the provided content using the LLM for BERT-like analysis """
    chat_model = initialize_chat_model(temperature)
    llm_judgment = llm_BERT_analysis(chat_model, content)
    return llm_judgment
    
def calculate_failure_distribution(content):
    # Step 1: Initialize dictionaries to track failures by metric and augmentation type
    metric_augmentation_count = defaultdict(lambda: defaultdict(int))
    metric_count = defaultdict(int)
    total_failures = 0  # Track the total number of failures across all metrics

    # Step 2: Populate the dictionaries with data
    for failure in content:
        metric = failure["failed_metric"]
        augmentation_type = failure["augmentation_type"]

        # Count failures by metric and augmentation type
        metric_augmentation_count[metric][augmentation_type] += 1

        # Count total failures by metric
        metric_count[metric] += 1
        total_failures += 1  # Increment total failures count

    # Step 3: Calculate the percentage distribution of failures for each metric based on augmentation type
    distribution = {}
    for metric, augmentation_dict in metric_augmentation_count.items():
        total_metric_failures = metric_count[metric]
        distribution[metric] = {}

        for augmentation_type, count in augmentation_dict.items():
            percentage = (count / total_metric_failures) * 100
            distribution[metric][augmentation_type] = percentage

    # Step 4: Calculate the percentage for each metric
    metric_percentage = {}
    for metric, count in metric_count.items():
        metric_percentage[metric] = (count / total_failures) * 100

    return distribution, metric_percentage

def generate_random_hex_color():
    """Generate a random hex color code."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def map_metric_distribution_to_ui(distribution, metric_percentage):
    # Create a list to hold the formatted metrics data
    metrics_ui = []

    # Iterate over each metric and prepare the data
    for metric, augmentation_distribution in distribution.items():
        segments = []
        # Iterate over each augmentation type within the metric
        for idx, (augmentation_type, percentage) in enumerate(augmentation_distribution.items(), start=1):
            # Create a segment dictionary
            segment = {
                "color": generate_random_hex_color(),  # Assign a dynamically generated random hex color
                "percentage": percentage,
                "label": augmentation_type,
                "id": idx  # Use index as ID
            }
            segments.append(segment)

        # Create a metric dictionary
        metric_ui = {
            "heading": metric,
            "totalPercentage": metric_percentage[metric],
            "segments": segments
        }

        metrics_ui.append(metric_ui)


    return {"metrics": metrics_ui}

# Example usage
if __name__ == "__main__":
    content = [
    {
        "call_conversation": "Customer: I tried doing it online but your website is completely useless.",
        "augmentation_type": "Informal language",
        "failed_metric": "Coverage",
        "failure_reason": "The summary does not mention the customer's complaint about the website."
    },
    {
        "call_conversation": "This is taking forever, you should have had this done already.",
        "augmentation_type": "Short conversation",
        "failed_metric": "Coverage",
        "failure_reason": "The summary fails to capture the urgency and impatience expressed by the customer."
    },
    {
        "call_conversation": "Customer: But I wanted first class. Do you have anything in first class?",
        "augmentation_type": "Informal",
        "failed_metric": "Factuality",
        "failure_reason": "The summary does not mention the customer's specific request for first-class seating and meals."
    },
    {
        "call_conversation": "Customer: Why is my payment not working? Agent: Could you try again? Let me check your details.",
        "augmentation_type": "Noise Level - Noisy Environment",
        "failed_metric": "Factuality",
        "failure_reason": "Background noise affects clarity, resulting in an incomplete transaction context."
    },
    {
        "call_conversation": "Agent: Your payment is complete. Customer: But I haven't provided my card details yet.",
        "augmentation_type": "Contradiction Handling",
        "failed_metric": "Contradiction",
        "failure_reason": "The model incorrectly assumes payment completion without customer input."
    },
    {
        "call_conversation": "CSR: My apologies for the delay, Mr. Johnson. I'm pulling up options now.",
        "augmentation_type": "Polite tone",
        "failed_metric": "Contradiction",
        "failure_reason": "The summary incorrectly states that the agent was rude, contradicting the actual tone."
    },
    {
        "call_conversation": "Customer: Please ensure my name is spelled correctly as 'John Smith.'",
        "augmentation_type": "Formal conversation",
        "failed_metric": "Exact Match",
        "failure_reason": "The name in the summary is incorrectly recorded as 'Jon Smith.'"
    },
    {
        "call_conversation": "Agent: Your payment is due tomorrow. Customer: Okay, I'll pay it today.",
        "augmentation_type": "Time Sensitivity",
        "failed_metric": "Exact Match",
        "failure_reason": "The summary incorrectly mentions the payment is due today."
    },
    {
        "call_conversation": "Customer: How can I reach the support team? Agent: You can call us at 1-800-555-1234.",
        "augmentation_type": "Information delivery",
        "failed_metric": "Rouge",
        "failure_reason": "The summary omits the contact details provided by the agent."
    },
    {
        "call_conversation": "Customer: Is this issue fixed permanently? Agent: Yes, it’s fixed for now.",
        "augmentation_type": "Vague responses",
        "failed_metric": "Rouge",
        "failure_reason": "The summary omits the temporary nature of the resolution."
    },
    {
        "call_conversation": "Customer: I hope this refund process is quick.",
        "augmentation_type": "Customer sentiment analysis",
        "failed_metric": "Toxicity",
        "failure_reason": "The summary incorrectly states that the customer was aggressive."
    },
    {
        "call_conversation": "Agent: You should have read the instructions carefully.",
        "augmentation_type": "Harsh tone",
        "failed_metric": "Toxicity",
        "failure_reason": "The summary fails to capture the agent's harsh tone."
    },
    {
        "call_conversation": "Customer: I need my refund by tomorrow.",
        "augmentation_type": "Urgency",
        "failed_metric": "Coherence",
        "failure_reason": "The summary fails to maintain logical flow between urgency and resolution."
    },
    {
        "call_conversation": "Agent: This is resolved. Customer: Great! What about the next step?",
        "augmentation_type": "Incomplete conversation",
        "failed_metric": "Coherence",
        "failure_reason": "The summary ignores the follow-up question from the customer."
    },
    {
        "call_conversation": "Customer: I was promised a refund within 3 days, and it's been a week.",
        "augmentation_type": "Delayed service",
        "failed_metric": "Completeness",
        "failure_reason": "The summary does not mention the delay in the refund."
    },
    {
        "call_conversation": "Agent: Your issue has been escalated to the technical team.",
        "augmentation_type": "Escalation handling",
        "failed_metric": "Completeness",
        "failure_reason": "The summary omits the escalation step."
    },
    {
        "call_conversation": "Customer: What do I need to do to reset my password?",
        "augmentation_type": "Critical information",
        "failed_metric": "Informativeness",
        "failure_reason": "The summary does not mention the password reset instructions."
    },
    {
        "call_conversation": "Agent: Click on 'Forgot Password' and follow the steps.",
        "augmentation_type": "Instruction delivery",
        "failed_metric": "Informativeness",
        "failure_reason": "The summary fails to convey the critical steps shared by the agent."
    },
    {
        "call_conversation": "Customer: Why do I need to provide my ID again? Agent: It's for verification purposes.",
        "augmentation_type": "Repetitive requests",
        "failed_metric": "Redundancy",
        "failure_reason": "The summary repeats the customer's question unnecessarily."
    },
    {
        "call_conversation": "Customer: What's the cost of this service? Agent: $50. Customer: And for premium service? Agent: $100.",
        "augmentation_type": "Pricing details",
        "failed_metric": "Redundancy",
        "failure_reason": "The summary redundantly lists both prices without distinction."
    },
    {
        "call_conversation": "Customer: Is this illegal? Agent: No, it's completely legal.",
        "augmentation_type": "Legal compliance",
        "failed_metric": "Criminality",
        "failure_reason": "The summary implies potential criminal activity when there is none."
    },
    {
        "call_conversation": "Customer: Are you sure this is allowed? Agent: Yes, it's within regulations.",
        "augmentation_type": "Policy compliance",
        "failed_metric": "Criminality",
        "failure_reason": "The summary raises unnecessary doubt about legality."
    },
    {
        "call_conversation": "Agent: We can't help you with that.",
        "augmentation_type": "Lack of support",
        "failed_metric": "Unethical",
        "failure_reason": "The summary fails to reflect the ethical implications of denying help."
    },
    {
        "call_conversation": "Customer: I feel discriminated against. Agent: That's not our intention.",
        "augmentation_type": "Sensitive topic",
        "failed_metric": "Unethical",
        "failure_reason": "The summary does not address the customer's concern about discrimination."
    }
]

    #### the next 4 functions called in evaluationasync.py --- use this file separate###
    insights_analysis = analyze_insights(content)
    distribution , metric_percentage =calculate_failure_distribution(content)
    insights =map_metric_distribution_to_ui(distribution,metric_percentage)
    insights_analysis_result = json.loads(insights_analysis)
    #### the previous 4 functions called in evaluationasync.py --- use this file separate###
    print(json.dumps(insights_analysis_result, indent=4))
    execution_id = str(ObjectId())  # Generating a new unique ObjectId

    # Add executionId to the insights data
    insights_with_execution_id = {
        "executionId": execution_id,
        "metrics": insights,
        "insights":insights_analysis_result  # Add BERT analysis results under insights
    }
    collection_name="insights"   ####if collection not present, create it , as of now create it manually###
    try:
        # Insert insights into the MongoDB collection
        inserted_id = mongo_obj.write_single_data(collection_name, insights_with_execution_id)
        logging.info(f"Insights data inserted successfully with ID: {inserted_id}")
    except Exception as e:
        logging.error(f"Failed to insert insights data into the database: {str(e)}")

    # print(distribution)
    # print(metric_percentage)

    # print(BERT_analysis)