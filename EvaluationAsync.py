import sys

import pandas as pd
import torch

from src.GenerativeAI.DBOperation.EvaluationReadWrite import evaluationReadWrite
# from src.GENAI.DataGenDbOperations import DataGenerator

from src.GenerativeAI.CoreLogicLayer.Evaluation.EvaluationEngineAsyncNew import Feedback
from src.GenerativeAI.LLMLayer.LLMInitializer import LLMInterface
from src.Utilities.SetupUtilities import setup_utilities
# from src.GENAI.EvaluationEngineAsync import Feedback
from src.Utilities.utils import unique_id_generator
# from src.daolayer import MongoReadWrite, VectorReadWrite
from datetime import datetime
from flask import send_file
from io import BytesIO
import time
import asyncio
from src.config_file import config
from src.GenerativeAI.CoreLogicLayer.Metrices.insights import *


from src.Constants import CURRENT_METRICS

from src.GenerativeAI.CoreLogicLayer.Evaluation.insights import *


# from src.LlmLayer.LiteLLM import LiteLlm
# from src.GENAI.DataGenerator import obj1
record = 0

class EvaluationAsync:
    mongo_obj = evaluationReadWrite()


    # total_latency = 0
    # total_cost = 0
    # total_words = 0
    # total_characters = 0

    def evaluation_calculation(self, input_param):
        try:
            botData_df = pd.DataFrame(
                list(self.mongo_obj.read_data('bot_question_answer', input_param['datasetId'], 'datasetId')))
            predefined_answer_data = list(
                self.mongo_obj.read_data('predefined_questions_answers', input_param['datasetId'], 'datasetId'))
            data_df = botData_df[['question', 'answer', 'context', 'augmentedType']]
            config_doc = self.mongo_obj.read_single_data('metrices_config_collection', input_param['configId'])
            usecase = None
            no_of_templates = 1
            prompt = None
            testcase = None
            additional_context = None

            # Single pass through the structure
            for category in config_doc['metrices_data']:
                if category['name'] == 'Trustworthy Assurance':
                    for metric in category['metrices']:
                        if metric['value'] == 'jailbreak' and metric['enabled']:
                            config['JAILBREAK_FLAG'] = True
                            usecase = metric['usecase']
                            no_of_templates = metric['noOfTemplates']
                        elif metric['value'] == 'adversarial_attack' and metric['enabled']:
                            config['ADVERSARIAL_ATTACK_FLAG'] = True
                            prompt = metric['prompt']
                        elif metric['value'] == 'red_teaming' and metric['enabled']:
                            config['RED_TEAMING_FLAG'] = True
                            testcase = metric['testcase']
                            additional_context = metric['additional_context']

            print("*****************************USE CASE :: ", usecase)
            print("*****************************NO OF TEMPLATES :: ", no_of_templates)
            print("*****************************PROMPT :: ", prompt)
            print("*****************************TEST CASE :: ", testcase)
            print("*****************************ADDITIONAL CONTEXT :: ", additional_context)

            #########################################################################

            dataset_doc = self.mongo_obj.read_single_data('dataset', input_param['datasetId'])
            model = input_param['model']
            print(model, "77 77 77")
            LLMInterface(input_param['datasetId'])
            for predefined_answer in predefined_answer_data:
                predefined_ans = predefined_answer['answer']
            evaluation_data = {
                'jobId': input_param['jobID'],
                'datasetId': input_param['datasetId'],
                'executionStatus': 'Inprogress',
                'executedBy': input_param['executedBy'],
                'projectName': input_param['projectName'],
                'executedDate': datetime.utcnow().isoformat(),
                'modifiedDate': datetime.utcnow().isoformat(),
                'errorMessage': 'evaluation started successfully',
                'executionType': input_param['executionType'],
                'model': model,
                'executionTime': 0,
                'totalRecords': len(data_df),
                'configId': input_param['configId'],
                'configurationName': config_doc['name'],
                'datasetName': dataset_doc['datasetName']
            }

            execution_Id = self.mongo_obj.write_single_data('evaluation', evaluation_data)
            overall_start_time = time.time()
            async_execution = asyncio.run(
                self.asyncExeFun(data_df, input_param['datasetId'], input_param['jobID'], execution_Id, model, usecase,
                                 no_of_templates, prompt,
                                 testcase, additional_context, predefined_ans))
            overall_execution_time = (time.time() - overall_start_time) / 60
            self.mongo_obj.write_multiple_data('evaluationRecord', async_execution)
            evaluation_data['executionStatus'] = 'Completed'
            evaluation_data['errorMessage'] = 'Metrics evaluation completed successfully.'
            evaluation_data['modifiedDate'] = datetime.utcnow().isoformat()
            evaluation_data['executionTime'] = (round(overall_execution_time / 60, 2))
            self.mongo_obj.update_single_data('evaluation', evaluation_data, execution_Id)
        except Exception as e:
            import sys
            print('in error', str(e), sys.exc_info()[-1].tb_lineno)
            if input_param['jobID'] is not None:
                try:
                    evaluation_detail = self.mongo_obj.read_single_data_with_filter('evaluation', input_param['jobID'],
                                                                                    'jobId')
                    evaluation_detail['executionStatus'] = 'Failed'
                    evaluation_detail['errorMessage'] = 'Evaluation Calculation failed with error: ' + str(e)
                    self.mongo_obj.update_single_data('evaluation', evaluation_detail, evaluation_detail['_id'])
                except Exception as exc:
                    print('in error', str(exc), sys.exc_info()[-1].tb_lineno)

    async def asyncExeFun(self, data_df, dataset_id, jobId, execution_Id, model, usecase, no_of_templates, prompt,
                          testcase,
                          additional_context, predefined_ans):
        try:
            combineList = []
            for index, row in data_df.iterrows():
                feedback = await self.process_feedback(row, dataset_id, jobId, execution_Id, model, usecase,
                                                       no_of_templates, prompt, testcase, additional_context,
                                                       predefined_ans)

                print("@@@@@@@@@@@@@@@@@@@@@@@@ FEEDBACK in AsyncExecFun Return :: ", feedback)
                combineList.append(feedback)
                await asyncio.sleep(1)
            # if usecase or prompt or testcase:
            #     print("************** if usecase or prompt or testcase ********************************")
            #     for i in range(data_df.shape[0]):
            #         feedback = await self.process_feedback(data_df, i, jobId, execution_Id, model, usecase,
            #                                                no_of_templates, prompt, testcase, additional_context)
            #
            #         print("@@@@@@@@@@@@@@@@@@@@@@@@ FEEDBACK in AsyncExecFun Return :: ", feedback)
            #         combineList.append(feedback)
            #         await asyncio.sleep(1)
            # else:
            #     print(
            #         "************** if usecase or prompt or testcase ELSE CASE loop 6 times ********************************")
            #     for i in range(data_df.shape[0]):
            #         feedback = await self.process_feedback(data_df, i, jobId, execution_Id, model, usecase,
            #                                                no_of_templates, prompt, testcase, additional_context)
            #
            #         combineList.append(feedback)
            #         await asyncio.sleep(1)
            return combineList
        except Exception as exc:
            print('in error', str(exc), sys.exc_info()[-1].tb_lineno)

    @staticmethod
    async def call_function_with_timeout(feedback, func, func_name, timeout):
        try:
            print(f"Starting function '{func_name}' with {feedback.query}...")
            result = await asyncio.wait_for(func(), timeout)
            print(f"Function '{func_name}' completed successfully")
            print("async check result===", result)
            return func_name, result, None
        except asyncio.TimeoutError:
            print(f"Function '{func_name}' timed out")
            return func_name, None, "timeout"
        except Exception as e:
            print(f"Function '{func_name}' raised an exception: {e}")
            return func_name, None, f"exception: {e}"

    # async def call_functions(self, feedback, function_names,
    #                          timeout=3000):  # value increased for adversarial or something change it back to 300 send value for the specific metric instead
    #     tasks = []
    #     results = {}
    #     valid_metrices = []
    #     invalid_metrices = []
    #     print("function_names_check", function_names)
    #     if not function_names:
    #         print("No function name selected")
    #         return results, valid_metrices, invalid_metrices
    #
    #     # Split function names into custom and standard metrics
    #     custom_metrics = [fn for fn, _ in function_names if fn.startswith("custom_")]
    #     standard_metrics = [fn for fn, _ in function_names if not fn.startswith("custom_")]
    #     print("function_names",function_names)
    #     print("custom metrics",custom_metrics)
    #
    #     # Process standard metrics
    #     for func_name in standard_metrics:
    #         if hasattr(feedback, func_name):
    #             func = getattr(feedback, func_name)
    #             task = asyncio.create_task(self.call_function_with_timeout(feedback, func, func_name, timeout))
    #             tasks.append(task)
    #             valid_metrices.append(func_name)
    #         else:
    #             print(f"Function '{func_name}' not found in 'feedback' object.")
    #             invalid_metrices.append(func_name)
    #
    #     if custom_metrics:
    #         print("if custom metrics")
    #         # Remove 'custom_' prefix to get the actual metric names
    #         original_metrics = [metric[len("custom_"):] for metric in custom_metrics]
    #         if hasattr(feedback, "feedback_custom_metrics"):
    #             print("trying to call custom.....")
    #             func = getattr(feedback, "feedback_custom_metrics")
    #             task = asyncio.create_task(
    #                 self.call_function_with_timeout(feedback, lambda: func(original_metrics), "feedback_custom_metrics",
    #                                                 timeout)
    #             )
    #             tasks.append(task)
    #             valid_metrices.extend(original_metrics)  # Add original metric names to valid metrics
    #         else:
    #             print("Custom metric handler 'feedback_custom_metrics' not found in 'feedback' object.")
    #             invalid_metrices.extend(original_metrics)  # Add original metric names to invalid metrics
    #
    #     completed_tasks = await asyncio.gather(*tasks)
    #
    #     for func_name, result, error in completed_tasks:
    #         if error:
    #             results[func_name] = (False, error)
    #         else:
    #             results[func_name] = (True, result)
    #
    #     return (results, valid_metrices, invalid_metrices, feedback.latency,
    #             feedback.totalcost, feedback.total_word, feedback.total_characters)

    async def process_metrics_batch(self, feedback, metrics_batch):
        tasks = []
        results = {}
        valid_metrics = []
        invalid_metrics = []

        # Split metrics into custom and standard metrics
        custom_metrics = [(metric_value, metric_name) for metric_value, metric_name in metrics_batch if
                          metric_value.startswith("custom_")]
        standard_metrics = [(metric_value, metric_name) for metric_value, metric_name in metrics_batch if
                            not metric_value.startswith("custom_")]

        # Process standard metrics
        for metric_value, metric_name in standard_metrics:
            if hasattr(feedback, metric_value):
                func = getattr(feedback, metric_value)
                task = asyncio.create_task(self.call_function_with_timeout(feedback, func, metric_value, timeout=300))
                tasks.append(task)
                valid_metrics.append((metric_value, metric_name))
            else:
                print(f"Function '{metric_value}' not found in 'feedback' object.")
                invalid_metrics.append((metric_value, metric_name))
                results[metric_value] = (False, f"Function not available: {metric_value}")

        # Process custom metrics
        if custom_metrics:
            original_metrics = [metric_value[len("custom_"):] for metric_value, _ in custom_metrics]
            if hasattr(feedback, "feedback_custom_metrics"):
                func = getattr(feedback, "feedback_custom_metrics")
                task = asyncio.create_task(
                    self.call_function_with_timeout(feedback, lambda: func(original_metrics), "feedback_custom_metrics",
                                                    timeout=300)
                )
                tasks.append(task)
                valid_metrics.extend(
                    [(f"custom_{m}", m) for m in original_metrics])  # Add custom metrics with prefix to valid metrics
            else:
                print("Custom metric handler 'feedback_custom_metrics' not found in 'feedback' object.")
                invalid_metrics.extend([(f"custom_{m}", m) for m in original_metrics])  # Mark custom metrics as invalid

        # Wait for all tasks to complete
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results of completed tasks
        for i, completed_task in enumerate(completed_tasks):
            if isinstance(completed_task, Exception):
                metric_value, _ = valid_metrics[i]
                results[metric_value] = (False, str(completed_task))
            else:
                func_name, result, error = completed_task
                if error:
                    results[func_name] = (False, error)
                else:
                    results[func_name] = (True, result)
        print("results*****", results)
        return results, valid_metrics, invalid_metrics

    # custom metric modification is not done in the below methods

    async def store_batch_results(self, jobId, execution_Id, question, answer, context, batch_results):
        try:
            # Prepare the data to be stored
            data_to_store = {
                'jobId': jobId,
                'executionId': execution_Id,
                'question': question,
                'answer': answer,
                'context': context,
                'metrics': {},
                'trustworthy_metrics': {},
                'metadata': {}
            }

            for metric_name, (is_valid, result) in batch_results.items():
                print("metric name check====", metric_name)
                print("result check===", result)
                if is_valid:
                    if metric_name in ["toxicity", "language_match", "cosine_similarity"]:
                        data_to_store['metrics'][metric_name] = result
                    elif metric_name in ["jailbreak", "adversarial_attack", "red_teaming"]:
                        data_to_store['trustworthy_metrics'][metric_name] = result
                    else:
                        data_to_store['metrics'][metric_name] = result[0]
                        data_to_store['metrics'][f'{metric_name}_reason'] = result[1]

            print("data to store*********", data_to_store)

            # Use an asynchronous database operation to store the results
            await self.mongo_obj.async_update_evaluation_record(execution_Id, data_to_store)

        except Exception as e:
            print(f"Error storing chunk results: {str(e)}")
            # Log the error for later analysis

    # async def process_feedback(self, row, dataset_id, jobId, execution_Id, model, usecase, no_of_templates, prompt, testcase,
    #                            additional_context,predefined_ans):
    #     try:
    #         start_time = time.time()
    #
    #         # Fetching enabled metrics configuration from the database
    #         evaluation = self.mongo_obj.read_single_data_with_filter('evaluation', jobId, 'jobId')
    #
    #         golden_response = self.mongo_obj.read_data_with_multiple_filter('predefined_questions_answers',
    #                                                                         {"datasetId": dataset_id,
    #                                                                          'question': row['question']})
    #         golden_response = [question.get('answer') for question in golden_response]
    #         feedback = Feedback(row['question'], row['answer'], row['context'], golden_response,
    #                             model, usecase, no_of_templates, prompt, testcase, additional_context,predefined_ans)
    #
    #         config_doc = self.mongo_obj.read_single_data('metrices_config_collection', evaluation['configId'])
    #         document = self.mongo_obj.read_single_data_with_filter('metrices_config_collection', config_doc['name'],
    #                                                                'name')
    #         enabled_metrics = []
    #         enabled_custom_metrics = []
    #
    #         # Check if document contains custom metrics and filter them
    #
    #         for metric_data in document['metrices_data']:
    #             for metric in metric_data['metrices']:
    #                 if metric['enabled']:
    #                     if metric_data['name'] == "Custom Metrices":
    #                         enabled_custom_metrics.append((f"custom_{metric['value']}", metric['name']))
    #                     else:
    #                         enabled_metrics.append((metric['value'], metric['name']))
    #
    #         # Process metrics in chunks
    #         all_metrics = dict()
    #         valid_metrics = list()
    #         invalid_metrics = list()
    #         batch_size = 10
    #         for i in range(0, len(enabled_metrics), batch_size):
    #             metrics_batch = enabled_metrics[i:i + batch_size]
    #             batch_results, batch_valid, batch_invalid = await self.process_metrics_batch(feedback, metrics_batch)
    #             all_metrics.update(batch_results)
    #             valid_metrics.extend(batch_valid)
    #             invalid_metrics.extend(batch_invalid)
    #
    #         print(len(enabled_metrics))
    #         print(len(all_metrics))
    #         print("enabled_metrics_list", enabled_metrics)
    #         print("all_metrics\n\n", all_metrics)
    #         print("valid_metrics\n", valid_metrics)
    #         print("invalid_metrics: ", invalid_metrics)
    #         truLens_metrics = {}
    #         trustworthy_metrices = {}
    #         # metrics_valuation = {}
    #         metadata_metrics = {"latency": f"{round(feedback.latency, 2)} s",
    #                             "cost": f"${round(feedback.totalcost, 4)}", 'total_words': feedback.total_word,
    #                             'total_char': feedback.total_characters}
    #
    #         # Grouping metrics into categories for easier management
    #         non_trulens_based_metrics = {"toxicity", "language_match", "cosine_similarity", "jailbreak",
    #                                      "adversarial_attack", "red_teaming", "f1_score", "bleu", "rouge_1",
    #                                      "rouge_2", "rouge_l", "meteor_score", "hallucination", "profanity",
    #                                      "consistent", "is_unethical", "professionalism"}
    #
    #         reason_metrics = {"profanity", "consistent", "is_unethical", "professionalism", "f1_score",
    #                           "bleu", "rouge_1", "rouge_2", "rouge_l", "meteor_score", "hallucination"}
    #
    #         trustworthy_metrics = {"jailbreak", "adversarial_attack", "red_teaming"}
    #
    #         # Iterating through the zipped metrics
    #         for ((function_name, (run_successfully, result)), (metric_fn, display_name)) in zip(all_metrics.items(),
    #                                                                                             enabled_metrics):
    #
    #             if run_successfully:
    #                 if function_name not in non_trulens_based_metrics:
    #                     print("if in async")
    #                     print("tuple processing:", function_name, result, sep="\t")
    #                     truLens_metrics[display_name] = result[0]
    #                     truLens_metrics[f'{display_name}_REASON'] = result[1]
    #
    #                 elif function_name in reason_metrics:
    #                     truLens_metrics[display_name] = result[0]
    #                     truLens_metrics[f'{display_name}_reason'] = result[1]
    #
    #                 elif function_name in trustworthy_metrics:
    #                     trustworthy_metrices[display_name] = result
    #
    #                 else:
    #                     print("result in else=====", result)
    #                     truLens_metrics[display_name] = result
    #             else:
    #                 print(f"Error occurred while processing metric '{metric_fn}': {result[0]}")
    #
    #         latency_rounded = round(feedback.latency, 4)  # Round to 2 decimal places
    #
    #         # Calculating execution time
    #         end_time = (time.time() - start_time) * 100
    #
    #         # Constructing combined data
    #         combined_data = {
    #             'jobId': jobId,
    #             'configId': evaluation['configId'],
    #             'executionId': execution_Id,
    #             'question': row['question'],
    #             'answer': row['answer'],
    #             'context': row['context'],
    #             'metrics': truLens_metrics,
    #             'trustworthy_metrices': trustworthy_metrices,
    #             # 'metrics_valuation': metrics_valuation,
    #             'metadata': metadata_metrics,
    #             'executionTime': f"{round(end_time, 2)} s",
    #             'latency': latency_rounded,
    #             'totalcost': round(feedback.totalcost, 2),
    #         }
    #         print("metrics====", combined_data['metrics'])
    #         return combined_data
    #
    #     except Exception as e:
    #         import sys
    #         print('in error', str(e), sys.exc_info()[-1].tb_lineno)

    # def insights_ui(executionId, content):
    #
    #     insights_analysis = analyze_insights(content)
    #     distribution, metric_percentage = calculate_failure_distribution(content)
    #     insights = map_metric_distribution_to_ui(distribution, metric_percentage)
    #     insights_analysis_result = json.loads(insights_analysis)
    #
    #     insights_with_execution_id = {
    #         "executionId": executionId,
    #         "metrics": insights,
    #         "insights": insights_analysis_result  # Add BERT analysis results under insights
    #     }
    #     collection_name = "insights"  ####if collection not present, create it , as of now create it manually###
    #     try:
    #         # Insert insights into the MongoDB collection
    #         inserted_id = mongo_obj.write_single_data(collection_name, insights_with_execution_id)
    #         logging.info(f"Insights data inserted successfully with ID: {inserted_id}")
    #     except Exception as e:
    #         logging.error(f"Failed to insert insights data into the database: {str(e)}")

    async def process_feedback(self, row, dataset_id, jobId, execution_Id, model, usecase, no_of_templates, prompt,
                               testcase,
                               additional_context, predefined_ans):
        import time
        import datetime


        try:
            start_time =time.time()
            print(start_time)

            timestamp = datetime.datetime.now()
            print('TIME STAMP',timestamp)



            # Fetching enabled metrics configuration from the database
            evaluation = self.mongo_obj.read_single_data_with_filter('evaluation', jobId, 'jobId')

            golden_response = self.mongo_obj.read_data_with_multiple_filter('predefined_questions_answers',
                                                                            {"datasetId": dataset_id,
                                                                             'question': row['question']})
            golden_response = [question.get('answer') for question in golden_response]
            feedback = Feedback(row['question'], row['answer'], row['context'], golden_response,
                                model, usecase, no_of_templates, prompt, testcase, additional_context, predefined_ans)

            # Fetch metrics configuration
            config_doc = self.mongo_obj.read_single_data('metrices_config_collection', evaluation['configId'])
            document = self.mongo_obj.read_single_data_with_filter('metrices_config_collection', config_doc['name'],
                                                                   'name')

            # Prepare lists of enabled metrics
            enabled_metrics = []
            enabled_custom_metrics = []

            for metric_data in document['metrices_data']:
                for metric in metric_data['metrices']:
                    if metric['enabled']:
                        if metric_data['name'] == "Custom Metrices":
                            enabled_custom_metrics.append((f"custom_{metric['value']}", metric['name']))
                        else:
                            enabled_metrics.append((metric['value'], metric['name']))

            all_enabled_metrics = enabled_metrics + enabled_custom_metrics

            print(all_enabled_metrics)

            # Process metrics in chunks
            all_metrics = dict()
            valid_metrics = list()
            invalid_metrics = list()
            batch_size = 10

            # Process both standard and custom metrics in batches
            for i in range(0, len(all_enabled_metrics), batch_size):
                metrics_batch = all_enabled_metrics[i:i + batch_size]
                batch_results, batch_valid, batch_invalid = await self.process_metrics_batch(feedback, metrics_batch)
                all_metrics.update(batch_results)
                valid_metrics.extend(batch_valid)
                invalid_metrics.extend(batch_invalid)
                print("all_metrics for loop", all_metrics)
            print("all metrics outside for loop", all_metrics)

            # Initialize dictionaries
            truLens_metrics = {}
            trustworthy_metrices = {}
            custom_metrics = {}
            metadata_metrics = {"latency": f"{round(feedback.latency, 2)} s",
                                "cost": f"${round(feedback.totalcost, 4)}", 'total_words': feedback.total_word,
                                'total_char': feedback.total_characters}

            # Define metric categories

            non_trulens_based_metrics = {"toxicity", "language_match", "cosine_similarity", "jailbreak",
                                         "adversarial_attack", "red_teaming", "f1_score", "bleu", "rouge_1",
                                         "rouge_2", "rouge_l", "meteor_score", "hallucination", "profanity",
                                         "consistent", "is_unethical", "professionalism"}

            reason_metrics = {"profanity", "consistent", "is_unethical", "professionalism", "f1_score",
                              "bleu", "rouge_1", "rouge_2", "rouge_l", "meteor_score", "hallucination"}

            trustworthy_metrics = {"jailbreak", "adversarial_attack", "red_teaming"
                                   }

            # custom_metrics={"feedback_custom_metrics"}

            content = []
            # Process all metrics
            for ((function_name, (run_successfully, result)), (metric_fn, display_name)) in zip(all_metrics.items(),
                                                                                                all_enabled_metrics):
                print("run_successfully", run_successfully)
                print("to add custom result", result)
                print("function name", function_name)
                if run_successfully:
                    if function_name == "feedback_custom_metrics" and isinstance(result, dict):
                        print("Processing custom metrics")
                        # Process custom metrics only when feedback_custom_metrics is True
                        for custom_metric_name, metric_details in result.items():
                            custom_metrics[custom_metric_name] = {
                                'final_output': metric_details['final_output'],
                                'status': metric_details['status'],
                                'custom_name': metric_details['custom_name'],
                                'color': metric_details['color']
                            }
                    elif function_name not in non_trulens_based_metrics:
                        print("if not in non trulens")
                        print("tuple processing:", function_name, result, sep="\t")
                        truLens_metrics[display_name] = result[0]
                        truLens_metrics[f'{display_name}_REASON'] = result[1]
                        truLens_metrics[f'{display_name}_TABLE'] = result[2]
                        ###from prompt take failures and hae it in separate list - input to insights###

                        
                        explicit_metrics = {'toxicity', 'criminality', 'insensitivity', 'unethical'}

                        if non_trulens_based_metrics in explicit_metrics and result[0] > 0.5:
                            content_entry = {
                                "call_conversation": row.get('question', "N/A"),
                                "augmentation_type": row.get('augmentedType', "Unknown"),
                                "failed_metric": display_name,
                                "failure_reason": result[1]
                            }

                            content.append(content_entry)

                        elif result[0] < 0.5:
                            content_entry = {
                                "call_conversation": row.get('question', "N/A"),
                                "augmentation_type": row.get('augmentedType', "Unknown"),
                                "failed_metric": display_name,
                                "failure_reason": result[1]
                            }

                            content.append(content_entry)

                        

                    elif function_name in reason_metrics:
                        print("if in reason_metrics")
                        truLens_metrics[display_name] = result[0]
                        truLens_metrics[f'{display_name}_reason'] = result[1]

                    elif function_name in trustworthy_metrics:
                        print("if in trustworthy_metrices")
                        trustworthy_metrices[display_name] = result
                    else:
                        print("result in else=====", result)
                        truLens_metrics[display_name] = result
                else:
                    print(f"Error occurred while processing metric '{metric_fn}': {result[0]}")

            end_time = time.time()

            insights_analysis = analyze_insights(content)
            distribution, metric_percentage = calculate_failure_distribution(content)
            insights = map_metric_distribution_to_ui(distribution, metric_percentage)
            insights_analysis_result = json.loads(insights_analysis)

            insights_with_execution_id = {
                "executionId": execution_Id,
                "metrics": insights,
                "insights": insights_analysis_result
            }

            collection_name = "insights"

            try:
                # Insert insights into the MongoDB collection
                inserted_id = mongo_obj.write_single_data(collection_name, insights_with_execution_id)
                logging.info(f"Insights data inserted successfully with ID: {inserted_id}")
            except Exception as e:
                logging.error(f"Failed to insert insights data into the database: {str(e)}")

            # Prepare record
            record = {
                'jobId': jobId,
                'configId': evaluation['configId'],
                'executionId': execution_Id,
                'question': row['question'],
                'answer': row['answer'],
                'context': row['context'],
                'model': model,
                'metrics': truLens_metrics,
                'trustworthy_metrices': trustworthy_metrices,
                'custom_metrics': custom_metrics,
                'metadata_metrices': metadata_metrics,
                'latency': feedback.latency,
                'totalcost': feedback.totalcost,
                'totalwords': feedback.total_word,
                'totalcharacters': feedback.total_characters,
                'executionTime': round((end_time - start_time), 2)
            }


            return record#, insights_ui(execution_Id, content)

        except Exception as exc:
            import sys
            print('Error in process_feedback:', str(exc), sys.exc_info()[-1].tb_lineno)


    # def insights_ui(executionId, content):
    #
    #     insights_analysis = analyze_insights(content)
    #     distribution, metric_percentage = calculate_failure_distribution(content)
    #     insights = map_metric_distribution_to_ui(distribution, metric_percentage)
    #     insights_analysis_result = json.loads(insights_analysis)
    #
    #     insights_with_execution_id = {
    #         "executionId": executionId,
    #         "metrics": insights,
    #         "insights": insights_analysis_result  # Add BERT analysis results under insights
    #     }
    #     collection_name = "insights"  ####if collection not present, create it , as of now create it manually###
    #     try:
    #         # Insert insights into the MongoDB collection
    #         inserted_id = mongo_obj.write_single_data(collection_name, insights_with_execution_id)
    #         logging.info(f"Insights data inserted successfully with ID: {inserted_id}")
    #     except Exception as e:
    #         logging.error(f"Failed to insert insights data into the database: {str(e)}")
    #
    # # execution_Id = mongo_obj.write_single_data('evaluation', evaluation_data)
