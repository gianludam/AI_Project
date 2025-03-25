import timeit
import argparse
import json
from Applying_RAG import build_rag_pipeline

def get_rag_response(query, chain):
    """
    This function sends a query to the QA chain and attempts to extract a JSON object from the response.
    
    Args:
        query (str): The question to send to the QA chain.
        chain: The QA chain built by build_rag_pipeline.
    
    Returns:
        Either a parsed JSON object (if present) or the raw response string.
    """
    response = chain({'query': query})
    res = response['result']

    start_index = res.find('{')
    end_index = res.rfind('}')

    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_fragment = res[start_index:end_index + 1]
        try:
            json_data = json.loads(json_fragment)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    else:
        print("No JSON object found in the string.")

    return res

if __name__ == "__main__":

    """
    Main entry point for the query script.

    This block parses a command-line argument (the query to send to the LLM), 
    builds the Retrieval-Augmented Generation (RAG) pipeline, sends the query to the QA chain,
    and then prints out the answer along with the time taken to retrieve it.
    """
  
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='Who was William Howe?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    start = timeit.default_timer()

    qa_chain = build_rag_pipeline()
    print('Retrieving answer...')
    answer = get_rag_response(args.input, qa_chain)

    end = timeit.default_timer()

    print(f'\nAnswer:\n{answer}')
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start:.2f} seconds")



