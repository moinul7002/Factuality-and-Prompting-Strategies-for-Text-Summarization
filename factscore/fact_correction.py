# Pipeline for Fact Correction

import argparse
import os
import json
import logging
from factscore.atomic_facts import AtomicFactGenerator
from factscore.factscorer import FactScorer

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 3: Fact Correction
# Function to correct false facts using LLM and relevant knowledge source context
def correct_false_facts(af_generator, false_facts, scorer, knowledge_source, topic):
    corrected_facts = []
    for false_fact in false_facts:
        # Retrieve relevant context from the knowledge source
        passages = scorer.retrieval[knowledge_source].get_passages(topic, false_fact, k=3)
        context = "\n".join([f"Title: {p['title']}\nText: {p['text']}" for p in passages])
        
        # Generate corrected fact using LLM with context
        prompt = f"The following statement is factually incorrect: '{false_fact}'. Based on the given context, please provide a corrected version as a single sentence:\n\nContext:\n{context}"
        corrected, _ = af_generator.openai_lm.generate(prompt)
        # Ensure the corrected fact is a single sentence
        corrected = corrected.split('.')[0] + '.'
        corrected_facts.append(corrected)
        logger.debug(f"Corrected fact for '{false_fact}': {corrected}")
    return corrected_facts

# Step 4: Generate Corrected Decisions
# Function to generate new decisions list based on corrected facts
def generate_corrected_decisions(corrected_facts, original_decisions):
    corrected_decisions = []
    for decision in original_decisions:
        if not decision["is_supported"]:
            # If the original decision was false, replace with the corrected fact
            corrected_fact = corrected_facts.pop(0)
            corrected_decisions.append({"atom": corrected_fact, "is_supported": True})
        else:
            # Otherwise, keep the original decision
            corrected_decisions.append(decision)
    return corrected_decisions

# Step 5: Final Output Generation
def generate_final_output(input_text, verification_results, corrected_decisions):
    final_output = {
        "score": verification_results["score"],
        "respond_ratio": verification_results["respond_ratio"],
        "decisions": [corrected_decisions],
        "topics": verification_results["topics"],
        "num_facts_per_response": verification_results["num_facts_per_response"],
        "init_score": verification_results["init_score"]
    }
    return final_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True, help="Path to the input JSON file containing fact extraction and verification results.")
    parser.add_argument('--model_dir', type=str, default=".cache/factscore", help="Path to the model directory.")
    parser.add_argument('--knowledge_source', type=str, default=None, help="Knowledge source to verify the facts.")
    parser.add_argument('--openai_key', type=str, default="api.key", help="OpenAI API key path.")
    args = parser.parse_args()

    # Load existing fact extraction and verification results
    with open(args.input_json, 'r') as f:
        existing_results_list = json.load(f)

    # Ensure that the loaded data is a list of dictionaries
    if isinstance(existing_results_list, dict):
        existing_results_list = [existing_results_list]
    elif isinstance(existing_results_list, list):
        # Ensure each item in the list is a dictionary
        existing_results_list = [json.loads(item) if isinstance(item, str) else item for item in existing_results_list]

    # Initialize the atomic fact generator and fact scorer
    atomic_fact_generator = AtomicFactGenerator(args.openai_key, os.path.join(args.model_dir, "demos"), gpt3_cache_file=os.path.join(args.model_dir, "InstructGPT_af.pkl"), model_name='InstructGPT')
    fact_scorer = FactScorer(model_name="retrieval+ChatGPT", model_dir=args.model_dir, openai_key=args.openai_key)
    fact_scorer.register_knowledge_source(name=args.knowledge_source)

    final_outputs = []

    for existing_results in existing_results_list:
        input_text = existing_results.get("topics", [""])[0]
        original_decisions = existing_results.get("decisions", [[]])[0]
        atomic_facts = [decision["atom"] for decision in original_decisions]

        # Step 3: Fact Correction if necessary
        false_facts = [fact for fact, decision in zip(atomic_facts, original_decisions) if not decision["is_supported"]]
        if false_facts:
            corrected_facts = correct_false_facts(atomic_fact_generator, false_facts, fact_scorer, args.knowledge_source, input_text)
            logger.debug(f"Corrected Facts: {corrected_facts}")

            # Step 4: Generate corrected decisions
            corrected_decisions = generate_corrected_decisions(corrected_facts, original_decisions)
        else:
            corrected_decisions = original_decisions
            logger.debug("All facts are verified correctly, no corrections needed.")

        # Step 5: Generate Final Output
        final_output = generate_final_output(input_text, existing_results, corrected_decisions)
        final_outputs.append(final_output)

    # Save Final Output
    output_filename = os.path.splitext(args.input_json)[0] + '_corrected.json'
    with open(output_filename, 'w') as f:
        json.dump(final_outputs, f, indent=4)
    
    print(f"Final Output saved to: {output_filename}")
