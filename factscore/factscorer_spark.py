import argparse
from pyspark.sql import *
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import string
import json
import numpy as np
import os, sys
import logging
import torch
from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval

class FactScorer(object):

    def __init__(self,
                 model_name="retrieval+ChatGPT",
                 af_generator_name = "InstructGPT",
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 max_passage_length = 256,
                 batch_size=256):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type
        self.af_generator_name = af_generator_name

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate

        if "llama" in model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "ChatGPT" in model_name:
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
        else:
            self.lm = None
        self.max_passage_length = max_passage_length

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path,max_passage_length=self.max_passage_length)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use gpt-3.5-turbo-instruct, the cost is $0.50 / 1M tokens
        # if we use gpt-3.5-turbo-0125, the cost is $0.50 / 1M tokens
        if model == "gpt-3.5-turbo-instruct":
            rate = 0.50
        elif model == "gpt-3.5-turbo-0125":
            rate = 0.50
        else:
            raise ValueError(f"Unsupported model: {model}")

        total_cost = total_tokens * rate / 1000000

        # print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.4f / 1M tokens): $%.4f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False,
                  n = 7,
                  batch_size=4,
                  k = 5):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if isinstance(topics, str) and isinstance(generations, str):
            topics = [topics]
            generations = [generations]
        else:
            assert isinstance(topics, list) and isinstance(generations, list), "`topics` and `generations` should be strings or lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, f"{self.af_generator_name}_af.pkl"),
                                                        model_name = self.af_generator_name,
                                                        n = n)

            # estimate the total cost of atomic fact generation
            total_words = 0
            for gen in generations:
                total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="atomic fact generation", model="gpt-3.5-turbo-instruct" if self.af_generator_name == 'InstructGPT' else "gpt-3.5-turbo-0125")

            # if verbose:
            #     topics = tqdm(topics)
            atomic_facts = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    total_words += self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate,batch_size=batch_size,k = k)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo-0125")
        if verbose:
            topics = tqdm(topics,desc='Scoring topics')
        scores = []
        init_scores = []
        decisions = []
        failed_count = 0
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            if facts is None:
                decisions.append(None)
            else:
                decision,no_count = self._get_score(topic, generation, facts, knowledge_source,batch_size=batch_size,k = k)
                if len(decision) > 0:
                    score = np.mean([d["is_supported"] for d in decision])
                
                    if gamma:
                        init_scores.append(score)
                        penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                        score = penalty * score
                
                    decisions.append(decision)
                    scores.append(score)
                if len(scores) % 10 == 0:
                    self.save_cache()
                failed_count += no_count
        if 'ChatGPT' in self.model_name:
            print ('Failed to predict the answer for %d atomic facts'%failed_count)
        self.save_cache()

        out = {"score": np.mean(scores) if len(scores)>0 else 0.0,
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "topics": topics,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None,batch_size=4,k=5):
        decisions = []
        total_words = 0
        prompts = []
        for atom in atomic_facts:
            atom = atom.strip()
            if self.lm:
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=k)
                definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())
                if cost_estimate:
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue
                prompts.append((atom,prompt))
            else:
                is_supported = True
                if "npm" in self.model_name:
                    npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                    is_supported = npprob > 0.3
                decisions.append({"atom": atom, "is_supported": is_supported})

        if cost_estimate:
            return total_words

        if not self.lm:
            return decisions, 0
        
        outputs= []
        for i in range(0, len(prompts), batch_size):
            curr_prompts = prompts[i:i+batch_size]
            outputs.extend(self.lm.generate([p[1] for p in curr_prompts]))
        
        total_no_counts = 0
        for i,output in enumerate(outputs):
            atom = prompts[i][0]
            if type(output[1])==np.ndarray or isinstance(output[1],torch.Tensor):
                # when logits are available
                logits = np.array(output[1])
                assert logits.shape[0] in [32000, 32001]
                true_score = logits[5852]
                false_score = logits[7700]
                is_supported = true_score > false_score
            else:
                # when logits are unavailable
                generated_answer = output[0].lower()
                if "true" in generated_answer or "false" in generated_answer:
                    if "true" in generated_answer and "false" not in generated_answer:
                        is_supported = True
                    elif "false" in generated_answer and "true" not in generated_answer:
                        is_supported = False
                    else:
                        is_supported = generated_answer.index("true") > generated_answer.index("false")
                else:
                    # is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])
                    is_supported = None
                    total_no_counts += 1
            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                is_supported = npprob > 0.3
                
            if 'ChatGPT' in self.model_name and is_supported is None:
                continue
            decisions.append({"atom": atom, "is_supported": is_supported})
        return decisions,total_no_counts
    
    def process_partition(partition, fact_scorer):
        results = []
        for line in partition:
            dp = json.loads(line)
            topics, generations, atomic_facts = [], [], []

            if "annotations" in dp:
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])

            out = fact_scorer.get_score(topics=topics,
                                        generations=generations,
                                        gamma=10,
                                        atomic_facts=atomic_facts if "annotations" in dp else None,
                                        knowledge_source="enwiki-20230401",
                                        batch_size=4,
                                        verbose=False)
            
            logging.critical("Topic: '%s'", topics[0])
            logging.critical("FActScore = %.1f%%" % (100*out["score"]))
            if "init_score" in out:
                logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
            logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
            logging.critical("Atomic facts per valid response = %.1f\n" % (out["num_facts_per_response"]))

            results.append({
                "topic": topics[0],
                "score": out["score"],
                "init_score": out.get("init_score"),
                "respond_ratio": out["respond_ratio"],
                "num_facts_per_response": out["num_facts_per_response"],
                "decisions": out["decisions"]
            })
        return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")
    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default=None)
    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")    
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)

    os.environ['PYSPARK_PYTHON']=sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON']=sys.executable

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    def numpy_to_python(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # Handles arrays
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    spark = SparkSession.builder \
        .appName("FactScorer") \
        .master("local[*]") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memoryOverhead", "2g") \
        .config("spark.driver.memoryOverhead", "2g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

    sc = spark.sparkContext
    
    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    # Read the input data
    with open(args.input_path) as f:
        data = [json.loads(line) for line in f]

    # Parallel processing with Spark
    data_rdd = sc.parallelize(data, numSlices=args.n_samples or (len(data) // 4))

    def process_entry(dp):
        topics, generations, atomic_facts = [], [], []

        if args.use_atomic_facts:
            if "annotations" not in dp or dp["annotations"] is None:
                return None
            topics.append(dp["topic"])
            generations.append(dp["output"])
            atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
        else:
            topics.append(dp["topic"])
            generations.append(dp["output"])

        out = fs.get_score(topics=topics,
                           generations=generations,
                           gamma=args.gamma,
                           atomic_facts=atomic_facts if args.use_atomic_facts else None,
                           knowledge_source=args.knowledge_source,
                           batch_size=4,
                           verbose=args.verbose)
        
        logging.critical("Topic: '%s'", topics[0])
        logging.critical("FActScore = %.1f%%" % (100*out["score"]))
        if "init_score" in out:
            logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
        logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
        logging.critical("Atomic facts per valid response = %.1f\n" % (out["num_facts_per_response"]))


        return {
            "topic": topics[0], 
            "score": out["score"],
            "init_score": out["init_score"],
            "respond_ratio": out["respond_ratio"],
            "num_facts_per_response": out["num_facts_per_response"],
            "decisions": out["decisions"]
        }

    results_rdd = data_rdd.map(process_entry).filter(lambda x: x is not None)
    results = results_rdd.collect()

    # Save the results
    with open(args.input_path.replace(".jsonl", f"_{args.model_name}_factscore_output.json"), 'w') as f:
        f.write(json.dumps(results, default=numpy_to_python, indent=4) + "\n")

    # Compute average metrics
    total_score = 0.0
    total_respond_ratio = 0.0
    total_num_facts_per_response = 0.0
    total_init_score = 0.0
    count = 0

    for result in results:
        total_score += result["score"]
        total_respond_ratio += result["respond_ratio"]
        total_num_facts_per_response += result["num_facts_per_response"]
        if "init_score" in result:
            total_init_score += result["init_score"]
        count += 1

    average_score = total_score / count if count > 0 else 0
    average_respond_ratio = total_respond_ratio / count if count > 0 else 0
    average_num_facts_per_response = total_num_facts_per_response / count if count > 0 else 0
    average_init_score = total_init_score / count if count > 0 else 0

    logging.critical("Average FActScore = %.1f%%", 100 * average_score)
    logging.critical("Average FActScore w/o length penalty = %.1f%%", 100 * average_init_score)
    logging.critical("Average Respond Ratio = %.1f%%", 100 * average_respond_ratio)
    logging.critical("Average Atomic Facts per Valid Response = %.1f\n", average_num_facts_per_response)
    
    spark.stop()
