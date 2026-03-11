import sys
sys.path.append("TrackEval")

from trackeval import Evaluator, datasets, metrics

# Config
eval_config = {
    'USE_PARALLEL': False,
    'NUM_PARALLEL_CORES': 1,
}

dataset_config = {
    'GT_FOLDER': 'TrackEval/data/gt/mot_challenge/',
    'TRACKERS_FOLDER': 'TrackEval/data/trackers/mot_challenge/',
    'OUTPUT_FOLDER': None,
    'TRACKERS_TO_EVAL': [ 'ByteTrack'],     #'DeepSORT', 'ByteTrack', 'DeepSORT_turkey'
    'BENCHMARK': 'Turkey',
    'SPLIT_TO_EVAL': 'test',
    'CLASSES_TO_EVAL': ['pedestrian'],  # name doesn't matter, kept for compatibility
    'PRINT_CONFIG': True,
}

metrics_config = {
    'METRICS': ['HOTA'],
}

# Create evaluator
evaluator = Evaluator(eval_config)

dataset_list = [datasets.MotChallenge2DBox(dataset_config)]
metrics_list = [metrics.HOTA()]

# Run evaluation
results = evaluator.evaluate(dataset_list, metrics_list)
