from importlib import import_module
from dataset.info import DatasetInfo


def _evaluator_factory(name, result_dir, anno_file, eval_format, cfg):
    file = import_module('evaluator.{}.{}'.format(name, 'boundary' if eval_format == 'bound' else 'snake'))
    if eval_format in ('segm', 'bound'):
        evaluator = file.Evaluator(result_dir, anno_file, cfg.test)
    else:
        evaluator = file.DetectionEvaluator(result_dir, anno_file)
    return evaluator


def make_evaluator(cfg, format=None):
    name = cfg.test.dataset.split('_')[0]
    anno_file = DatasetInfo.dataset_info[cfg.test.dataset]['anno_dir']
    eval_format = cfg.test.segm_or_bbox if format is None else format
    return _evaluator_factory(name, cfg.commen.result_dir, anno_file, eval_format, cfg)
