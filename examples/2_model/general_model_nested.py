import numpy as np
import pandas as pd
import torch
from torch import nn

from torch_concepts import Annotations, AxisAnnotation, ConceptGraph
from torch_concepts.nn import ExogEncoder, ProbPredictor, ProbEncoderFromExog, BipartiteModel, Propagator, GraphModel, \
    COSMOGraphLearner, LearnedGraphModel, BaseGraphLearner, ProbEncoderFromEmb, HyperLinearPredictor, MixProbExogPredictor
from torch_concepts.nn import KnownGraphInference, UnknownGraphInference, ProbEncoderFromEmb


def main():
    concept_names = ('c', 'b', 'a', 'd', 'e')
    cardinalities = (1, 2, 3, 5, 8)

    annotations = Annotations({1: AxisAnnotation(concept_names, cardinalities=cardinalities)})
    model_graph = ConceptGraph(torch.tensor([[0, 1, 0, 0, 1],
                                            [0, 0, 0, 0, 1],
                                            [0, 0, 0, 1, 0],
                                            [0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 0]]).float(),
                                           list(annotations.get_axis_annotation(1).labels))

    x = torch.randn(100, 13)
    concept_probs = torch.ones(100, sum(cardinalities))

    model = GraphModel(model_graph=model_graph,
                       exogenous=Propagator(ExogEncoder, embedding_size=7),
                       encoder=Propagator(ProbEncoderFromExog),
                       predictor=Propagator(HyperLinearPredictor, embedding_size=11),
                       annotations=annotations,
                       predictor_in_embedding=0,
                       predictor_in_exogenous=7,
                       has_self_exogenous=True,
                       has_parent_exogenous=False,
                       input_size=x.shape[1])
    inference_train = KnownGraphInference()
    cy_preds = inference_train.query(x, model=model)
    print(cy_preds)
    model = GraphModel(model_graph=model_graph,
                       encoder=Propagator(ProbEncoderFromEmb),
                       predictor=Propagator(ProbPredictor),
                       predictor_in_embedding=0,
                       predictor_in_exogenous=0,
                       has_self_exogenous=False,
                       has_parent_exogenous=False,
                       annotations=annotations,
                       input_size=x.shape[1])
    inference_train = KnownGraphInference()
    cy_preds = inference_train.query(x, model=model)
    print(cy_preds)

    # CGM
    model = LearnedGraphModel(model_graph=COSMOGraphLearner,
                              exogenous=Propagator(ExogEncoder, embedding_size=7),
                              encoder=Propagator(ProbEncoderFromExog),
                              predictor=Propagator(HyperLinearPredictor, embedding_size=11),
                              annotations=annotations,
                              predictor_in_embedding=0,
                              predictor_in_exogenous=7,
                              has_self_exogenous=True,
                              has_parent_exogenous=False,
                              input_size=x.shape[1])
    inference_train = UnknownGraphInference()
    c_encoder, c_predictor = inference_train.query(x, concept_probs, model=model)
    print(c_encoder)
    print(c_predictor)
    model = LearnedGraphModel(model_graph=COSMOGraphLearner,
                              exogenous=Propagator(ExogEncoder, embedding_size=7*2),
                              encoder=Propagator(ProbEncoderFromExog),
                              predictor=Propagator(MixProbExogPredictor),
                              annotations=annotations,
                              predictor_in_embedding=0,
                              predictor_in_exogenous=7,
                              has_self_exogenous=False,
                              has_parent_exogenous=True,
                              input_size=x.shape[1])
    inference_train = UnknownGraphInference()
    c_encoder, c_predictor = inference_train.query(x, concept_probs, model=model)
    print(c_encoder)
    print(c_predictor)

    # CEM
    model = BipartiteModel(task_names=['c', 'e'],
                           exogenous=Propagator(ExogEncoder, embedding_size=7*2),
                           encoder=Propagator(ProbEncoderFromExog),
                           predictor=Propagator(MixProbExogPredictor),
                           annotations=annotations,
                           predictor_in_embedding=0,
                           predictor_in_exogenous=7,
                           has_self_exogenous=False,
                           has_parent_exogenous=True,
                           input_size=x.shape[1])
    inference_test = KnownGraphInference()
    cy_pred = inference_test.query(x, model=model)

    # CBM
    model = BipartiteModel(task_names=['c', 'e'],
                           exogenous=Propagator(ExogEncoder, embedding_size=7),
                           encoder=Propagator(ProbEncoderFromExog),
                           predictor=Propagator(HyperLinearPredictor, embedding_size=11),
                           annotations=annotations,
                           predictor_in_embedding=0,
                           predictor_in_exogenous=7,
                           has_self_exogenous=True,
                           has_parent_exogenous=False,
                           input_size=x.shape[1])
    inference_test = KnownGraphInference()
    cy_pred = inference_test.query(x, model=model)
    model = BipartiteModel(task_names=['c', 'e'],
                           encoder=Propagator(ProbEncoderFromEmb),
                           predictor=Propagator(ProbPredictor),
                           annotations=annotations,
                           predictor_in_embedding=0,
                           predictor_in_exogenous=0,
                           has_self_exogenous=False,
                           has_parent_exogenous=False,
                           input_size=x.shape[1])
    inference_test = KnownGraphInference()
    cy_pred = inference_test.query(x, model=model)

    print(cy_pred)


if __name__ == "__main__":
    main()
