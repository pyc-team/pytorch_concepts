import torch
from torch import nn

from torch_concepts import ConceptTensor, Annotations, AxisAnnotation, AnnotatedAdjacencyMatrix
from torch_concepts.nn import ExogEncoder, ProbPredictor, ProbEncoderFromExog, BipartiteModel, Propagator, GraphModel, \
    COSMOGraphLearner, LearnedGraphModel, BaseGraphLearner, ProbEncoderFromEmb, HyperLinearPredictor, MixProbExogPredictor
from torch_concepts.nn import KnownGraphInference, UnknownGraphInference, ProbEncoderFromEmb


def main():
    n_concepts = 5

    x = torch.randn(100, 13)
    concept_embs = torch.ones(100, n_concepts, 7) * 10       # embs
    concept_probs = torch.ones(100, n_concepts) * 5        # probs
    residuals = torch.ones(100, n_concepts) * -1

    annotations = Annotations({1: AxisAnnotation(('c', 'b', 'a', 'd', 'e'))})

    c = ConceptTensor(annotations, concept_probs)

    model_graph = AnnotatedAdjacencyMatrix(torch.tensor([[0, 1, 0, 0, 1],
                                                         [0, 0, 0, 0, 1],
                                                         [0, 0, 0, 1, 0],
                                                         [0, 1, 0, 0, 0],
                                                         [0, 0, 0, 0, 0]]).float(),
                                           annotations)
    model = GraphModel(model_graph=model_graph,
                       exogenous=Propagator(ExogEncoder, embedding_size=7),
                       encoder=Propagator(ProbEncoderFromExog),
                       predictor=Propagator(HyperLinearPredictor, embedding_size=11),
                       annotations=annotations,
                       predictor_in_embedding=0,
                       predictor_in_exogenous=7*2,
                       input_size=x.shape[1])
    inference_train = KnownGraphInference(model=model)
    cy_preds = inference_train.query(x)
    print(cy_preds)
    model = GraphModel(model_graph=model_graph,
                       encoder=Propagator(ProbEncoderFromEmb),
                       predictor=Propagator(ProbPredictor),
                       predictor_in_embedding=0,
                       predictor_in_exogenous=0,
                       annotations=annotations,
                       input_size=x.shape[1])
    inference_train = KnownGraphInference(model=model)
    cy_preds = inference_train.query(x)
    print(cy_preds)

    # CGM
    model = LearnedGraphModel(model_graph=COSMOGraphLearner,
                              exogenous=Propagator(ExogEncoder, embedding_size=7),
                              encoder=Propagator(ProbEncoderFromExog),
                              predictor=Propagator(HyperLinearPredictor, embedding_size=11),
                              annotations=annotations,
                              predictor_in_embedding=0,
                              predictor_in_exogenous=7*2,
                              input_size=x.shape[1])
    inference_train = UnknownGraphInference(model=model)
    c_encoder, c_predictor = inference_train.query(x, c)
    print(c_encoder)
    print(c_predictor)
    model = LearnedGraphModel(model_graph=COSMOGraphLearner,
                              exogenous=Propagator(ExogEncoder, embedding_size=7),
                              encoder=Propagator(ProbEncoderFromExog),
                              predictor=Propagator(MixProbExogPredictor),
                              annotations=annotations,
                              predictor_in_embedding=0,
                              predictor_in_exogenous=7,
                              input_size=x.shape[1])
    inference_train = UnknownGraphInference(model=model)
    c_encoder, c_predictor = inference_train.query(x, c)
    print(c_encoder)
    print(c_predictor)

    # CEM
    model = BipartiteModel(task_names=['c', 'e'],
                           exogenous=Propagator(ExogEncoder, embedding_size=7),
                           encoder=Propagator(ProbEncoderFromExog),
                           predictor=Propagator(MixProbExogPredictor),
                           annotations=annotations,
                           predictor_in_embedding=0,
                           predictor_in_exogenous=7,
                           input_size=x.shape[1])
    inference_test = KnownGraphInference(model=model)
    cy_pred = inference_test.query(x)

    # CBM
    model = BipartiteModel(task_names=['c', 'e'],
                           exogenous=Propagator(ExogEncoder, embedding_size=7),
                           encoder=Propagator(ProbEncoderFromExog),
                           predictor=Propagator(HyperLinearPredictor, embedding_size=11),
                           annotations=annotations,
                           predictor_in_embedding=0,
                           predictor_in_exogenous=7*2,
                           input_size=x.shape[1])
    inference_test = KnownGraphInference(model=model)
    cy_pred = inference_test.query(x)
    model = BipartiteModel(task_names=['c', 'e'],
                           encoder=Propagator(ProbEncoderFromEmb),
                           predictor=Propagator(ProbPredictor),
                           annotations=annotations,
                           predictor_in_embedding=0,
                           predictor_in_exogenous=0,
                           input_size=x.shape[1])
    inference_test = KnownGraphInference(model=model)
    cy_pred = inference_test.query(x)

    print(cy_pred)


if __name__ == "__main__":
    main()
