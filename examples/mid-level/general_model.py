import torch
from torch import nn

from torch_concepts import ConceptTensor, Annotations, AxisAnnotation, AnnotatedAdjacencyMatrix
from torch_concepts.nn import ExogEncoder, ProbPredictor, ProbEncoder, BipartiteModel, Propagator, GraphModel, \
    COSMOGraphLearner, LearnedGraphModel, BaseGraphLearner, ProbEmbEncoder, MixProbEmbPredictor, HyperNetLinearPredictor
from torch_concepts.nn.modules.inference.forward import KnownGraphInference, UnknownGraphInference


def main():
    n_concepts = 5

    x = torch.randn(100, 13)
    concept_embs = torch.ones(100, n_concepts, 7) * 10       # embs
    concept_probs = torch.ones(100, n_concepts) * 5        # probs
    residuals = torch.ones(100, n_concepts) * -1

    annotations = Annotations({1: AxisAnnotation(('a', 'b', 'c', 'd', 'e'))})

    c = ConceptTensor(annotations, concept_probs)

    # FIXME: there is something wrong in the init predictors, we may need to change the predictor propagator into a residual layer
    model_graph = AnnotatedAdjacencyMatrix(torch.tensor([[0, 1, 0, 0, 1],
                                                         [0, 0, 1, 0, 0],
                                                         [0, 0, 0, 1, 0],
                                                         [0, 0, 0, 0, 1],
                                                         [0, 0, 0, 0, 0]]).float(),
                                           annotations)
    # C2BM. FIXME: check layers are initialized correctly inside the model
    model = GraphModel(model_graph=model_graph,
                       exogenous=Propagator(ExogEncoder, embedding_size=7),
                       encoder=Propagator(ProbEncoder, exogenous=True),
                       predictor=Propagator(HyperNetLinearPredictor),
                       annotations=annotations,
                       input_size=x.shape[1])
    inference_train = KnownGraphInference(model=model)
    cy_preds = inference_train.query(x)
    print(cy_preds)

    # CGM
    model = LearnedGraphModel(model_graph=COSMOGraphLearner,
                              encoder=Propagator(ProbEmbEncoder, embedding_size=7),
                              predictor=Propagator(MixProbEmbPredictor),
                              annotations=annotations,
                              input_size=x.shape[1])
    inference_train = UnknownGraphInference(model=model)
    c_encoder, c_predictor = inference_train.query(x, c)
    print(c_encoder)
    print(c_predictor)

    # CEM
    model = BipartiteModel(task_names=['c', 'e'],
                           encoder=Propagator(ProbEmbEncoder, embedding_size=7),
                           predictor=Propagator(MixProbEmbPredictor),
                           annotations=annotations,
                           input_size=x.shape[1])
    inference_test = KnownGraphInference(model=model)
    cy_pred = inference_test.query(x)

    # CBM
    model = BipartiteModel(task_names=['c', 'e'],
                           encoder=Propagator(ProbEncoder),
                           predictor=Propagator(ProbPredictor),
                           annotations=annotations,
                           input_size=x.shape[1])
    inference_test = KnownGraphInference(model=model)
    cy_pred = inference_test.query(x)

    print(cy_pred)


if __name__ == "__main__":
    main()
