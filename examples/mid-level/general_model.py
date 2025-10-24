import torch
from torch import nn

from torch_concepts import ConceptTensor, Annotations, AxisAnnotation
from torch_concepts.nn import LinearPredictorLayer, LinearEncoderLayer, BipartiteModel, Propagator, GraphModel, \
    COSMOGraphLearner, LearnedGraphModel, BaseGraphLearner
from torch_concepts.nn.modules.inference.forward import KnownGraphInference, UnknownGraphInference


def main():
    n_concepts = 5

    x = torch.randn(100, 13)
    concept_embs = torch.ones(100, n_concepts, 7) * 10       # embs
    concept_probs = torch.ones(100, n_concepts) * 5        # probs
    residuals = torch.ones(100, n_concepts) * -1

    annotations = Annotations({1: AxisAnnotation(('a', 'b', 'c', 'd', 'e'))})

    c = ConceptTensor(annotations, concept_probs)
    model = LearnedGraphModel(model_graph=COSMOGraphLearner,
                              encoder=Propagator(LinearEncoderLayer),
                              predictor=Propagator(LinearPredictorLayer),
                              annotations=annotations,
                              input_size=x.shape[1])
    inference_train = UnknownGraphInference(model=model)
    c_encoder, c_predictor = inference_train.query(x, c)
    known_graph_model = model.get_model_known_graph()
    inference_test = KnownGraphInference(model=known_graph_model)
    cy_pred = inference_test.query(x)

    print(known_graph_model.model_graph.data)
    print(c_encoder.concept_probs[0])
    print(c_predictor.concept_probs[0])
    print(cy_pred.concept_probs[0])

    model = BipartiteModel(task_names=['c', 'e'],
                           encoder=Propagator(LinearEncoderLayer),
                           predictor=Propagator(LinearPredictorLayer),
                           annotations=annotations,
                           input_size=x.shape[1])
    inference_test = KnownGraphInference(model=model)
    cy_pred = inference_test.query(x)

    print(cy_pred)




if __name__ == "__main__":
    main()
