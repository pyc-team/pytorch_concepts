from torchmetrics import Metric

# class ConceptCausalEffect(Metric):
#     """
#     Concept Causal Effect (CaCE) is a metric that measures the causal effect between concept pairs
#     or between a concept and the task.
#     NOTE: only works on binary concepts.
#     """
#     def __init__(self):
#         super().__init__()
#         self.add_state("preds_do_1", default=torch.tensor(0.), dist_reduce_fx="sum")
#         self.add_state("preds_do_0", default=torch.tensor(0.), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, 
#                preds_do_1: torch.Tensor, 
#                preds_do_0: torch.Tensor):
#         _check_same_shape(preds_do_1, preds_do_0)
#         # expected value = 1*p(output=1|do(1)) + 0*(1-p(output=1|do(1))
#         self.preds_do_1 += preds_do_1[:,1].sum()
#         # expected value = 1*p(output=1|do(0)) + 0*(1-p(output=1|do(0))
#         self.preds_do_0 += preds_do_0[:,1].sum()
#         self.total += preds_do_1.size()[0]

#     def compute(self):
#         return (self.preds_do_1.float() / self.total) - (self.preds_do_0.float()  / self.total)
