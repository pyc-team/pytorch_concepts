import torch

from ...nn.base.layer import BaseEncoderLayer, BasePredictorLayer


class Propagator(torch.nn.Module):
    def __init__(self,
                 module_cls: type[torch.nn.Module],  # Stores the class reference
                 *module_args,
                 **module_kwargs):
        super().__init__()

        # Store the module class and any additional keyword arguments
        self._module_cls = module_cls
        self._module_args = module_args
        self._module_kwargs = module_kwargs

        # The actual module is initially None.
        # It MUST be a torch.nn.Module or ModuleList/Sequential, not a lambda.
        self.module = None

    def build(self,
              in_object,
              out_annotations: 'Annotations',  # Assuming Annotations is a defined type
              ) -> torch.nn.Module:
        """
        Constructor method to instantiate the underlying module with required arguments.
        """
        if self.module is not None:
            # Optional: Add logic to re-initialize or raise an error if already built
            print("Warning: Propagator module is being rebuilt.")

        # Instantiate the module using the stored class and kwargs
        # The module is instantiated with the provided arguments
        if issubclass(self._module_cls, BaseEncoderLayer):
            self.module = self._module_cls(
                in_features=in_object,
                out_annotations=out_annotations,
                *self._module_args,
                **self._module_kwargs
            )
        elif issubclass(self._module_cls, BasePredictorLayer):
            self.module = self._module_cls(
                in_contracts=in_object,
                out_annotations=out_annotations,
                *self._module_args,
                **self._module_kwargs
            )

        # Crucial for PyTorch: Check if the module is properly registered
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("The instantiated module is not a torch.nn.Module.")

        return self.module

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass calls the instantiated module.
        """
        if self.module is None:
            raise RuntimeError(
                "Propagator module not built. Call .build(in_features, annotations) first."
            )

        # Forward calls the *instantiated* module instance
        return self.module(x, *args, **kwargs)
