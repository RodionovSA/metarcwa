# src/solver/solver.py
# RCWA Solver class
from typing import List, Union, Tuple

from src.solver.config import Config
from src.model import Model
from src.solver.tvf import TVF
from src.solver.python import PySolverConfig, PySolver

class Solver:
    """ 
    RCWA Solver class.
    Prepares the model(s) and configuration for simulation 
    and initializes the appropriate solver engine.
    
    """
    def __init__(self, 
                 models: Union["Model", List["Model"]],
                 cfg: "Config" = Config()):
        """
        Parameters
        ----------
        models : Union[Model, List[Model]]
            RCWA models to solve. If there is a few models - it will perform batch simulation.
            For multiple models, they must share the same backend and lattice.
        cfg : Config
            Configuration parameters for the solver.
        """
        self.models = self._init_validation(models, cfg)
        self.cfg = cfg
        self.tvf = self._init_tvf()
        self.solver_engine = self._init_solver_engine()
        
    @property
    def backend(self):
        """ Return the computational backend. """
        return self.models[0].backend
    
    @property
    def lattice(self):
        """ Return the lattice object. """
        return self.models[0].lattice
    
    @property
    def source(self):
        """ Return the source object. """
        return self.models[0].source
        
    def solve(self):
        inputs = self._prepare_solver_inputs()
        results = self.solver_engine.solve(inputs)
        return results
        
    def _init_tvf(self) -> TVF:
        """ Initialize Tangent Vector Fields (TVF) optimizer if needed. """
        if self.cfg.factorization is not None:
            tvf = TVF(backend=self.backend,
                      lattice=self.lattice,
                      method=self.cfg.factorization,
                      optimizer=self.cfg.tvf_optimizer)
            return tvf
        return None
    
    def _prepare_tvf(self) -> List:
        """ Prepare TVF for the model if applicable. """
        if self.tvf is None:
            return [
                [None for _ in model.layers]
                for model in self.models
            ]
        
        tvf_all_models = []
        for model in self.models:
            tvf_layers = []

            for layer in model.layers:
                if layer.is_homogeneous(self.backend, self.lattice):
                    tvf_layers.append(None)
                    continue

                bitmap = layer.bitmap(self.backend, self.lattice)
                bitmap = self.backend.unsqueeze(bitmap, dim=0)  # (1, Nx, Ny)
                tx, ty = self.tvf.compute(bitmap, 
                                          alpha=self.cfg.tvf_alpha, 
                                          beta=self.cfg.tvf_beta, 
                                          gamma=self.cfg.tvf_gamma, 
                                          steps=self.cfg.tvf_steps)
                tvf_layers.append((tx, ty))
                
            tvf_all_models.append(tvf_layers)
                
        return tvf_all_models
    
    def _init_solver_engine(self):
        """ Initialize the solver engine based on configuration. """
        if self.cfg.solver.lower() == "python":
            solver_cfg = PySolverConfig(
                inverse_matrix_method=self.cfg.inverse_matrix_method,
                factorization=self.cfg.factorization,
                modes_solver=self.cfg.modes_solver,
                hsimplify=self.cfg.hsimplify,
                smlayer=self.cfg.smlayer
            )
           
            solver_engine = PySolver(solver_cfg)
        else:
            raise ValueError(f"Unknown solver engine: {self.engine}")       
        
        return solver_engine
    
    def _prepare_solver_inputs(self, 
                               inc_field: List[Tuple[Tuple[int, int], complex, complex]]) -> dict:
        """ Prepare inputs for the solver engine. """
        
        # Validation of inc_field
        if not isinstance(inc_field, list):
            raise TypeError("inc_field must be a list of incident field specifications per model.")
        if any(not isinstance(field, tuple) for field in inc_field):
            raise TypeError("Each element of inc_field must be a list of incident field specifications.")
        if any(len(field) != 3 for field in inc_field):
            raise ValueError("Each incident field specification must be a tuple of (mn, s, p).")
        if any(
            not isinstance(mn, tuple) or len(mn) != 2
            for mn, s, p in inc_field
        ):
            raise TypeError("Each mn in incident field specification must be a tuple of (m, n).")
        
        # Input field
        input_fields = []
        for idx, field in enumerate(inc_field):
            input_field = self.source.plane_wave_field(self.backend, self.lattice, 
                                                       self.models[0].n_inc, 
                                                       field[0], field[1], field[2])  # (Ex, Ey, Hx, Hy), each [wvl, theta, phi, 2M+1, 2N+1]
            input_fields.append(input_field)
        
        # Prepare TVF for all models
        tvf_layers_all_models = self._prepare_tvf() # List of List of TVF per layer
        
        # Get Kx and Ky from the first model
        Kx, Ky = self.models[0].Kxy  # Shared for all models (wvl, theta, phi, 2M+1, 2N+1)
        
        # Prepare k0 
        k0 = self.models[0].k0  # (wvl,)
        
        models_data = []
        for model_idx, model in enumerate(self.models):
            tvf_layers = tvf_layers_all_models[model_idx]
            layers_data = []
            for layer_idx, layer in enumerate(model.layers):
                tvf = tvf_layers[layer_idx]
                layer_data = self._prepare_layer_data(layer, tvf)
                layers_data.append(layer_data)
            
            models_data.append(layers_data)
        
        return {
            "models": models_data,
            "input_fields": input_fields,
            "Kx": Kx,
            "Ky": Ky,
            "k0": k0
        }
    
    def _prepare_layer_data(self, layer, tvf) -> dict:
        """ Prepare layer data for the solver. """
        d = layer.thickness
        epsilon = layer.epsilon_mn(self.backend, self.lattice,
                                   closed_form=self.cfg.closed_form,
                                   inverse = False, regularized = False,
                                   regularization = self.cfg.inverse_regularization) # (wvl, 1, 1, 4M+1, 4N+1) or (wvl, 1, 3, 4M+1, 4N+1) or (wvl, 3, 3, 4M+1, 4N+1)
        
        epsilon_inv = layer.epsilon_mn(self.backend, self.lattice,
                                       closed_form=self.cfg.closed_form,
                                       inverse = True, regularized = True,
                                       regularization = self.cfg.inverse_regularization) # (wvl, 1, 1, 4M+1, 4N+1) or (wvl, 1, 3, 4M+1, 4N+1) or (wvl, 3, 3, 4M+1, 4N+1)
        
        mu = layer.mu_mn(self.backend, self.lattice,
                        closed_form=self.cfg.closed_form,
                        inverse = False, regularized = False,
                        regularization = self.cfg.inverse_regularization) # (wvl, 1, 1, 4M+1, 4N+1) or (wvl, 1, 3, 4M+1, 4N+1) or (wvl, 3, 3, 4M+1, 4N+1)

        mu_inv = layer.mu_mn(self.backend, self.lattice,
                            closed_form=self.cfg.closed_form,
                            inverse = True, regularized = True,
                            regularization = self.cfg.inverse_regularization) # (wvl, 1, 1, 4M+1, 4N+1) or (wvl, 1, 3, 4M+1, 4N+1) or (wvl, 3, 3, 4M+1, 4N+1)
        
        return {
            "thickness": d,
            "epsilon": epsilon,
            "epsilon_inv": epsilon_inv,
            "mu": mu,
            "mu_inv": mu_inv,
            "tvf": tvf,
            "is_homogeneous": layer.is_homogeneous(self.backend, self.lattice),
            "is_magnetic": layer.is_magnetic,
            "type": layer.type
        }

    def _init_validation(self, models, cfg) -> List["Model"]:
        if not isinstance(models, Model) and not (isinstance(models, list) and all(isinstance(m, Model) for m in models)):
            raise TypeError("model must be an instance of Model class or a list of Model instances.")
        if not isinstance(cfg, Config):
            raise TypeError("cfg must be an instance of Config class.")
        
        if isinstance(models, Model):
            models = [models]
        
        # Check same backend and lattice for all models
        if isinstance(models, list):
            backend = models[0].backend
            lattice = models[0].lattice
            source = models[0].source
            inc_layer = models[0].layers[0]
            for i, model in enumerate(models):
                if model.backend != backend:
                    raise ValueError(f"All models must have the same backend. Model at index {i} has a different backend.")
                if model.lattice != lattice:
                    raise ValueError(f"All models must have the same lattice. Model at index {i} has a different lattice.")
                if model.source != source:
                    raise ValueError(f"All models must have the same source. Model at index {i} has a different source.")
                if model.layers[0] != inc_layer:
                    raise ValueError(f"All models must have the same incident layer. Model at index {i} has a different incident layer.")
                
        return models