import sys
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm

class CustomProgressBar(TQDMProgressBar):
    def __init__(self, enable_val=False, *args, **kwargs):
        self.enable_validation_bar=enable_val
        super().__init__(*args, **kwargs)
        
    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=not self.enable_validation_bar, #self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar


    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items.pop("loss", None)
        return items