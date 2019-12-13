from . import submission
from .submission.run_context import RunContext
from .submission.submit import (
    PathType,
    SubmitConfig,
    SubmitTarget,
    convert_path,
    get_path_from_template,
    make_run_dir_path,
    submit_run,
)
from .util import EasyDict

submit_config: SubmitConfig = None  # Package level variable for SubmitConfig which is only valid when inside the run function.
