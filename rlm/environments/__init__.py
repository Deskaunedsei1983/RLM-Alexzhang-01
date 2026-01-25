from typing import Any, Literal

from rlm.environments.base_env import BaseEnv, SupportsPersistence
from rlm.environments.local_repl import LocalREPL
from rlm.environments.secure_repl import SecureLocalREPL, SecurityViolationError

__all__ = [
    "BaseEnv",
    "LocalREPL",
    "SecureLocalREPL",
    "SecurityViolationError",
    "SupportsPersistence",
    "get_environment",
]


def get_environment(
    environment: Literal["local", "local_secure", "modal", "docker", "daytona", "prime"],
    environment_kwargs: dict[str, Any],
) -> BaseEnv:
    """
    Routes a specific environment and the args (as a dict) to the appropriate environment if supported.
    Currently supported environments: ['local', 'local_secure', 'modal', 'docker', 'daytona', 'prime']

    Note: 'local_secure' adds additional security checks before code execution.
    """
    if environment == "local":
        return LocalREPL(**environment_kwargs)
    elif environment == "local_secure":
        return SecureLocalREPL(**environment_kwargs)
    elif environment == "modal":
        from rlm.environments.modal_repl import ModalREPL

        return ModalREPL(**environment_kwargs)
    elif environment == "docker":
        from rlm.environments.docker_repl import DockerREPL

        return DockerREPL(**environment_kwargs)
    elif environment == "daytona":
        from rlm.environments.daytona_repl import DaytonaREPL

        return DaytonaREPL(**environment_kwargs)
    elif environment == "prime":
        from rlm.environments.prime_repl import PrimeREPL

        return PrimeREPL(**environment_kwargs)
    else:
        raise ValueError(
            f"Unknown environment: {environment}. "
            f"Supported: ['local', 'local_secure', 'modal', 'docker', 'daytona', 'prime']"
        )
