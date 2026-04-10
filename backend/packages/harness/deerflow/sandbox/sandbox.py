from abc import ABC, abstractmethod
from dataclasses import dataclass

from deerflow.sandbox.search import GrepMatch


@dataclass(frozen=True)
class FileStat:
    """Metadata returned by stat_file — mirrors stat(2) semantics.

    All fields are derived from os.stat() without reading file content,
    making this efficient for existence checks, size guards, and freshness
    comparisons without the cost of a full read.
    """

    path: str           # The virtual container path that was queried
    exists: bool        # False when the path does not exist
    is_file: bool
    is_dir: bool
    size: int           # Bytes; 0 when exists is False
    mtime: float        # Last-modified time as a UNIX timestamp; 0.0 when exists is False
    readable: bool      # True when the process has read permission
    writable: bool      # True when the path is NOT under a read-only mount AND process has write permission


class Sandbox(ABC):
    """Abstract base class for sandbox environments"""

    _id: str

    def __init__(self, id: str):
        self._id = id

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    def execute_command(self, command: str) -> str:
        """Execute bash command in sandbox.

        Args:
            command: The command to execute.

        Returns:
            The standard or error output of the command.
        """
        pass

    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read the content of a file.

        Args:
            path: The absolute path of the file to read.

        Returns:
            The content of the file.
        """
        pass

    @abstractmethod
    def list_dir(self, path: str, max_depth=2) -> list[str]:
        """List the contents of a directory.

        Args:
            path: The absolute path of the directory to list.
            max_depth: The maximum depth to traverse. Default is 2.

        Returns:
            The contents of the directory.
        """
        pass

    @abstractmethod
    def write_file(self, path: str, content: str, append: bool = False) -> None:
        """Write content to a file.

        Args:
            path: The absolute path of the file to write to.
            content: The text content to write to the file.
            append: Whether to append the content to the file. If False, the file will be created or overwritten.
        """
        pass

    @abstractmethod
    def glob(self, path: str, pattern: str, *, include_dirs: bool = False, max_results: int = 200) -> tuple[list[str], bool]:
        """Find paths that match a glob pattern under a root directory."""
        pass

    @abstractmethod
    def grep(
        self,
        path: str,
        pattern: str,
        *,
        glob: str | None = None,
        literal: bool = False,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> tuple[list[GrepMatch], bool]:
        """Search for matches inside text files under a directory."""
        pass

    @abstractmethod
    def update_file(self, path: str, content: bytes) -> None:
        """Update a file with binary content.

        Args:
            path: The absolute path of the file to update.
            content: The binary content to write to the file.
        """
        pass

    @abstractmethod
    def stat_file(self, path: str) -> "FileStat":
        """Return file-system metadata for *path* without reading its content.

        Mirrors the semantics of the POSIX stat(2) syscall: the caller learns
        whether the path exists, its size, and when it was last modified — all
        without the cost of a full read.  Use this to:
          - Guard large reads (check size before reading)
          - Detect stale cache entries (compare mtime)
          - Confirm a write succeeded (exists + size > 0)

        Args:
            path: The absolute path (container or host) to inspect.

        Returns:
            A FileStat dataclass.  ``exists=False`` is returned instead of
            raising when the path does not exist, so callers can branch on
            existence without a try/except.
        """
        pass
