import os
from loguru import logger
from rich.logging import RichHandler
from zairachem.base.vars import BASE_DIR, LOGGING_FILE

ROTATION = "10 MB"
RETENTION = 5
# Full module:function:line context makes file logs traceable; backtrace=True records
# tracebacks for logged exceptions; diagnose=False avoids leaking variable values
# (e.g. SMILES) into the log.
_FILE_FORMAT = (
  "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
)

logger.remove()

logger.level("DEBUG", color="<cyan><bold>")
logger.level("INFO", color="<blue><bold>")
logger.level("WARNING", color="<white><bold><bg yellow>")
logger.level("ERROR", color="<white><bold><bg red>")
logger.level("CRITICAL", color="<white><bold><bg red>")
logger.level("SUCCESS", color="<black><bold><bg green>")


class Logger:
  def __init__(self):
    self.logger = logger
    self._console = None
    self._file = None
    self.configure()

  def configure(self):
    # Re-assert our sinks on the shared, process-global loguru logger. Third-party
    # packages (e.g. lazyqsar, isaura) call logger.remove() at import time, which wipes
    # ALL handlers — including ours — leaving zairachem silent. Call this after imports
    # (e.g. at CLI startup) so logging works regardless of import order.
    self.logger.remove()
    self._file = None
    self._console = None
    self._log_to_file()
    self._log_to_console()

  def _log_to_file(self):
    self._file = self.logger.add(
      os.path.join(BASE_DIR, LOGGING_FILE),
      level="DEBUG",
      rotation=ROTATION,
      retention=RETENTION,
      format=_FILE_FORMAT,
      backtrace=True,
      diagnose=False,
    )

  def _log_to_console(self):
    if self._console is None:
      rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        log_time_format="%H:%M:%S",
        show_path=False,
        show_level=True,
      )
      self._console = self.logger.add(
        rich_handler,
        format="{message}",
        colorize=True,
      )

  def _unlog_from_console(self):
    if self._console is not None:
      try:
        self.logger.remove(self._console)
      except Exception:
        pass
      self._console = None

  def set_verbosity(self, verbose):
    if verbose:
      self._log_to_console()
    else:
      self._unlog_from_console()

  def debug(self, text):
    self.logger.debug(text)

  def info(self, text):
    self.logger.info(text)

  def warning(self, text):
    self.logger.warning(text)

  def error(self, text):
    self.logger.error(text)

  def critical(self, text):
    self.logger.critical(text)

  def success(self, text):
    self.logger.success(text)

  def exception(self, text):
    # Log an error together with the active exception's full traceback.
    self.logger.opt(exception=True).error(text)


logger = Logger()


def setup_logging():
  """Re-assert zairachem's log sinks. Call once after all imports (e.g. at CLI startup),
  since third-party packages remove loguru handlers when they are imported."""
  logger.configure()
