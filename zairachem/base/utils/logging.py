import os
from loguru import logger
from rich.logging import RichHandler
from zairachem.base.vars import BASE_DIR, LOGGING_FILE

ROTATION = "10 MB"

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
    self._log_to_file()
    self._log_to_console()

  def _log_to_file(self):
    self._file = self.logger.add(
      os.path.join(BASE_DIR, LOGGING_FILE),
      rotation=ROTATION,
      format="{time:YYYY-MM-DD HH:mm:ss} | {level.name:^8} | {message}",
    )

  def _log_to_console(self):
    if self._console is None:
      rich_handler = RichHandler(
        rich_tracebacks=True, markup=True, log_time_format="%H:%M:%S", show_path=False
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


logger = Logger()
