from loguru import logger
import sys

logger.remove()

log_format = (
    "<g>{time:MM-DD HH:mm:ss}</g> <lvl>{level:<9}</lvl>| {file}:{line} | {message}"
)

logger.add(sys.stdout, format=log_format, backtrace=True, diagnose=True)