import logging


def init_logging_configs(debugging: bool = False) -> None:
    """Add preinformation here"""
    if debugging:
        logging.basicConfig(
            format='%(asctime)s | %(levelname)s | %(filename)s | %(message)s',
            level=logging.DEBUG
        )
        logging.info('☢️ Debugging mode!!!')
    else:
        logging.basicConfig(
            format='%(asctime)s | %(levelname)s | %(message)s',
            level=logging.INFO
        )